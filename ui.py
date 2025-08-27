import os, sys, textwrap, warnings, logging, platform
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv

import numpy as np
import torch, chromadb, gradio as gr
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction

from sentence_transformers import CrossEncoder
from sentence_transformers.util import cos_sim
from rank_bm25 import BM25Okapi

from transformers import CLIPModel, CLIPProcessor, AutoTokenizer
from together import Together

# Env
load_dotenv(override=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Models
TEXT_MODEL     = "BAAI/bge-base-en-v1.5"
RERANK_MODEL   = "BAAI/bge-reranker-base"
CLIP_MODEL     = "openai/clip-vit-base-patch32"
TOGETHER_MODEL = "meta-llama/Llama-3.2-3B-Instruct-Turbo"

TOP_K_MMR    = 40
TOP_K_BM25   = 10
TOP_K_IMAGE  = 2
TOP_K_RERANK = 4

MAX_EMB_TOKENS = 512
CHUNK_OVERLAP  = 32

# Optimize MacOS
IS_MAC = platform.system() == "Darwin"
if torch.cuda.is_available():
    device = "cuda"
elif IS_MAC and torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass
else:
    device = "cpu"
try:
    torch.set_num_threads(min(8, os.cpu_count() or 4))
except Exception:
    pass

# Lazy globals
_tokenizer: Optional[AutoTokenizer] = None
_reranker: Optional[CrossEncoder] = None
_clip_model: Optional[CLIPModel] = None
_clip_proc: Optional[CLIPProcessor] = None
_together: Optional[Together] = None

# Embedding helpers
def _ensure_models_loaded():
    global _tokenizer, _reranker, _clip_model, _clip_proc, _together
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL, device=device)
    if _clip_model is None or _clip_proc is None:
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
        _clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL, use_fast=True)
    if _together is None:
        _together = Together()


def _text_to_ids(text: str):
    return _tokenizer.encode(text, add_special_tokens=False)


def _ids_to_text(ids):
    return _tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def _split_ids(ids, max_len=MAX_EMB_TOKENS, overlap=CHUNK_OVERLAP):
    if len(ids) <= max_len:
        yield ids
        return
    step = max_len - overlap
    for start in range(0, len(ids), step):
        window = ids[start:start + max_len]
        if not window:
            break
        yield window


def _embed_many_texts(texts: List[str]) -> List[List[float]]:
    _ensure_models_loaded()
    resp = _together.embeddings.create(model=TEXT_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_text_token_safe(text: str) -> List[float]:
    ids = _text_to_ids(text)
    chunks = list(_split_ids(ids))
    if len(chunks) == 1:
        return _embed_many_texts([text])[0]
    chunk_texts = [_ids_to_text(c) for c in chunks]
    chunk_vecs = _embed_many_texts(chunk_texts)
    return (np.mean(np.array(chunk_vecs, dtype=np.float32), axis=0)).tolist()


class SciEmbedding(EmbeddingFunction):
    def __init__(self):
        self.dim = 768
    def name(self):
        return TEXT_MODEL
    def dimensions(self):
        return self.dim
    def __call__(self, texts):
        return [embed_text_token_safe(t) for t in texts]


class CLIPTextEF(EmbeddingFunction):
    def __init__(self):
        _ensure_models_loaded()
        self.proc, self.model = _clip_proc, _clip_model
    def name(self):
        return f"clip-text-{CLIP_MODEL}"
    def dimensions(self):
        return 512
    def __call__(self, texts):
        with torch.no_grad():
            inp = self.proc(text=texts, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            self.model.to(device)
            return self.model.get_text_features(**inp).cpu().numpy().tolist()


clip_text_ef = None

# Retrieval utils
def mmr_select(query_emb, doc_embs, docs, metas, k=20, weight=0.6):
    query_emb = np.asarray(query_emb, dtype=np.float32)
    doc_embs = np.asarray(doc_embs, dtype=np.float32)
    selected, sel_docs, sel_metas = [], [], []
    candidates = list(range(len(docs)))
    sims_query = cos_sim(torch.tensor(query_emb), torch.tensor(doc_embs)).numpy().flatten()
    while len(selected) < k and candidates:
        if not selected:
            idx = int(np.argmax(sims_query[candidates]))
        else:
            sims_selected = cos_sim(
                torch.tensor(doc_embs[candidates]),
                torch.tensor(doc_embs[selected])
            ).numpy().max(axis=1)
            mmr = weight * sims_query[candidates] - (1 - weight) * sims_selected
            idx = candidates[int(np.argmax(mmr))]
        selected.append(idx)
        sel_docs.append(docs[idx])
        sel_metas.append(metas[idx])
        candidates.remove(idx)
    return sel_docs, sel_metas


class BM25Store:
    def __init__(self, docs_lower: List[str]):
        self.model = BM25Okapi([d.split() for d in docs_lower])
    def query(self, q: str, k: int):
        idxs = self.model.get_top_n(q.split(), range(len(self.model.doc_freqs)), n=k)
        return idxs


def shorten(t, w=200):
    return textwrap.shorten(" ".join(t.split()), width=w)


def label(meta, idx):
    from pathlib import Path as _P
    if meta.get("image_path") and meta.get("caption"):
        src = _P(meta["image_path"]).name + " (image)"
    else:
        src = _P(meta.get("source", meta.get("image_path", "unknown"))).name
    flags = [k for k in ("table_md", "table_json", "table_row", "chartqa", "ocr") if meta.get(k)]
    return f"Source {idx}: {src}{' (' + '/'.join(flags) + ')' if flags else ''}"


def rerank_ctx(query: str, docs: List[str], metas: List[dict], keep=TOP_K_RERANK):
    _ensure_models_loaded()
    scores = _reranker.predict([(query, d if d else " ") for d in docs])
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep]
    return [docs[i] for i in best], [metas[i] for i in best]


def embed_query_together(text: str) -> np.ndarray:
    return np.array(embed_text_token_safe(text), dtype=np.float32)

# App state
class RAGState:
    def __init__(self):
        self.db_path: Optional[str] = None
        self.client: Optional[chromadb.Client] = None
        self.txt_col = None
        self.img_col = None
        self.corpus_docs: List[str] = []
        self.corpus_metas: List[dict] = []
        self.bm25: Optional[BM25Store] = None
        self.ready: bool = False
        self.show_context: bool = False

    def reset(self):
        self.__init__()

# Connect to Chroma DB
def connect_vectordb(db_folder: str, state: RAGState) -> Tuple[str, dict]:
    if not db_folder:
        return ("❌ Please provide a folder path to your Chroma DB.", gr.update(value=False))

    path = Path(db_folder).expanduser().resolve()
    if not path.exists():
        return (f"❌ Folder not found: {path}", gr.update(value=False))

    _ensure_models_loaded()
    global clip_text_ef
    clip_text_ef = CLIPTextEF()

    try:
        client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        try:
            txt_col = client.get_collection("text", embedding_function=SciEmbedding())
        except Exception:
            txt_col = None
        if not txt_col or txt_col.count() == 0:
            return ("❌ No text vectors found in this DB.", gr.update(value=False))

        try:
            img_col = client.get_collection("images", embedding_function=clip_text_ef)
        except Exception:
            img_col = None

        corpus_docs = txt_col.get()["documents"]
        corpus_metas = txt_col.get()["metadatas"]
        bm25 = BM25Store([d.lower() for d in corpus_docs])

        state.db_path = str(path)
        state.client = client
        state.txt_col = txt_col
        state.img_col = img_col
        state.corpus_docs = corpus_docs
        state.corpus_metas = corpus_metas
        state.bm25 = bm25
        state.ready = True

        status = f"✅ Connected to Chroma at `{path}` — text docs: {len(corpus_docs)} | images: {img_col.count() if img_col else 0}"
        return (status, gr.update(value=True))
    except Exception as e:
        return (f"❌ Failed to connect: {e}", gr.update(value=False))

# Chat handler
def chat_infer(message: str, history: List[dict], state: RAGState) -> List[dict]:
    if not state.ready:
        return history + [{"role": "assistant", "content": "Please connect to your Chroma DB first."}]

    q = (message or "").strip()
    if not q:
        return history

    txt = state.txt_col.query(
        query_texts=[q],
        n_results=60,
        include=["documents", "metadatas", "embeddings"],
    )
    docs_raw = txt["documents"][0]
    metas_raw = txt["metadatas"][0]
    embs_raw = np.array(txt["embeddings"][0])

    query_emb = embed_query_together(q)
    docs_mmr, metas_mmr = mmr_select(query_emb, embs_raw, docs_raw, metas_raw, k=TOP_K_MMR, weight=0.6)

    idxs = state.bm25.query(q.lower(), k=TOP_K_BM25)
    docs_bm = [state.corpus_docs[i] for i in idxs]
    metas_bm = [state.corpus_metas[i] for i in idxs]

    docs_all = docs_mmr + docs_bm
    metas_all = metas_mmr + metas_bm
    docs, metas = rerank_ctx(q, docs_all, metas_all, keep=TOP_K_BM25)

    if state.img_col and state.img_col.count() > 0:
        with torch.no_grad():
            clip_vec = CLIPTextEF()([q])[0]
        img = state.img_col.query(query_embeddings=[clip_vec], n_results=TOP_K_IMAGE)
        docs += img["documents"][0]
        metas += img["metadatas"][0]

    blocks = []
    lines = ["\n🔎 Retrieved context:"]
    for i, (d, m) in enumerate(zip(docs, metas), 1):
        snippet = "<image>" if m.get("image_path") else shorten(d)
        lines.append(f"  {i}. {label(m, i)} — {snippet}")
        content = f"<image at {m['image_path']}>" if m.get("image_path") else d
        blocks.append(f"{label(m, i)}\n{content}")
    context_md = "\n".join(lines)

    prompt = (
        "You are a helpful assistant. Use ONLY the sources below.\n\n"
        f"User question: {q}\n\n" + "\n\n".join(blocks) +
        "\n\nCite facts like (Source 2)."
    )

    history.append({"role": "user", "content": q})
    if state.show_context:
        history.append({"role": "assistant", "content": context_md})

    _ensure_models_loaded()
    resp = _together.chat.completions.create(
        model=TOGETHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content

    history.append({"role": "assistant", "content": "🧠 LLM response:\n" + answer})
    return history

# UI definition
css = """
#status {min-height: 38px}
"""

def build_ui():
    state = RAGState()

    with gr.Blocks(css=css, title="RAG Chat UI") as demo:
        gr.Markdown("""
        # 🧠 RAG Chat
        1) Select your Chroma DB folder and **Connect**.  
        2) Chat with your knowledge base.  
        3) Use **Clear Chat** to reset the conversation.
        """)

        with gr.Row():
            db_folder = gr.Textbox(label="Chroma DB folder", value=str(Path("./chroma_db").resolve()), scale=4)
            connect_btn = gr.Button("🔌 Connect", variant="primary", scale=1)
            connected = gr.Checkbox(label="Connected", value=False, interactive=False)
        status = gr.Markdown("", elem_id="status")

        chatbot = gr.Chatbot(height=420, type="messages")
        user_in = gr.Textbox(placeholder="Ask anything about your data…", label="Your message")
        with gr.Row():
            send_btn = gr.Button("Send")
            clear_btn = gr.ClearButton([chatbot, user_in], value="Clear Chat")
        
        show_context_toggle = gr.Checkbox(label="Show Retrieval Context", value=False)

        def _connect(db_path):
            s, ok = connect_vectordb(db_path, state)
            return s, ok

        connect_btn.click(_connect, inputs=[db_folder], outputs=[status, connected])

        def _toggle_show_context(val):
            state.show_context = val
            return

        show_context_toggle.change(_toggle_show_context, inputs=[show_context_toggle], outputs=[])

        def _respond(msg, chat_history):
            updated_history = chat_infer(msg, chat_history or [], state)
            return updated_history, ""

        send_btn.click(_respond, inputs=[user_in, chatbot], outputs=[chatbot, user_in])
        user_in.submit(_respond, inputs=[user_in, chatbot], outputs=[chatbot, user_in])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()


