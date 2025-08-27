import os, sys, textwrap, warnings, logging, platform
from pathlib import Path
from typing import List
from dotenv import load_dotenv

import numpy as np
import torch, chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction

from sentence_transformers import CrossEncoder
from sentence_transformers.util import cos_sim
from rank_bm25 import BM25Okapi

from transformers import CLIPModel, CLIPProcessor, AutoTokenizer   # <— tokenizer added
from together import Together

# Env
load_dotenv(override=True)

# Together client (uses TOGETHER_API_KEY)
together_client = Together()

# Warnings/logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

CHROMA_DIR = Path("./chroma_db")

# Models & Settings
TEXT_MODEL     = "BAAI/bge-base-en-v1.5"                 # embeddings model
RERANK_MODEL   = "BAAI/bge-reranker-base"               # local cross-encoder reranker
CLIP_MODEL     = "openai/clip-vit-base-patch32"         # image retrieval
TOGETHER_MODEL = "meta-llama/Llama-3.2-3B-Instruct-Turbo"

TOP_K_MMR    = 40
TOP_K_BM25   = 10
TOP_K_IMAGE  = 2
TOP_K_RERANK = 4

MAX_EMB_TOKENS = 512
CHUNK_OVERLAP  = 32

# Optimize for MacOS
IS_MAC = platform.system() == "Darwin"
if torch.cuda.is_available():
    device = "cuda"
elif IS_MAC and torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    try: torch.set_float32_matmul_precision("medium")
    except: pass
else:
    device = "cpu"
try: torch.set_num_threads(min(8, os.cpu_count() or 4))
except: pass

print("Loading models …")

# Tokenizer for counting/splitting tokens (no model weights)
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)

reranker = CrossEncoder(RERANK_MODEL, device=device)

clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL, use_fast=True)
print("Models ready\n")

# Embedding helpers
def _text_to_ids(text: str):
    return tokenizer.encode(text, add_special_tokens=False)

def _ids_to_text(ids):
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def _split_ids(ids, max_len=MAX_EMB_TOKENS, overlap=CHUNK_OVERLAP):
    if len(ids) <= max_len:
        yield ids
        return
    step = max_len - overlap
    for start in range(0, len(ids), step):
        window = ids[start:start+max_len]
        if not window: break
        yield window

def _embed_many_texts(texts: List[str]) -> List[List[float]]:
    resp = together_client.embeddings.create(model=TEXT_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def embed_text_token_safe(text: str) -> List[float]:
    ids = _text_to_ids(text)
    chunks = list(_split_ids(ids))
    if len(chunks) == 1:
        return _embed_many_texts([text])[0]
    chunk_texts = [_ids_to_text(c) for c in chunks]
    chunk_vecs = _embed_many_texts(chunk_texts)
    return (np.mean(np.array(chunk_vecs, dtype=np.float32), axis=0)).tolist()

# Chroma embedding function
class SciEmbedding(EmbeddingFunction):
    def __init__(self): self.dim = 768
    def name(self): return TEXT_MODEL
    def dimensions(self): return self.dim
    def __call__(self, texts):
        return [embed_text_token_safe(t) for t in texts]

# CLIP text EF
class CLIPTextEF(EmbeddingFunction):
    def __init__(self): self.proc, self.model = clip_proc, clip_model
    def name(self): return f"clip-text-{CLIP_MODEL}"
    def dimensions(self): return 512
    def __call__(self, texts):
        with torch.no_grad():
            inp = self.proc(text=texts, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            self.model.to(device)
            return self.model.get_text_features(**inp).cpu().numpy().tolist()
clip_text_ef = CLIPTextEF()

# Retrieval utils
def mmr_select(query_emb, doc_embs, docs, metas, k=20, weight=0.6):
    query_emb = np.asarray(query_emb, dtype=np.float32)
    doc_embs  = np.asarray(doc_embs,  dtype=np.float32)
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

def shorten(t, w=200): return textwrap.shorten(" ".join(t.split()), width=w)
def label(meta, idx):
    from pathlib import Path as _P
    if meta.get("image_path") and meta.get("caption"):
        src = _P(meta["image_path"]).name + " (image)"
    else:
        src = _P(meta.get("source", meta.get("image_path","unknown"))).name
    flags = [k for k in ("table_md","table_json","table_row","chartqa","ocr") if meta.get(k)]
    return f"Source {idx}: {src}{' ('+'/'.join(flags)+')' if flags else ''}"

def rerank_ctx(query: str, docs: List[str], metas: List[dict], keep=TOP_K_RERANK):
    scores = reranker.predict([(query, d if d else " ") for d in docs])
    best   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep]
    return [docs[i] for i in best], [metas[i] for i in best]

def embed_query_together(text: str) -> np.ndarray:
    """Use same token-safe embedding for the query."""
    return np.array(embed_text_token_safe(text), dtype=np.float32)

# Main
def main():
    client_chroma = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))

    try:
        txt_col = client_chroma.get_collection("text", embedding_function=SciEmbedding())
    except Exception:
        txt_col = None
    if not txt_col or txt_col.count() == 0:
        print("ERROR: No text vectors found. Run ingest.py first to build the Chroma DB.")
        sys.exit(1)

    try:
        img_col = client_chroma.get_collection("images", embedding_function=clip_text_ef)
    except Exception:
        img_col = None  # optional

    corpus_docs  = txt_col.get()["documents"]
    corpus_metas = txt_col.get()["metadatas"]
    bm25 = BM25Store([d.lower() for d in corpus_docs])

    print("Ready: ask me anything (Ctrl-C to quit)\n")

    while True:
        try:
            q = input("User: ").strip()
            if not q:
                continue

            txt = txt_col.query(
                query_texts=[q],
                n_results=60,
                include=['documents', 'metadatas', 'embeddings']
            )
            docs_raw   = txt["documents"][0]
            metas_raw  = txt["metadatas"][0]
            embs_raw   = np.array(txt["embeddings"][0])

            # MMR with token-safe query embedding
            query_emb = embed_query_together(q)
            docs_mmr, metas_mmr = mmr_select(query_emb, embs_raw, docs_raw, metas_raw, k=TOP_K_MMR, weight=0.6)

            # BM25 lexical
            idxs = bm25.query(q.lower(), k=TOP_K_BM25)
            docs_bm  = [corpus_docs[i]  for i in idxs]
            metas_bm = [corpus_metas[i] for i in idxs]

            # Merge + rerank
            docs_all  = docs_mmr + docs_bm
            metas_all = metas_mmr + metas_bm
            docs, metas = rerank_ctx(q, docs_all, metas_all, keep=TOP_K_BM25)

            # Optional CLIP image retrieval
            if img_col and img_col.count() > 0:
                with torch.no_grad():
                    clip_vec = CLIPTextEF()([q])[0]
                img = img_col.query(query_embeddings=[clip_vec], n_results=TOP_K_IMAGE)
                docs  += img["documents"][0]
                metas += img["metadatas"][0]

            # Show context
            print("\n🔎 Retrieved context:")
            for i, (d, m) in enumerate(zip(docs, metas), 1):
                snippet = "<image>" if m.get("image_path") else shorten(d)
                print(f"  {i}. {label(m, i)} — {snippet}")
            print("")

            # Build prompt
            blocks = []
            for i, (d, m) in enumerate(zip(docs, metas), 1):
                content = f"<image at {m['image_path']}>" if m.get("image_path") else d
                blocks.append(f"{label(m, i)}\n{content}")
            prompt = (
                "You are a helpful assistant. Use ONLY the sources below.\n\n"
                f"User question: {q}\n\n" + "\n\n".join(blocks) +
                "\n\nCite facts like (Source 2)."
            )

            # Together chat
            resp = together_client.chat.completions.create(
                model=TOGETHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content
            print("\nAssistant:", answer, "\n")

        except KeyboardInterrupt:
            print("\nGood Bye")
            break

if __name__ == "__main__":
    main()

