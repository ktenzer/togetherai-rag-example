import os, sys, time, textwrap, mimetypes, warnings, logging, csv, io, json, platform
from pathlib import Path
from typing import List
from dotenv import load_dotenv

import numpy as np
import torch, chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction  # (used for images collection)
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor,
    BlipProcessor, BlipForConditionalGeneration,
    Pix2StructProcessor, Pix2StructForConditionalGeneration,
    AutoTokenizer,
)
import pdfplumber
from pypdf import PdfReader
import easyocr
from tqdm import tqdm
from together import Together

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document

# Env / Logging
load_dotenv(override=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

DOCS_DIR   = Path("./docs").resolve()
CHROMA_DIR = Path("./chroma_db")

# Models / Config 
TEXT_MODEL     = "BAAI/bge-base-en-v1.5"   # Together embeddings (512-token limit)
CLIP_MODEL     = "openai/clip-vit-base-patch32"
BLIP_MODEL     = "Salesforce/blip-image-captioning-base"
CHARTQA_MODEL  = "google/deplot"

MAX_EMB_TOKENS = 480
CHUNK_OVERLAP  = 32    # token overlap for long texts

# Optimize for MacOS
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

# Init models
print("Loading models …")

# Together client (uses TOGETHER_API_KEY)
together_client = Together()

# Tokenizer (for counting/splitting tokens only)
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)

# Vision / caption / chart
clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL, use_fast=True)

blip_proc  = BlipProcessor.from_pretrained(BLIP_MODEL)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)

p2s_proc  = Pix2StructProcessor.from_pretrained(CHARTQA_MODEL)
p2s_model = Pix2StructForConditionalGeneration.from_pretrained(CHARTQA_MODEL).to(device)

# OCR: EasyOCR (doesn't support MPS sadly)
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

print("Models ready\n")

# Embedding helpers
class SciEmbedding(EmbeddingFunction):
    """Schema-only (and fallback) embedding fn that matches inference.
    Not used during ingest adds because we pass embeddings explicitly."""
    def __init__(self):
        self.dim = 768  # bge-base-en-v1.5
    def name(self): return TEXT_MODEL
    def dimensions(self): return self.dim
    def __call__(self, texts):
        # Fallback path; normally unused since we supply embeddings.
        return embed_texts_token_safe_batch(list(texts))
    
def _text_to_ids(text: str):
    return tokenizer.encode(text, add_special_tokens=False)

def _ids_to_text(ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def _split_ids(ids: List[int], max_len=MAX_EMB_TOKENS, overlap=CHUNK_OVERLAP):
    """Yield max_len windows over token ids with fixed overlap."""
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
    """Single Together embeddings call for a list of texts (each ≤512 tokens)."""
    if not texts:
        return []
    resp = together_client.embeddings.create(model=TEXT_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def embed_text_token_safe(text: str) -> List[float]:
    """Token-safe: split >512-token text → embed windows → mean-pool to one vector."""
    ids = _text_to_ids(text)
    chunks = list(_split_ids(ids))
    if len(chunks) == 1:
        return _embed_many_texts([text])[0]
    chunk_texts = [_ids_to_text(c) for c in chunks]
    chunk_vecs = _embed_many_texts(chunk_texts)
    return (np.mean(np.array(chunk_vecs, dtype=np.float32), axis=0)).tolist()

def embed_texts_token_safe_batch(texts: List[str], chunk_batch_size: int = 96) -> List[List[float]]:
    """
    Faster, batched embeddings via Together.ai with token-safe chunking.
    - Splits each text into ≤512-token windows (with overlap).
    - Embeds all windows in batched API calls.
    - Mean-pools window vectors back to one vector per input text.
    """
    # tokenize & split each input into chunk strings
    chunk_texts: List[str] = []
    seg_sizes: List[int] = []
    for t in texts:
        ids = _text_to_ids(t)
        chunks = list(_split_ids(ids))
        seg_sizes.append(len(chunks))
        chunk_texts.extend([_ids_to_text(c) for c in chunks])

    if not chunk_texts:
        return []

    # embed all chunks in batches
    chunk_vecs: List[List[float]] = []
    for i in range(0, len(chunk_texts), chunk_batch_size):
        sub = chunk_texts[i:i + chunk_batch_size]
        chunk_vecs.extend(_embed_many_texts(sub))

    # mean-pool back to per-document vectors
    out: List[List[float]] = []
    cursor = 0
    for n in seg_sizes:
        vecs = chunk_vecs[cursor:cursor + n]
        cursor += n
        out.append((np.mean(np.array(vecs, dtype=np.float32), axis=0)).tolist())
    return out

# CLIP / Caption / OCR 
def clip_embed_image(path: str, device=None):
    with torch.no_grad():
        x = clip_proc(images=Image.open(path), return_tensors="pt")
        x = {k: v.to(device) for k, v in x.items()}
        clip_model.to(device)
        feats = clip_model.get_image_features(**x)
    return feats[0].cpu().numpy().tolist()

def blip_caption(path: Path, device=None) -> str:
    img = Image.open(path).convert("RGB")
    inputs = blip_proc(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    blip_model.to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=40)
    return blip_proc.decode(out[0], skip_special_tokens=True)

def chartqa_caption(path: Path, device=None) -> str:
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    img = Image.open(path).convert("RGB")
    inputs = p2s_proc(images=img, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = p2s_model.generate(**inputs, max_new_tokens=64)
    return p2s_proc.decode(out[0], skip_special_tokens=True).strip()

def easy_ocr(path: Path) -> str:
    results = ocr_reader.readtext(str(path), detail=0, paragraph=True)
    return " ".join(results).strip()

# Text splitters / loaders
def md_split(docs, chunk=800, overlap=80):
    header = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"), ("##","h2"), ("###","h3")])
    rc     = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
    out: List[Document] = []
    print("Chunking text …")
    t0 = time.time()
    for doc in docs:
        for sec in header.split_text(doc.page_content):
            for ch in rc.split_text(sec.page_content):
                out.append(Document(page_content=ch, metadata=doc.metadata))
    print(f"{len(out)} chunks ({time.time()-t0:.1f}s)\n")
    return out

def ocr_pdf(path: Path):
    print(f"Extract (PyPDF) {path.name}")
    try:
        reader = PdfReader(str(path))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        print(f"PyPDF failed on {path.name}: {e}")
        return ""

def csv_to_sentences(raw_csv: str, hdr: List[str]) -> List[str]:
    out = []
    for row in csv.reader(io.StringIO(raw_csv)):
        if row == hdr:
            continue
        out.append("Row -> " + ", ".join(f"{h.strip()}: {v.strip()}" for h, v in zip(hdr, row)))
    return out

def tables_docs(path: Path) -> List[Document]:
    docs = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    for tbl in page.extract_tables():
                        hdr, *rows = tbl
                        hdr = ["" if c is None else str(c) for c in hdr]

                        # Markdown table
                        md = "| " + " | ".join(hdr) + " |\n" + "|---" * len(hdr) + "|\n"
                        for r in rows:
                            md += "| " + " | ".join("" if c is None else str(c) for c in r) + " |\n"
                        docs.append(Document(page_content=md, metadata={"source": str(path), "page": page_num, "table_md": True}))

                        # JSON table
                        docs.append(
                            Document(
                                page_content=json.dumps({"headers": hdr, "rows": rows}, ensure_ascii=False),
                                metadata={"source": str(path), "page": page_num, "table_json": True},
                            )
                        )

                        # Row sentences
                        raw_csv = "\n".join([",".join(hdr)] + [",".join("" if c is None else str(c) for c in r) for r in rows])
                        for sent in csv_to_sentences(raw_csv, hdr):
                            docs.append(
                                Document(page_content=sent, metadata={"source": str(path), "page": page_num, "table_row": True})
                            )
                except Exception as e:
                    print(f"Skipping table extraction on {path.name} page {page_num}: {e}")
    except Exception as e:
        print(f"Failed to open {path.name} with pdfplumber: {e}")
    return docs

def load_docs():
    text, images = [], []
    for fp in DOCS_DIR.rglob("*"):
        if fp.is_dir():
            continue
        mime = mimetypes.guess_type(fp)[0] or ""

        if mime.startswith("image"):
            images.append(Document(page_content="", metadata={"image_path": str(fp)}))
            cap   = blip_caption(fp)
            chart = chartqa_caption(fp)
            ocr   = easy_ocr(fp)
            combo = " • ".join(t for t in (cap, chart, ocr) if t)
            text.append(
                Document(
                    page_content=combo,
                    metadata={"image_path": str(fp), "caption": True, "chartqa": bool(chart), "ocr": bool(ocr)},
                )
            )

        elif fp.suffix.lower() == ".pdf":
            text.append(Document(page_content=ocr_pdf(fp), metadata={"source": str(fp)}))
            text.extend(tables_docs(fp))

        elif fp.suffix.lower() in {".txt", ".md"}:
            print(f"Load {fp.name}")
            text.append(Document(page_content=fp.read_text(), metadata={"source": str(fp)}))

    print(f"Loaded {len(text)} text docs & {len(images)} images\n")
    return text, images

# Embedding functions for Chroma (images)
class CLIPTextEF(EmbeddingFunction):
    """Only used for 'images' collection to enable text→image queries."""
    def __init__(self):
        self.proc, self.model = clip_proc, clip_model
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

clip_text_ef = CLIPTextEF()

# Build Chroma stores (with progress)
def build_stores(txt_chunks: List[Document], img_docs: List[Document], client):
    """
    Batched add to Chroma with progress bars.
    - Text: compute embeddings (together.ai).
      We create the 'text' collection without an embedding_function so we can
      supply our precomputed vectors and show progress.
    - Images: batched CLIP embeddings with progress as well.
    """
    print("Embedding & writing to Chroma …")

    # Text collection: no embedding_function (we pass embeddings ourselves)
    txt_col = client.get_or_create_collection("text", embedding_function=SciEmbedding())

    # Images collection: keep text embedding_function for later text->image queries
    img_col = client.get_or_create_collection("images", embedding_function=clip_text_ef)

    # ---- Text embeddings (progress + batching) ----
    DOC_BATCH = 16
    print("")
    pbar_text = tqdm(total=len(txt_chunks), desc="Text embeddings", unit="doc")

    next_id_base = 0
    for start in range(0, len(txt_chunks), DOC_BATCH):
        batch = txt_chunks[start:start + DOC_BATCH]
        batch_docs  = [d.page_content for d in batch]
        batch_metas = [d.metadata     for d in batch]
        batch_ids   = [f"t{next_id_base + i}" for i in range(len(batch))]

        # Token-safe, batched Together embeddings (mean-pooled per doc)
        batch_vecs = embed_texts_token_safe_batch(batch_docs)

        # Flush this batch to Chroma
        txt_col.add(
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_vecs,
            ids=batch_ids,
        )

        next_id_base += len(batch)
        pbar_text.update(len(batch))

    pbar_text.close()

    # Image embeddings (progress bar and batching)
    if img_docs:
        IMG_BATCH = 64
        pbar_img = tqdm(total=len(img_docs), desc="Image embeddings", unit="img")
        next_img_id_base = 0

        for i in range(0, len(img_docs), IMG_BATCH):
            sub = img_docs[i:i + IMG_BATCH]
            sub_vecs  = [clip_embed_image(d.metadata["image_path"]) for d in sub]
            sub_metas = [d.metadata for d in sub]
            sub_ids   = [f"i{next_img_id_base + j}" for j in range(len(sub))]

            img_col.add(
                documents=[""] * len(sub),
                metadatas=sub_metas,
                embeddings=sub_vecs,
                ids=sub_ids,
            )

            next_img_id_base += len(sub)
            pbar_img.update(len(sub))

        pbar_img.close()

    print("Vector DB ready\n")
    return

# Main 
def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))

    # If vectors already exist, skip ingest (but ensure images collection exists)
    try:
        existing_txt = client.get_collection("text")
    except Exception:
        existing_txt = None

    if existing_txt and existing_txt.count() > 0:
        print(f"Found existing Chroma DB ({existing_txt.count()} text vectors) – skipping ingest\n")
        try:
            _ = client.get_collection("images", embedding_function=clip_text_ef)
        except Exception:
            _ = client.get_or_create_collection("images", embedding_function=clip_text_ef)
        return

    # Build from scratch
    print("No text vectors detected; running full ingest (OCR, split, embed) …")
    text_docs, img_docs = load_docs()
    chunks = md_split(text_docs)
    build_stores(chunks, img_docs, client)

if __name__ == "__main__":
    main()

