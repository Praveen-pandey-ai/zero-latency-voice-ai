"""Ingest pipeline: chunk documents, embed, build FAISS index and BM25 data.

Usage:
    python rag/ingest.py

This script is defensive: it only runs if `sentence-transformers` and
`faiss` are available. It reads text files from `data/` under the
project root, splits into chunks, encodes with a SentenceTransformer,
builds a FAISS index and saves the index and documents for retrieval.
"""
import os
import json
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[0]
INDEX_PATH = OUT_DIR / "faiss.index"
DOCS_PATH = OUT_DIR / "docs.npy"

def load_text_files(data_dir: Path) -> List[str]:
    docs = []
    if not data_dir.exists():
        print(f"data dir {data_dir} does not exist")
        return docs
    for p in sorted(data_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    docs.append(f.read())
            except Exception as e:
                print("failed reading", p, e)
    return docs


def chunk_text(text: str, max_words: int = 200, stride: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        chunk = words[i:i+max_words]
        chunks.append(" ".join(chunk))
        if i + max_words >= n:
            break
        i += max_words - stride
    return chunks


def main():
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import faiss
    except Exception as e:
        print("Missing heavy dependencies for ingest:", e)
        print("Install sentence-transformers, numpy, faiss-cpu to run ingest")
        return

    print("Loading text files from", DATA_DIR)
    docs = load_text_files(DATA_DIR)
    if not docs:
        print("No documents found in data directory; place .txt files there and retry")
        return

    # chunk all documents
    chunks = []
    source_map = []  # maps chunk -> (doc_idx, start_word)
    for di, d in enumerate(docs):
        for c in chunk_text(d):
            chunks.append(c)
            source_map.append(di)

    print(f"Created {len(chunks)} chunks from {len(docs)} docs")

    # embed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    # normalize for cosine (use inner product in FAISS)
    import numpy as _np
    norms = _np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))

    # save index and docs/chunks mapping
    faiss.write_index(index, str(INDEX_PATH))
    print("Saved FAISS index to", INDEX_PATH)

    # Save chunk texts and source map for retrieval
    import numpy as np
    np.save(str(DOCS_PATH), np.array(chunks, dtype=object))
    # Also save a small metadata file
    meta = {"source_map": source_map, "source_docs": [str(p) for p in sorted(DATA_DIR.iterdir()) if p.is_file()]}
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved docs and meta files")


if __name__ == "__main__":
    main()
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en")

documents = []
data_dir = "data"
for filename in os.listdir(data_dir):
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        continue
    # Open with replacement of invalid bytes to avoid UnicodeDecodeError
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        documents.append(f.read())

embeddings = model.encode(documents)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "rag/faiss.index")
np.save("rag/docs.npy", np.array(documents))
