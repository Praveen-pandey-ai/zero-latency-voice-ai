"""Hybrid retriever: FAISS vector search + BM25 fallback.

This module provides a `hybrid_search(query, k=5)` function that
attempts to use a vector index (faiss + sentence-transformers) if
available, and falls back to BM25 (rank_bm25) or a simple lexical
scorer when dependencies are missing. All imports are optional so the
module is safe to import during development.
"""
from typing import List, Tuple
import os

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:
    np = None
    SentenceTransformer = None
    faiss = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


_index = None
_docs = []
_embedder = None


def load_index(index_path: str = "rag/faiss.index", docs_path: str = "rag/docs.npy"):
    global _index, _docs, _embedder
    if faiss is None or np is None:
        return False
    if not os.path.exists(index_path) or not os.path.exists(docs_path):
        return False
    try:
        _index = faiss.read_index(index_path)
        _docs = np.load(docs_path, allow_pickle=True).tolist()
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return True
    except Exception:
        return False


def _bm25_from_documents(documents: List[str]):
    if BM25Okapi is None:
        return None
    tokenized = [d.split() for d in documents]
    return BM25Okapi(tokenized)


def hybrid_search(query: str, k: int = 5) -> List[Tuple[float, str]]:
    """Return list of (score, doc_text) tuples ordered best-first.

    Scores are higher = better. If vector search is available it is
    used; otherwise BM25 or lexical fallback is used.
    """
    # Try vector search
    if _index is not None and _embedder is not None:
        try:
            emb = _embedder.encode([query])
            D, I = _index.search(np.array(emb).astype('float32'), k)
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(_docs):
                    continue
                results.append((float(-score), _docs[idx]))
            return results
        except Exception:
            pass

    # Try BM25 over in-memory docs
    if _docs and BM25Okapi is not None:
        try:
            bm25 = _bm25_from_documents(_docs)
            scores = bm25.get_scores(query.split())
            ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:k]
            return [(float(s), _docs[i]) for i, s in ranked]
        except Exception:
            pass

    # Simple lexical fallback (count overlap)
    if _docs:
        qset = set(query.lower().split())
        scored = []
        for d in _docs:
            score = len(qset.intersection(set(d.lower().split())))
            if score > 0:
                scored.append((float(score), d))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    return []
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

model = SentenceTransformer("BAAI/bge-base-en")
index = faiss.read_index("rag/faiss.index")
docs = np.load("rag/docs.npy", allow_pickle=True)

tokenized = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized)

def hybrid_search(query, k=5):
    q_emb = model.encode([query])
    _, I = index.search(q_emb, k)
    vector_hits = [docs[i] for i in I[0]]

    bm25_hits = bm25.get_top_n(query.split(), docs, n=k)

    return list(set(vector_hits + bm25_hits))
