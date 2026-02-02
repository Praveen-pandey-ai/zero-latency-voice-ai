"""Cross-encoder reranker with fallback.

Provides `rerank(query, candidates, top_k)` where candidates is a
list of document texts. If a cross-encoder model is available it is
used; otherwise a simple lexical score is returned.
"""
from typing import List, Tuple

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


def rerank(query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[float, str]]:
    """Return list of (score, doc) ordered best-first."""
    if not candidates:
        return []

    if CrossEncoder is not None:
        try:
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            pairs = [[query, c] for c in candidates]
            scores = model.predict(pairs)
            ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])[:top_k]
            return [(float(s), d) for s, d in ranked]
        except Exception:
            pass

    # Fallback: simple overlap score
    qset = set(query.lower().split())
    scored = []
    for d in candidates:
        score = len(qset.intersection(set(d.lower().split())))
        scored.append((float(score), d))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]

