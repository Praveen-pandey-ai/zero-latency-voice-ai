"""Simple conversation-aware query rewriter.

This lightweight rewriter uses the last conversation turn to resolve
pronouns like "it", "that", "the second one" by substituting nouns
from the last user or assistant message. It's intentionally simple and
meant as a placeholder for a proper rewrite model.
"""
from typing import List


def rewrite_query(query: str, history: List[str]) -> str:
    """Return rewritten query using conversation history.

    `history` is expected to be a list of previous messages (most recent last).
    """
    if not history:
        return query

    last = history[-1]
    # trivial heuristic: if query contains pronouns, append last utterance
    pronouns = ["it", "that", "this", "they", "them", "second", "first"]
    if any(p in query.lower() for p in pronouns):
        return f"{query} (context: {last})"
    return query

