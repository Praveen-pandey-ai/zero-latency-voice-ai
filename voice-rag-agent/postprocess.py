"""Voice-optimized postprocessing helpers.

Functions to shorten/simplify LLM responses for spoken output.
"""
import re


def postprocess_for_voice(text: str, max_sentence_words: int = 18) -> str:
    """Shorten long sentences and simplify punctuation for TTS.

    This simple implementation breaks text into sentences and truncates
    any sentence longer than `max_sentence_words` words. It also replaces
    excessive whitespace and fixes spacing around punctuation.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split into sentences (very simple splitter)
    parts = re.split(r'(?<=[.!?]) +', text)
    out_parts = []
    for p in parts:
        words = p.split()
        if len(words) > max_sentence_words:
            # truncate and add an ellipsis to indicate continuation
            p = " ".join(words[:max_sentence_words]) + "..."
        out_parts.append(p)

    return " ".join(out_parts)
