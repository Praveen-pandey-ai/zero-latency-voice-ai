"""Runner demonstrating speculative execution and measuring TTFB.

This script performs a simple speculative pipeline:
- Immediately emit a short filler phrase (TTS) to reduce perceived latency.
- In parallel, run `hybrid_search` and `rerank` to get context.
- When context is ready, call the LLM for a final answer and play it.

The implementation uses safe fallbacks so it can run without heavy deps.
"""
import time
import asyncio
from typing import List

from app import get_llm_response
from speech.tts import speak as tts_speak
from rag.retriever import hybrid_search, load_index
from rag.reranker import rerank
from rag.rewrite import rewrite_query
from postprocess import postprocess_for_voice


async def speculative_handler(query: str, history: List[str] = None):
    history = history or []

    # Start filler TTS immediately
    filler = "Let me check that for you."
    t0 = time.time()
    # run filler in thread to avoid blocking event loop
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, tts_speak, filler)
    ttfb = time.time() - t0
    print(f"TTFB (filler) = {ttfb*1000:.0f} ms")

    # Start retrieval+rerank in background
    async def do_retrieval(q):
        q2 = rewrite_query(q, history)
        candidates = hybrid_search(q2, k=8)
        # extract texts
        texts = [c[1] for c in candidates]
        reranked = rerank(q2, texts, top_k=3)
        return [r[1] for r in reranked]

    retrieval_task = asyncio.create_task(do_retrieval(query))

    # Meanwhile, start LLM streaming placeholder (we use synchronous call)
    # We'll block until retrieval completes so we can include top context
    contexts = await retrieval_task
    print(f"Retrieved {len(contexts)} contexts")

    # Call LLM with context appended
    ctx_text = "\n\n".join(contexts)
    prompt = f"Context:\n{ctx_text}\n\nQuestion: {query}"
    response = get_llm_response(prompt)
    response = postprocess_for_voice(response)

    # Play final response
    await loop.run_in_executor(None, tts_speak, response)


def main():
    load_index()  # best-effort
    q = input("Query: ")
    # Use RealtimeCoordinator for speculative execution and metrics
    from realtime import RealtimeCoordinator

    coord = RealtimeCoordinator()
    asyncio.run(coord.handle_partial(q))
    # allow background tasks to finish briefly
    try:
        asyncio.run(asyncio.sleep(0.1))
    except Exception:
        pass
    coord.dump_metrics()


if __name__ == "__main__":
    main()
