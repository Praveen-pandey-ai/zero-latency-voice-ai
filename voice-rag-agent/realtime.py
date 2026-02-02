"""Realtime speculative-execution coordinator.

This module provides a `RealtimeCoordinator` class that accepts
partial transcripts (from ASR) and performs parallel steps:

- Start a short "filler" LLM response immediately and play it via TTS.
- In parallel, run query rewriting, hybrid retrieval and cross-encoder
  reranking to produce high-quality context.
- When the reranker is done, call the LLM again with the retrieved
  context to produce the final answer and play it (replacing the
  filler response audibly).

Notes:
- This is a prototype that uses synchronous LLM/TTS functions via
  `asyncio.to_thread` so it can run without special async SDKs.
- Streaming token-level LLM/TTS is not implemented (depends on
  provider SDKs). The filler is generated and played quickly to
  reduce perceived latency; the final response replaces it.
"""
from __future__ import annotations

import asyncio
import time
import logging
import json
from dataclasses import dataclass, asdict
from typing import List, Optional

from app import get_llm_response
from rag.rewrite import rewrite_query
from rag.retriever import hybrid_search
from rag.reranker import rerank
from postprocess import postprocess_for_voice
from speech.tts import speak as tts_speak


logger = logging.getLogger("realtime")
logging.basicConfig(level=logging.INFO)


@dataclass
class MetricsCollector:
    runs: int = 0
    total_retrieval_time: float = 0.0
    total_time: float = 0.0

    def add(self, retrieval_time: float, total_time: float):
        self.runs += 1
        self.total_retrieval_time += retrieval_time
        self.total_time += total_time

    def summary(self):
        if self.runs == 0:
            return {}
        return {
            "runs": self.runs,
            "avg_retrieval_time": self.total_retrieval_time / self.runs,
            "avg_total_time": self.total_time / self.runs,
        }

    def dump(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"runs": self.runs, "total_retrieval_time": self.total_retrieval_time, "total_time": self.total_time, **self.summary()}, f, indent=2)


class RealtimeCoordinator:
    def __init__(self):
        # simple conversation history (most recent last)
        self.history: List[str] = []
        self._lock = asyncio.Lock()
        self._current_task: Optional[asyncio.Task] = None
        self.metrics = MetricsCollector()

    async def handle_partial(self, partial_transcript: str) -> None:
        """Process a partial transcript (ASR partial result).

        This starts the speculative pipeline: immediate filler TTS and a
        background retrieval+rerank. When retrieval completes, we call
        the LLM for the final answer including the retrieved context,
        postprocess it for speech, and play it.
        """
        async with self._lock:
            # cancel any previously running pipeline — for a real system
            # you may want to merge or keep multiple in-flight queries.
            if self._current_task and not self._current_task.done():
                try:
                    self._current_task.cancel()
                except Exception:
                    pass

            self._current_task = asyncio.create_task(self._run_pipeline(partial_transcript))

    async def _run_pipeline(self, partial: str) -> None:
        start_ts = time.time()

        # 1) Immediately generate a short filler from LLM and speak it
        filler_prompt = f"You are a helpful assistant. Give a very short acknowledgement for: {partial}"

        # run LLM in a thread to avoid blocking event loop
        filler_text = await asyncio.to_thread(get_llm_response, filler_prompt)
        # postprocess small filler to keep it short
        filler_text = filler_text.split(".")[0] + "."

        # play filler (blocking in separate thread) while retrieval runs
        loop = asyncio.get_running_loop()
        filler_play = loop.run_in_executor(None, tts_speak, filler_text)

        # 2) start retrieval+rerank in parallel
        retrieval_task = asyncio.create_task(self._retrieve_and_rerank(partial))

        # Wait for retrieval to finish
        try:
            contexts = await retrieval_task
        except asyncio.CancelledError:
            return

        retrieval_time = time.time() - start_ts

        # 3) Call LLM with context and user partial to produce final answer
        ctx_text = "\n\n".join(contexts)
        final_prompt = f"Context:\n{ctx_text}\n\nQuestion: {partial}"
        final_answer = await asyncio.to_thread(get_llm_response, final_prompt)
        final_answer = postprocess_for_voice(final_answer)

        # wait for filler playback to finish before playing final (optional)
        try:
            await filler_play
        except Exception:
            pass

        # play final answer
        await asyncio.to_thread(tts_speak, final_answer)

        # record metrics
        try:
            self.metrics.add(retrieval_time, time.time() - start_ts)
            logger.info("metrics updated: %s", self.metrics.summary())
        except Exception:
            logger.exception("failed to record metrics")

        # update conversation history
        self.history.append(partial)
        self.history.append(final_answer)

        total_time = time.time() - start_ts
        logger.info("[realtime] retrieval_time=%.3fs total_time=%.3fs", retrieval_time, total_time)

    def dump_metrics(self, path: str = "realtime_metrics.json"):
        try:
            self.metrics.dump(path)
            logger.info("dumped metrics to %s", path)
        except Exception:
            logger.exception("failed to dump metrics")

    async def _retrieve_and_rerank(self, query: str) -> List[str]:
        # rewrite with conversation history
        q2 = rewrite_query(query, self.history)

        # hybrid search
        candidates = hybrid_search(q2, k=8)
        texts = [c[1] for c in candidates]

        # rerank (may be CPU-heavy)
        reranked = rerank(q2, texts, top_k=3)
        # rerank returns list of (score, text)
        top_texts = [t for _, t in reranked]
        return top_texts


if __name__ == "__main__":
    # quick interactive demo: type partial transcripts and watch pipeline
    import sys

    async def demo():
        coord = RealtimeCoordinator()
        print("Realtime demo. Type partial transcripts (empty to quit).")
        while True:
            try:
                partial = await asyncio.to_thread(sys.stdin.readline)
            except Exception:
                break
            if not partial:
                break
            partial = partial.strip()
            if partial == "":
                break
            await coord.handle_partial(partial)

    asyncio.run(demo())
