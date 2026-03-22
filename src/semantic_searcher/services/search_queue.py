"""Bounded search concurrency with FIFO queuing.

Uses an asyncio.Semaphore to cap parallel search operations and a
ThreadPoolExecutor to run CPU-bound FAISS work off the event loop.
Excess requests wait in FIFO order (asyncio.Semaphore is fair).
"""
import asyncio
import logging
import queue as _queue
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

log = logging.getLogger(__name__)


class SearchQueue:
    def __init__(self, max_workers: int = 4, max_queued: int = 50):
        self._max_workers = max_workers
        self._max_queued = max_queued
        self._semaphore = asyncio.Semaphore(max_workers)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="search")
        self._active = 0
        self._queued = 0
        self._total_served = 0
        self._total_queued_time = 0.0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    @property
    def queued(self) -> int:
        return self._queued

    @property
    def stats(self) -> dict:
        avg_wait = (self._total_queued_time / self._total_served) if self._total_served else 0
        return {
            "max_workers": self._max_workers,
            "active": self._active,
            "queued": self._queued,
            "total_served": self._total_served,
            "avg_queue_wait_ms": round(avg_wait * 1000, 1),
        }

    async def execute(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Run a sync search function with bounded concurrency.

        If all workers are busy, the request waits in FIFO order.
        If the queue is full, raises an exception immediately.
        """
        async with self._lock:
            if self._queued >= self._max_queued:
                raise SearchQueueFullError(self._queued, self._active)
            self._queued += 1

        queue_start = time.monotonic()
        try:
            await self._semaphore.acquire()
            queue_wait = time.monotonic() - queue_start

            async with self._lock:
                self._queued -= 1
                self._active += 1
                self._total_queued_time += queue_wait

            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))
                return result
            finally:
                async with self._lock:
                    self._active -= 1
                    self._total_served += 1
                self._semaphore.release()
        except Exception:
            async with self._lock:
                if self._queued > 0:
                    self._queued -= 1
            raise

    async def execute_generator(self, fn: Callable[..., Any], *args, **kwargs):
        """Run a sync generator function with bounded concurrency.

        Collects all results from the generator while holding the semaphore,
        then yields them. This ensures the CPU-bound work is bounded.
        """
        async with self._lock:
            if self._queued >= self._max_queued:
                raise SearchQueueFullError(self._queued, self._active)
            self._queued += 1

        queue_start = time.monotonic()
        try:
            await self._semaphore.acquire()
            queue_wait = time.monotonic() - queue_start

            async with self._lock:
                self._queued -= 1
                self._active += 1
                self._total_queued_time += queue_wait

            try:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    self._executor, lambda: list(fn(*args, **kwargs))
                )
                return results
            finally:
                async with self._lock:
                    self._active -= 1
                    self._total_served += 1
                self._semaphore.release()
        except Exception:
            async with self._lock:
                if self._queued > 0:
                    self._queued -= 1
            raise

    async def execute_streaming(self, fn: Callable[..., Any], *args, **kwargs):
        """Run a sync generator with bounded concurrency, yielding events live.

        Returns an async generator that yields items as the worker produces them.
        The semaphore is held for the entire duration of the generator.
        """
        async with self._lock:
            if self._queued >= self._max_queued:
                raise SearchQueueFullError(self._queued, self._active)
            self._queued += 1

        queue_start = time.monotonic()
        try:
            await self._semaphore.acquire()
            queue_wait = time.monotonic() - queue_start

            async with self._lock:
                self._queued -= 1
                self._active += 1
                self._total_queued_time += queue_wait

            # Thread-safe queue + event for signaling (no polling)
            q: _queue.Queue = _queue.Queue()
            _SENTINEL = object()
            loop = asyncio.get_event_loop()
            data_ready = asyncio.Event()

            def _worker():
                try:
                    for item in fn(*args, **kwargs):
                        q.put(item)
                        loop.call_soon_threadsafe(data_ready.set)
                except Exception as exc:
                    q.put(exc)
                    loop.call_soon_threadsafe(data_ready.set)
                finally:
                    q.put(_SENTINEL)
                    loop.call_soon_threadsafe(data_ready.set)

            fut = loop.run_in_executor(self._executor, _worker)

            try:
                while True:
                    # Drain all available items before waiting
                    while True:
                        try:
                            item = q.get_nowait()
                        except _queue.Empty:
                            break
                        if item is _SENTINEL:
                            await fut
                            return
                        if isinstance(item, Exception):
                            raise item
                        yield item
                    # Wait for worker to signal new data
                    data_ready.clear()
                    await data_ready.wait()
            finally:
                async with self._lock:
                    self._active -= 1
                    self._total_served += 1
                self._semaphore.release()
        except SearchQueueFullError:
            raise
        except Exception:
            async with self._lock:
                if self._queued > 0:
                    self._queued -= 1
            raise

    def shutdown(self):
        self._executor.shutdown(wait=False)


class SearchQueueFullError(Exception):
    def __init__(self, queued: int, active: int):
        self.queued = queued
        self.active = active
        super().__init__(f"Search queue full ({queued} queued, {active} active)")
