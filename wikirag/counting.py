"""A counting wrapper around the LLM client.

The efficiency table reports agent calls per question, so that number has to be measured
rather than inferred from chain lengths -- an estimate drifts as soon as the search
changes shape, which is exactly when the number matters.
"""

from __future__ import annotations

import threading
import time


class CountingLLM:
    def __init__(self, inner):
        self.inner = inner
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_chars = 0
        self.seconds = 0.0

    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.prompt_chars = 0
            self.seconds = 0.0

    def _record(self, system: str, user: str, dt: float) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_chars += len(system) + len(user)
            self.seconds += dt

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        t0 = time.time()
        try:
            return self.inner.complete(system, user, max_tokens=max_tokens)
        finally:
            self._record(system, user, time.time() - t0)

    def complete_vision(self, system: str, user: str, image_paths: list[str],
                        max_tokens: int | None = None) -> str:
        t0 = time.time()
        try:
            return self.inner.complete_vision(system, user, image_paths,
                                              max_tokens=max_tokens)
        finally:
            self._record(system, user, time.time() - t0)

    def __getattr__(self, item):
        return getattr(self.inner, item)
