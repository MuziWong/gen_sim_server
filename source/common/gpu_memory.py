# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Return unused CUDA cache after a service's inference queue becomes idle."""

from __future__ import annotations

import gc
import json
import math
import os
import threading
import time
from typing import Any, Callable, TypeVar

__all__ = ["IdleCudaReclaimer"]

_Result = TypeVar("_Result")


class IdleCudaReclaimer:
    """Track queued and running work; serialize idle cleanup with admission.

    ``run`` wraps a function returning CPU-only results. A single-worker job
    queue can instead call ``begin`` when accepting a job and ``end`` after
    completion or queued cancellation. Every accepted job must be balanced.
    Existing service locks continue to serialize inference itself.
    """

    def __init__(self, torch_module: Any, service: str, device: str = "cuda:0") -> None:
        self._torch = torch_module
        self._service = service
        self._device = device
        self._idle_seconds = float(os.getenv("GENSIM_CUDA_IDLE_SECONDS", "1.0"))
        if not math.isfinite(self._idle_seconds) or self._idle_seconds <= 0:
            raise ValueError("GENSIM_CUDA_IDLE_SECONDS must be finite and positive")
        self._enabled = device.startswith("cuda") and torch_module.cuda.is_available()
        self._lock = threading.Lock()
        self._pending = 0
        self._generation = 0
        self._timer: threading.Timer | None = None
        print(
            "[gpu-memory] "
            + json.dumps(
                {
                    "service": service,
                    "event": "configured",
                    "enabled": self._enabled,
                    "idle_seconds": self._idle_seconds,
                }
            ),
            flush=True,
        )

    def begin(self) -> None:
        """Count work before it can enter inference or an external queue."""
        with self._lock:
            self._pending += 1
            self._generation += 1
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def end(self) -> None:
        """Release one work item after its temporary references unwind."""
        with self._lock:
            if self._pending <= 0:
                raise RuntimeError("Unbalanced GPU work accounting")
            self._pending -= 1
            if self._pending == 0 and self._enabled:
                self._generation += 1
                timer = threading.Timer(
                    self._idle_seconds, self._reclaim, args=(self._generation,)
                )
                timer.daemon = True
                self._timer = timer
                timer.start()

    def run(self, operation: Callable[[], _Result]) -> _Result:
        """Run work with balanced tracking, including exception paths."""
        self.begin()
        try:
            return operation()
        finally:
            self.end()

    def _reclaim(self, generation: int) -> None:
        # Holding this lock prevents a new request from entering GPU work
        # during collection. Stale callbacks cannot clean a newer idle period.
        with self._lock:
            if self._pending or generation != self._generation:
                return
            self._timer = None
            started = time.monotonic()
            try:
                cuda = self._torch.cuda
                with cuda.device(self._device):
                    allocated_before = cuda.memory_allocated(self._device)
                    reserved_before = cuda.memory_reserved(self._device)
                    collected = gc.collect()
                    cuda.synchronize(self._device)
                    cuda.empty_cache()
                    allocated_after = cuda.memory_allocated(self._device)
                    reserved_after = cuda.memory_reserved(self._device)
                record = {
                    "service": self._service,
                    "event": "idle_reclaim",
                    "allocated_before_bytes": allocated_before,
                    "allocated_after_bytes": allocated_after,
                    "reserved_before_bytes": reserved_before,
                    "reserved_after_bytes": reserved_after,
                    "released_bytes": max(0, reserved_before - reserved_after),
                    "collected_objects": collected,
                    "elapsed_seconds": round(time.monotonic() - started, 4),
                }
            except Exception as exc:
                # Cleanup must not take down a service or change an HTTP result.
                record = {
                    "service": self._service,
                    "event": "reclaim_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            print("[gpu-memory] " + json.dumps(record), flush=True)
