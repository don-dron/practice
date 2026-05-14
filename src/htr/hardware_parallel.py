"""Параллелизм DataLoader и PyTorch threads (профиль max/n/low для training.hardware_utilization)."""

from __future__ import annotations

import os
import queue
import sys
import threading
from typing import Any, Optional

import torch

_SENTINEL = object()


def hardware_profile(tc: dict) -> str:
    raw = tc.get("hardware_utilization", "max")
    s = str(raw).strip().lower()
    if s in ("max", "normal", "low"):
        return s
    return "normal"


def effective_prefetch_factor(
    dc: dict,
    nw: int,
    *,
    colab_rt: bool,
    win_fast: bool,
) -> int:
    raw = dc.get("prefetch_factor", 4)
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        if nw <= 0:
            return 2
        pf = max(2, min(16, nw))
        if sys.platform == "win32":
            pf = min(pf, 8 if win_fast else 4)
        if colab_rt:
            pf = min(pf, max(4, min(8, 2 + nw)))
        return pf
    try:
        pf = max(2, min(32, int(raw)))
    except (TypeError, ValueError):
        pf = 4
    if sys.platform == "win32":
        pf = min(pf, 8 if win_fast else 4)
    if colab_rt:
        pf = min(pf, max(4, min(8, 2 + nw)))
    return pf


def apply_main_thread_env_and_torch(tc: dict, *, nw: int, wt_per_worker: int) -> None:
    """main_torch_threads=null + hardware_utilization=max → авто; OMP/MKL только если env пустые."""
    if hardware_profile(tc) != "max":
        _maybe_set_main_torch_threads_explicit(tc)
        return

    mtp = tc.get("main_torch_threads")
    cpu = os.cpu_count() or 8
    w_load = nw * max(1, wt_per_worker)
    if mtp is None:
        reserved = min(w_load, max(1, cpu // 2))
        mt = max(2, min(32, cpu - reserved))
        mt = max(mt, min(8, max(2, cpu // 2)))
    else:
        try:
            mt = max(1, int(mtp))
        except (TypeError, ValueError):
            mt = max(2, min(32, cpu // 2))

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if not os.environ.get(name):
            os.environ[name] = str(mt)
    try:
        torch.set_num_threads(mt)
        print(f"[htr-train] hardware_utilization=max · main torch threads={mt} (cpu≈{cpu}, num_workers={nw})")
    except Exception:
        pass

    it = max(1, min(8, max(1, cpu // 8)))
    try:
        torch.set_num_interop_threads(it)
        print(f"[htr-train] hardware_utilization=max · torch.set_num_interop_threads({it})")
    except Exception:
        pass


def _maybe_set_main_torch_threads_explicit(tc: dict) -> None:
    mtp = tc.get("main_torch_threads")
    if mtp is None:
        return
    try:
        mt = max(1, int(mtp))
        torch.set_num_threads(mt)
        print(f"[htr-train] main_process torch.set_num_threads({mt})")
    except (TypeError, ValueError):
        pass


class AsyncCpuBatchPrefetcher:
    """Фоновый поток вытягивает следующие CPU-батчи из DataLoader, пока GPU считает предыдущий."""

    def __init__(self, loader: torch.utils.data.DataLoader, max_queue: int = 2) -> None:
        self._loader = loader
        self._max_q = max(1, int(max_queue))
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=self._max_q)
        self._thread: Optional[threading.Thread] = None
        self._exc: Optional[BaseException] = None

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> AsyncCpuBatchPrefetcher:
        self._queue = queue.Queue(maxsize=self._max_q)
        self._exc = None

        def _worker() -> None:
            try:
                for batch in self._loader:
                    self._queue.put(batch)
                self._queue.put(_SENTINEL)
            except BaseException as ex:
                self._exc = ex
                self._queue.put(_SENTINEL)

        self._thread = threading.Thread(target=_worker, daemon=True, name="htr-dataloader-prefetch")
        self._thread.start()
        return self

    def __next__(self) -> Any:
        item = self._queue.get()
        if item is _SENTINEL:
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        return item


def effective_dataloader_worker_torch_threads(dc: dict, profile: str, nw: int) -> int:
    """При max и auto — больше 1 потока в воркере для тяжёлого CPU (осторожно: больше конкуренции за ядра)."""
    raw = dc.get("dataloader_worker_torch_threads", 1)
    if raw is None:
        return 1
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        if profile != "max":
            return 1
        cpu = os.cpu_count() or 8
        if nw <= 0:
            return max(1, min(4, cpu // 4))
        return max(1, min(2, max(1, cpu // max(2, nw * 2))))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 1
