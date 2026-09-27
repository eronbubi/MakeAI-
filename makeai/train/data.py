"""Deterministic, random-access batch source over prepared token streams.

Batch ``k`` is a pure function of (seed, k): datasets are interleaved with a
smooth weighted round-robin, each dataset walks a per-epoch permutation of its
rows. Resuming therefore only needs the next batch index, and prefetch threads
can build batches in parallel without affecting the order.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


class PackedSplit:
    def __init__(self, prepared_dir: str | Path, split: str, context_length: int, dtype: str):
        p = Path(prepared_dir)
        self.row = context_length + 1
        tok = p / f"{split}.tokens.bin"
        n = tok.stat().st_size // np.dtype(dtype).itemsize if tok.exists() else 0
        self.n_rows = n // self.row
        if self.n_rows:
            self.tokens = np.memmap(tok, dtype=dtype, mode="r", shape=(self.n_rows * self.row,))
            self.mask = np.memmap(p / f"{split}.mask.bin", dtype=np.uint8, mode="r", shape=(self.n_rows * self.row,))

    def get_row(self, r: int) -> tuple[np.ndarray, np.ndarray]:
        s = r * self.row
        return self.tokens[s:s + self.row], self.mask[s:s + self.row]


def _swrr(weights: list[int]) -> list[int]:
    """Smooth weighted round robin pattern (one full period)."""
    cur = [0] * len(weights)
    total = sum(weights)
    out = []
    for _ in range(total):
        for i, w in enumerate(weights):
            cur[i] += w
        j = max(range(len(weights)), key=lambda i: cur[i])
        cur[j] -= total
        out.append(j)
    return out


class BatchSource:
    def __init__(self, sources: list[tuple[PackedSplit, float]], micro_batch: int, seed: int):
        sources = [(s, w) for s, w in sources if s.n_rows > 0 and w > 0]
        if not sources:
            raise ValueError("no training rows - dataset smaller than one context window?")
        self.splits = [s for s, _ in sources]
        ws = [w for _, w in sources]
        scale = 100 / sum(ws)
        iw = [max(1, round(w * scale)) for w in ws]
        self.pattern = _swrr(iw)
        self.period = len(self.pattern)
        self.prefix = [[self.pattern[:k].count(d) for k in range(self.period + 1)] for d in range(len(iw))]
        self.iw = iw
        self.mb = micro_batch
        self.seed = seed
        self._perm_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def total_rows(self) -> int:
        return sum(s.n_rows for s in self.splits)

    def _perm(self, d: int, epoch: int) -> np.ndarray:
        key = (d, epoch)
        with self._lock:
            p = self._perm_cache.get(key)
            if p is not None:
                self._perm_cache.move_to_end(key)
                return p
        p = np.random.default_rng([self.seed, d, epoch]).permutation(self.splits[d].n_rows)
        with self._lock:
            self._perm_cache[key] = p
            while len(self._perm_cache) > 2 * len(self.splits) + 2:
                self._perm_cache.popitem(last=False)
        return p

    def sample(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        q, r = divmod(i, self.period)
        d = self.pattern[r]
        j = q * self.iw[d] + self.prefix[d][r]
        n = self.splits[d].n_rows
        epoch, k = divmod(j, n)
        return self.splits[d].get_row(int(self._perm(d, epoch)[k]))

    def batch(self, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [self.sample(b * self.mb + t) for t in range(self.mb)]
        tok = np.stack([r[0] for r in rows]).astype(np.int64)
        msk = np.stack([r[1] for r in rows])
        x = torch.from_numpy(tok[:, :-1].copy())
        y = torch.from_numpy(tok[:, 1:].copy())
        y[torch.from_numpy(msk[:, 1:] == 0)] = -100
        return x, y


class Prefetcher:
    """Builds upcoming batches on ``workers`` threads; measures time the trainer waits for data."""

    def __init__(self, src: BatchSource, start: int, workers: int = 2, pin: bool = True):
        self.src = src
        self.next = start
        self.workers = max(1, workers)
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="data")
        self.pending: dict[int, Future] = {}
        self.pin = pin and torch.cuda.is_available()
        self.wait_s = 0.0

    def _build(self, b: int):
        x, y = self.src.batch(b)
        if self.pin:
            x, y = x.pin_memory(), y.pin_memory()
        return x, y

    def _fill(self):
        ahead = self.workers * 2
        for b in range(self.next, self.next + ahead):
            if b not in self.pending:
                self.pending[b] = self.pool.submit(self._build, b)

    def get(self) -> tuple[int, torch.Tensor, torch.Tensor]:
        self._fill()
        t = time.perf_counter()
        x, y = self.pending.pop(self.next).result()
        self.wait_s += time.perf_counter() - t
        b = self.next
        self.next += 1
        self._fill()
        return b, x, y

    def close(self):
        for f in self.pending.values():
            f.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)


def val_batches(split: PackedSplit, micro_batch: int, max_batches: int):
    if split.n_rows == 0:
        return
    n = min(max_batches, max(1, split.n_rows // micro_batch))
    for b in range(n):
        rows = [split.get_row((b * micro_batch + t) % split.n_rows) for t in range(micro_batch)]
        tok = np.stack([r[0] for r in rows]).astype(np.int64)
        msk = np.stack([r[1] for r in rows])
        x = torch.from_numpy(tok[:, :-1].copy())
        y = torch.from_numpy(tok[:, 1:].copy())
        y[torch.from_numpy(msk[:, 1:] == 0)] = -100
        yield x, y
