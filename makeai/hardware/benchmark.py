"""Measured device throughput (matmul TFLOPS, memory bandwidth).

The recommendation engine uses these measured numbers - not a spec-sheet table -
to estimate training time. Results are cached per device in ``benchmark.json``.
"""
from __future__ import annotations

import time
from typing import Any

from .. import store


def _time_cuda(fn, iters: int) -> float:
    import torch
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters


def run_benchmark(device_index: int = 0, size: int = 4096) -> dict[str, Any]:
    import torch
    res: dict[str, Any] = {"at": time.time()}
    if not torch.cuda.is_available():
        a = torch.randn(1024, 1024)
        b = torch.randn(1024, 1024)
        t0 = time.perf_counter()
        for _ in range(10):
            a @ b
        dt = (time.perf_counter() - t0) / 10
        res.update({"device": "cpu", "tflops": {"fp32": round(2 * 1024**3 / dt / 1e12, 3)}})
        return res
    dev = torch.device("cuda", device_index)
    res["device"] = torch.cuda.get_device_name(dev)
    tf = {}
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        for name, dtype, tf32 in (("fp32", torch.float32, False), ("tf32", torch.float32, True),
                                  ("fp16", torch.float16, False), ("bf16", torch.bfloat16, False)):
            if name == "bf16" and not torch.cuda.is_bf16_supported():
                continue
            torch.backends.cuda.matmul.allow_tf32 = tf32
            a = torch.randn(size, size, device=dev, dtype=dtype)
            b = torch.randn(size, size, device=dev, dtype=dtype)
            dt = _time_cuda(lambda: a @ b, 20)
            tf[name] = round(2 * size**3 / dt / 1e12, 2)
            del a, b
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    res["tflops"] = tf
    x = torch.empty(256 * 2**20, dtype=torch.uint8, device=dev)
    y = torch.empty_like(x)
    dt = _time_cuda(lambda: y.copy_(x), 20)
    res["mem_bandwidth_gbs"] = round(2 * x.numel() / dt / 1e9, 1)
    h = torch.empty(128 * 2**20, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(128 * 2**20, dtype=torch.uint8, device=dev)
    dt = _time_cuda(lambda: d.copy_(h, non_blocking=True), 10)
    res["h2d_bandwidth_gbs"] = round(h.numel() / dt / 1e9, 1)
    del x, y, h, d
    torch.cuda.empty_cache()
    return res


def cached_benchmark(force: bool = False) -> dict[str, Any] | None:
    path = store.home() / "benchmark.json"
    cur = store.read_json(path)
    try:
        import torch
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        name = "cpu"
    if cur and cur.get("device") == name and not force:
        return cur
    if not force:
        return None
    res = run_benchmark()
    store.write_json(path, res)
    return res
