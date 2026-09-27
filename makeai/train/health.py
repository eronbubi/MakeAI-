"""Training Performance Score (1-10) and health status.

This is a technical health indicator of the *training run* - how well the
hardware is being used and how stable optimisation is. It says nothing about
how intelligent or good the resulting model is.

Every component is computed from measured values: NVML telemetry, OS memory
counters and the trainer's own step metrics. A component whose inputs are not
available is left out rather than guessed.
"""
from __future__ import annotations

import statistics
from typing import Any

WEIGHTS = {"gpu_utilization": 0.25, "throughput": 0.20, "thermals": 0.15, "stability": 0.15,
           "vram_efficiency": 0.10, "data_loading": 0.10, "ram_pressure": 0.05}
LABELS = {"gpu_utilization": "GPU Utilization", "throughput": "Throughput", "thermals": "Thermals",
          "stability": "Stability", "vram_efficiency": "VRAM Efficiency", "data_loading": "Data Loading",
          "ram_pressure": "RAM Pressure"}


def _clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, x))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def status_for(score: float) -> tuple[str, str]:
    if score >= 8.5:
        return "excellent", "Excellent"
    if score >= 7.0:
        return "good", "Good"
    if score >= 5.5:
        return "moderate", "Moderate"
    if score >= 3.5:
        return "bottleneck", "Bottleneck"
    return "critical", "Critical"


def compute(steps: list[dict[str, Any]], telemetry: list[dict[str, Any]], *, gpu_index: int = 0,
            gpu_static: dict[str, Any] | None = None, run_cfg: dict[str, Any] | None = None,
            flops_per_token: float | None = None, peak_flops: float | None = None,
            baseline_gpu_util: float | None = None) -> dict[str, Any]:
    """Score from the most recent step metrics and telemetry samples (same time window)."""
    gpu_static = gpu_static or {}
    tr = (run_cfg or {}).get("training", {})
    comp: dict[str, float] = {}
    detail: dict[str, Any] = {}
    issues: list[tuple[float, str, str]] = []   # (severity 0-10, reason, recommendation)

    gpus = [s["gpus"][gpu_index] for s in telemetry if s.get("gpus") and len(s["gpus"]) > gpu_index]

    # ---- GPU utilisation
    util = _mean([g.get("util_pct") for g in gpus])
    if util is not None:
        comp["gpu_utilization"] = _clamp(util / 10)
        detail["gpu_util_pct"] = round(util, 1)

    # ---- data loading
    wait = _mean([s.get("data_wait_frac") for s in steps])
    if wait is not None:
        comp["data_loading"] = _clamp(10 - wait * 20)
        detail["data_wait_pct"] = round(100 * wait, 1)
        if wait > 0.08:
            w = int(tr.get("dataloader_workers", 2))
            issues.append((wait * 20, "CPU data loading is limiting GPU utilization.",
                           f"Increase DataLoader workers from {w} → {min(32, w * 2)}."))

    # ---- VRAM efficiency (memory the run reserved vs device total)
    total_mb = gpu_static.get("vram_total_mb") or (gpus[-1].get("vram_total_mb") if gpus else None)
    reserved = _mean([s.get("mem_reserved_mb") for s in steps])
    used_dev = _mean([g.get("vram_used_mb") for g in gpus])
    if total_mb and (reserved is not None or used_dev is not None):
        frac_run = (reserved or 0) / total_mb
        frac_dev = (used_dev or reserved or 0) / total_mb
        if frac_dev > 0.97:
            v = 4.0
            issues.append((6, f"VRAM is {frac_dev:.0%} full - close to out-of-memory.",
                           "Reduce micro batch size or enable gradient checkpointing."))
        elif frac_run < 0.6:
            v = 4 + 6 * frac_run / 0.6
        else:
            v = 10.0
        comp["vram_efficiency"] = _clamp(v)
        detail.update({"vram_run_pct": round(100 * frac_run, 1), "vram_device_pct": round(100 * frac_dev, 1)})

    # ---- throughput: model FLOP utilisation against the *measured* matmul peak
    tps = _mean([s.get("tokens_per_s") for s in steps])
    if tps:
        detail["tokens_per_s"] = round(tps)
        if flops_per_token and peak_flops:
            mfu = tps * flops_per_token / peak_flops
            comp["throughput"] = _clamp(mfu / 0.5 * 10)
            detail["mfu_pct"] = round(100 * mfu, 1)
        elif len(steps) >= 5:
            xs = [s["tokens_per_s"] for s in steps if s.get("tokens_per_s")]
            cv = statistics.pstdev(xs) / (sum(xs) / len(xs))
            comp["throughput"] = _clamp(10 - cv * 40)
            detail["throughput_cv_pct"] = round(100 * cv, 1)

    # ---- thermals
    temp = _mean([g.get("temp_c") for g in gpus])
    if temp is not None:
        slow = gpu_static.get("slowdown_temp_c") or 90
        margin = slow - temp
        t = 10.0 if margin >= 20 else _clamp(2 + 8 * margin / 20)
        throttles = {r for g in gpus for r in (g.get("throttle") or [])}
        if throttles & {"hw_thermal", "sw_thermal", "hw_slowdown"}:
            t = min(t, 3.0)
            issues.append((8, f"GPU is thermally throttling at {temp:.0f}°C.",
                           "Improve case airflow or lower the GPU power limit."))
        elif margin < 10:
            issues.append((10 - t, f"GPU at {temp:.0f}°C, {margin:.0f}°C below its slowdown temperature.",
                           "Improve airflow before long runs."))
        if "hw_power_brake" in throttles:
            t = min(t, 5.0)
            issues.append((5, "Hardware power brake engaged (PSU/power delivery limit).", "Check power delivery."))
        comp["thermals"] = t
        detail["gpu_temp_c"] = round(temp, 1)
        detail["throttle"] = sorted(throttles)

    # ---- stability
    losses = [s.get("loss") for s in steps]
    if losses:
        bad = sum(1 for x in losses if x is None or x != x or x in (float("inf"), float("-inf")))
        st = 10.0 - 30.0 * bad / len(losses)
        good = [x for x in losses if x is not None and x == x and abs(x) != float("inf")]
        if len(good) >= 8:
            med = statistics.median(good[:-3])
            spike = max(good[-3:]) / med if med > 0 else 1
            if spike > 1.5:
                st -= min(5, (spike - 1.5) * 6)
                lr = tr.get("learning_rate")
                issues.append((6, f"Loss spike detected ({spike:.1f}× recent median).",
                               f"Lower the learning rate{f' from {lr:g} → {lr / 2:g}' if lr else ''} or lengthen warmup."))
        errs = steps[-1].get("errors", 0) if steps else 0
        if errs:
            st -= min(4, errs)
            detail["non_finite_steps"] = errs
        gns = [s.get("grad_norm") for s in steps if s.get("grad_norm") is not None]
        clip = tr.get("grad_clip")
        if gns and clip and statistics.median(gns) > 5 * clip:
            st -= 1.5
            issues.append((3, "Gradient norm is far above the clipping threshold on most steps.",
                           "Consider a lower learning rate."))
        comp["stability"] = _clamp(st)

    # ---- RAM pressure
    ram = _mean([s.get("ram_pct") for s in telemetry])
    if ram is not None:
        r = 10.0 if ram < 80 else _clamp(10 - (ram - 80) * 7 / 15)   # 80% -> 10, 95% -> 3
        if ram >= 97:
            r = 1.0
        comp["ram_pressure"] = r
        detail["ram_pct"] = round(ram, 1)
        if ram >= 88:
            issues.append(((ram - 85) / 1.5, f"System RAM at {ram:.0f}%.",
                           "Close other applications or reduce DataLoader workers."))

    # ---- GPU underuse diagnosis
    if util is not None and util < 70 and (wait or 0) < 0.08:
        mb = tr.get("micro_batch_size")
        headroom = (total_mb - used_dev) / 1024 if total_mb and used_dev else None
        if headroom and headroom > 1.0 and mb:
            issues.append(((70 - util) / 7, f"GPU utilization is {util:.0f}%.",
                           f"Increase micro batch size from {mb} → {mb * 2} ({headroom:.1f} GB VRAM free)."))
        else:
            issues.append(((70 - util) / 7, f"GPU utilization is {util:.0f}%.",
                           "The model may be too small to saturate this GPU; a larger batch or model uses it better."))
    if baseline_gpu_util is not None and baseline_gpu_util > 15:
        issues.append((3, f"Other applications were using {baseline_gpu_util:.0f}% of the GPU before training started.",
                       "Close GPU-heavy applications (games, recorders, browsers with hardware acceleration)."))
        detail["baseline_gpu_util_pct"] = round(baseline_gpu_util, 1)

    if not comp:
        return {"available": False, "reason": "no measurements yet"}
    wsum = sum(WEIGHTS[k] for k in comp)
    score = sum(comp[k] * WEIGHTS[k] for k in comp) / wsum
    critical = (comp.get("thermals", 10) <= 3 or comp.get("stability", 10) <= 3 or comp.get("ram_pressure", 10) <= 1)
    if critical:
        score = min(score, 3.4)
    score = round(max(1.0, min(10.0, score)), 1)
    key, label = status_for(score)
    issues.sort(key=lambda x: -x[0])
    return {
        "available": True,
        "score": score,
        "status": key,
        "status_label": label,
        "components": {k: {"label": LABELS[k], "score": round(v, 1), "weight": WEIGHTS[k]} for k, v in comp.items()},
        "detail": detail,
        "reason": issues[0][1] if issues else "All measured components are healthy.",
        "recommendation": issues[0][2] if issues else None,
        "issues": [{"reason": r, "recommendation": rec} for _, r, rec in issues],
        "note": "Technical health of this training run - not a measure of model quality.",
    }
