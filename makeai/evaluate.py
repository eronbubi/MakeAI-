"""Evaluation: validation loss, perplexity, token accuracy, benchmarks, test prompts, comparison."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

from . import registry, store


def _native(uid: str):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return registry.load_model(uid, device=dev)


def dataset_metrics(uid: str, ds_id: str, split: str = "val", max_batches: int = 50, micro_batch: int = 4,
                    context_length: int | None = None, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Loss / perplexity / token accuracy of a model on a dataset split (tokenised with the model's tokenizer)."""
    import torch

    from .data import datasets as D
    from .train.data import PackedSplit, val_batches
    model, tok, m = _native(uid)
    ctx = min(context_length or model.cfg.context_length, model.cfg.context_length, 2048)
    tdir = registry.tokenizer_dir(uid) if registry.tokenizer_dir(uid).exists() else registry.tokenizer_dir(m["base_model"])
    if progress:
        progress("tokenising")
    info = D.prepare(ds_id, str(tdir), ctx, val_ratio=0.02 if split == "val" else 0.0, test_ratio=0.0)
    sp = PackedSplit(info["path"], split, ctx, info["dtype"])
    if sp.n_rows == 0 and split == "val":
        sp = PackedSplit(info["path"], "train", ctx, info["dtype"])
        split = "train"
    dev = next(model.parameters()).device
    tot = tok_n = correct = 0.0
    t0 = time.time()
    with torch.inference_mode():
        for i, (x, y) in enumerate(val_batches(sp, micro_batch, max_batches)):
            x, y = x.to(dev), y.to(dev)
            logits = model(x)[0].float()
            valid = y != -100
            n = int(valid.sum())
            if not n:
                continue
            tot += float(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                                           ignore_index=-100, reduction="sum"))
            tok_n += n
            correct += float(((logits.argmax(-1) == y) & valid).sum())
            if progress and i % 5 == 0:
                progress(f"batch {i + 1}")
    del model
    torch.cuda.empty_cache()
    if not tok_n:
        raise ValueError("no evaluable tokens")
    loss = tot / tok_n
    return {"kind": "dataset", "dataset": ds_id, "split": split, "loss": loss, "perplexity": math.exp(min(loss, 50)),
            "token_accuracy": correct / tok_n, "tokens": int(tok_n), "context_length": ctx,
            "seconds": round(time.time() - t0, 2)}


def _match(pred: str, expected: str, mode: str) -> bool:
    p, e = pred.strip(), expected.strip()
    if mode == "exact":
        return p == e
    if mode == "exact_ci":
        return p.lower() == e.lower()
    if mode == "starts_with":
        return p.lower().startswith(e.lower())
    if mode == "regex":
        return re.search(expected, pred, re.S) is not None
    return e.lower() in p.lower()      # contains


def run_benchmark(uid: str, items: list[dict[str, Any]], manager, params: dict | None = None,
                  progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Custom benchmark: items {prompt | messages, expected, match?, system?}. Greedy decoding by default."""
    p = {"temperature": 0, "max_tokens": 64, "repetition_penalty": 1.0, "top_k": 0, "top_p": 1, "min_p": 0}
    p.update(params or {})
    results = []
    t0 = time.time()
    for i, it in enumerate(items):
        msgs = it.get("messages") or [{"role": "user", "content": it["prompt"]}]
        out = ""
        for c in manager.stream(uid, msgs, p, system_prompt=it.get("system")):
            out += c.get("delta", "")
        ok = _match(out, str(it.get("expected", "")), it.get("match", "contains")) if "expected" in it else None
        results.append({"prompt": msgs[-1]["content"], "expected": it.get("expected"), "output": out, "correct": ok})
        if progress:
            progress(f"{i + 1}/{len(items)}")
    scored = [r for r in results if r["correct"] is not None]
    acc = sum(1 for r in scored if r["correct"]) / len(scored) if scored else None
    return {"kind": "benchmark", "accuracy": acc, "n": len(items), "scored": len(scored), "results": results,
            "params": p, "seconds": round(time.time() - t0, 2)}


def load_benchmark_file(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    return data["items"] if isinstance(data, dict) else data


def record(uid: str, result: dict[str, Any], name: str = "") -> dict[str, Any]:
    m = registry.load_manifest(uid)
    entry = {"id": store.new_id("eval"), "name": name or result["kind"], "at": store.now_iso(),
             **{k: v for k, v in result.items() if k != "results"}}
    d = registry.model_dir(uid) / "evaluations"
    d.mkdir(exist_ok=True)
    store.write_json(d / f"{entry['id']}.json", {**entry, "results": result.get("results")})
    m.setdefault("evaluations", []).append(entry)
    registry.save_manifest(m)
    return entry


def compare(uids: list[str]) -> list[dict[str, Any]]:
    """Side-by-side of recorded evaluations (same dataset / benchmark name)."""
    rows = []
    for uid in uids:
        m = registry.load_manifest(uid)
        rows.append({"uid": uid, "name": m["name"], "version": m["version"], "params": m.get("param_count"),
                     "last_eval": m.get("last_eval"), "evaluations": m.get("evaluations", [])})
    return rows
