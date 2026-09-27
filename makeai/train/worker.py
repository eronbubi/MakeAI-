"""The training process.

Started by the MakeAI server as ``python -m makeai.train.worker <run_dir>``.
Runs in its own OS process so the Kill Switch can always reclaim the GPU:
the graceful path stops between micro-batches, saves a checkpoint and exits;
the server's hard path terminates the process tree, which the driver answers
by freeing every byte of GPU memory the process held.

Files in ``run_dir``:
    run.json        configuration written by the server (read-only here)
    control.json    commands from the server: pause / resume / stop / checkpoint
    status.json     current state, rewritten atomically
    metrics.jsonl   one JSON line per optimizer step and per evaluation
    log.txt         human-readable log
    checkpoints/    see CheckpointManager
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .. import store

TRAINABLE_PRECISIONS = ("fp32", "tf32", "fp16", "bf16")
STORAGE_ONLY = ("int8", "int4", "nf4", "fp4")


class Run:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.cfg: dict[str, Any] = store.read_json(self.dir / "run.json")
        self.status: dict[str, Any] = store.read_json(self.dir / "status.json", {}) or {}
        self._ctrl_seq = int(self.status.get("control_seq", 0))
        self._ctrl_mtime = 0
        self.pending_set: dict[str, Any] = {}
        self._log = open(self.dir / "log.txt", "a", encoding="utf-8", buffering=1)
        self._metrics = open(self.dir / "metrics.jsonl", "a", encoding="utf-8", buffering=1)

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self._log.write(line + "\n")
        print(line, flush=True)

    def metric(self, rec: dict[str, Any]) -> None:
        rec["t"] = time.time()
        self._metrics.write(json.dumps(rec) + "\n")

    def set_status(self, **kw) -> None:
        self.status.update(kw)
        self.status["pid"] = os.getpid()
        self.status["heartbeat"] = time.time()
        self.status["control_seq"] = self._ctrl_seq
        store.write_json(self.dir / "status.json", self.status)

    def poll_control(self) -> list[str]:
        """New commands since the last poll. Cheap: only re-reads the file when its mtime changed.

        ``set`` commands (live-adjustable settings) are queued in ``self.pending_set``.
        """
        p = self.dir / "control.json"
        try:
            mt = p.stat().st_mtime_ns
        except FileNotFoundError:
            return []
        if mt == self._ctrl_mtime:
            return []
        self._ctrl_mtime = mt
        cmds = (store.read_json(p, {}) or {}).get("commands", [])
        new = [c for c in cmds if c.get("seq", 0) > self._ctrl_seq]
        if new:
            self._ctrl_seq = max(c["seq"] for c in new)
        out = []
        for c in new:
            if c["cmd"] == "set":
                self.pending_set.update(c.get("params") or {})
            else:
                out.append(c["cmd"])
        return out


class Stop(Exception):
    pass


def _set_seed(seed: int):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s):
    import numpy as np
    import torch
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def build_model(run: Run, device: str):
    """Create (from scratch) or load (base model) the model and prepare it for the training method."""
    import torch

    from .. import registry
    from ..model.config import ModelConfig
    from ..model.peft import apply_adapters, apply_lora, quantize_linears
    from ..model.transformer import MakeAIForCausalLM

    c = run.cfg
    method = c["method"]
    tr = c["training"]
    prec = tr["precision"]
    if method == "from_scratch":
        cfg = ModelConfig.from_dict(c["model_config"])
        cfg.dropout = float(tr.get("dropout", cfg.dropout))
        model = MakeAIForCausalLM(cfg).to(device)
        run.log(f"initialised {cfg.param_count():,} parameters from scratch")
        return model, cfg
    from safetensors.torch import load_file
    base_uid = c["base_model"]
    cfg = registry.get_config(base_uid)
    cfg.dropout = float(tr.get("dropout", 0.0))
    sd = load_file(str(registry.weights_path(base_uid)))
    if method == "full":
        dtype = torch.float32
        model = registry.build_with_weights(cfg, sd, device, dtype)
        run.log(f"loaded base {base_uid} for full training ({cfg.param_count():,} parameters, fp32 master weights)")
        return model, cfg
    compute = torch.bfloat16 if prec == "bf16" else (torch.float16 if prec == "fp16" else torch.float32)
    quant = tr.get("base_quant") if tr.get("base_quant") in STORAGE_ONLY else None
    if method == "qlora" and not quant:
        quant = "nf4"
    model = registry.build_with_weights(cfg, sd, "cpu" if quant else device, compute)
    del sd
    if quant:
        lc = tr.get("lora") or {}
        n = quantize_linears(model, quant, compute_dtype=compute, double_quant=bool(lc.get("double_quant", True)),
                             device=device)
        model.to(device)
        run.log(f"quantised {n} linear layers of the frozen base to {quant.upper()}"
                f"{' with double quantisation' if lc.get('double_quant', True) and quant != 'int8' else ''}")
    if method in ("lora", "qlora"):
        lc = tr["lora"]
        n = apply_lora(model, int(lc["rank"]), float(lc["alpha"]), float(lc.get("dropout", 0.0)),
                       lc.get("target_modules") or None, lc.get("bias", "none"))
        run.log(f"LoRA r={lc['rank']} alpha={lc['alpha']} on {n} modules")
    elif method == "adapter":
        n = apply_adapters(model, int(tr["adapter"]["bottleneck"]))
        run.log(f"{n} bottleneck adapters (size {tr['adapter']['bottleneck']})")
    # trainable params stay in fp32 for stable optimisation
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model, cfg


def main(run_dir: str) -> int:
    run = Run(Path(run_dir))
    c = run.cfg
    try:
        return _train(run)
    except Stop:
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        run.log("FAILED: " + tb)
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        if "out of memory" in tb.lower():
            msg = ("Out of GPU memory (limited to the VRAM that was free when the run started) - lower the micro "
                   "batch size or context length, enable gradient checkpointing, close GPU-heavy applications, "
                   "or use QLoRA.")
        run.set_status(state="failed", message=msg, ended_at=time.time())
        try:
            from .. import registry
            registry.settle_status(c["model_uid"], "failed")
        except Exception:
            pass
        return 1


def _train(run: Run) -> int:
    import numpy as np
    import torch

    from .. import registry
    from ..model.peft import trainable_state_dict
    from .checkpoint import CheckpointManager
    from .data import BatchSource, PackedSplit, Prefetcher, val_batches
    from .optim import build_optimizer, build_scheduler

    c = run.cfg
    tr = c["training"]
    ck = c.get("checkpoint", {})
    prec = tr["precision"]
    if prec in STORAGE_ONLY:
        raise ValueError(f"{prec.upper()} is a storage format for a frozen base model (QLoRA / 8-bit LoRA) and "
                         f"for inference. Trainable weights need FP32, TF32, FP16 or BF16.")
    if prec not in TRAINABLE_PRECISIONS:
        raise ValueError(f"unknown precision {prec}")
    device = c.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")
    if prec == "bf16" and device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support BF16 - use FP16")
    if device.startswith("cuda") and not tr.get("allow_vram_overflow", False):
        torch.cuda.init()
        free, total = torch.cuda.mem_get_info()
        frac = max(0.05, min(1.0, (free - 192 * 2**20) / total))
        torch.cuda.set_per_process_memory_fraction(frac)
        run.log(f"VRAM: {free / 2**20:,.0f} MB free of {total / 2**20:,.0f} MB (other applications use the rest). "
                f"This run is capped at {frac * total / 2**20:,.0f} MB so it cannot spill into shared system memory, "
                f"which on Windows slows training 10-100x instead of failing.")
    torch.backends.cuda.matmul.allow_tf32 = prec in ("tf32", "bf16", "fp16")
    torch.backends.cudnn.allow_tf32 = prec in ("tf32", "bf16", "fp16")
    seed = int(tr.get("seed", 1337))
    _set_seed(seed)

    run.set_status(state="starting", message="loading model", started_at=run.status.get("started_at") or time.time())
    model, mcfg = build_model(run, device)
    if tr.get("gradient_checkpointing"):
        model.set_gradient_checkpointing(True)
    n_params = model.num_parameters()
    n_train = model.num_parameters(trainable_only=True)
    run.log(f"parameters: {n_params:,} total, {n_train:,} trainable ({100 * n_train / max(1, n_params):.2f}%)")

    ctx = int(tr["context_length"])
    splits = []
    val_splits = []
    for d in c["datasets"]:
        if not d.get("prepared_path"):
            from ..data import datasets as dsets
            run.set_status(state="starting", message=f"tokenising dataset {d['id']}")
            p = c.get("data_prep", {})
            info = dsets.prepare(d["id"], c["tokenizer_dir"], ctx, val_ratio=float(p.get("val_ratio", 0.02)),
                                 test_ratio=float(p.get("test_ratio", 0.0)), packing=bool(p.get("packing", True)),
                                 min_tokens=int(p.get("min_tokens", 1)), max_tokens=int(p.get("max_tokens", 0)),
                                 seed=int(p.get("seed", 1337)),
                                 progress=lambda i: run.set_status(message=f"tokenising dataset {d['id']}: {i:,} records"))
            d["prepared_path"] = info["path"]
            run.log(f"dataset {d['id']}: {info['splits']['train']['tokens']:,} train / "
                    f"{info['splits']['val']['tokens']:,} val tokens")
        info = store.read_json(Path(d["prepared_path"]) / "prepared.json")
        if int(info["context_length"]) != ctx:
            raise ValueError(f"dataset {d['id']} was prepared for context {info['context_length']}, run uses {ctx}")
        splits.append((PackedSplit(d["prepared_path"], "train", ctx, info["dtype"]), float(d.get("weight", 1.0))))
        val_splits.append(PackedSplit(d["prepared_path"], "val", ctx, info["dtype"]))
    micro = int(tr["micro_batch_size"])
    accum = int(tr["gradient_accumulation"])
    src = BatchSource(splits, micro, seed)
    rows_per_step = micro * accum
    epochs = float(tr.get("epochs") or 1)
    total_steps = int(tr.get("max_steps") or 0) or max(1, math.ceil(epochs * src.total_rows / rows_per_step))
    steps_per_epoch = max(1, src.total_rows / rows_per_step)
    total_epochs = max(1, math.ceil(total_steps / steps_per_epoch))
    run.log(f"data: {src.total_rows:,} training sequences of {ctx} tokens; {total_steps:,} steps, "
            f"{rows_per_step} sequences ({rows_per_step * ctx:,} tokens) per step")

    opt = build_optimizer(tr["optimizer"], model, float(tr["learning_rate"]), float(tr.get("weight_decay", 0.0)),
                          betas=tr.get("betas") or (0.9, 0.95))
    sched = build_scheduler(opt, tr.get("scheduler", "cosine"), total_steps, int(tr.get("warmup_steps", 0)),
                            float(tr.get("min_lr_ratio", 0.1)))
    use_amp = prec in ("fp16", "bf16") and device != "cpu"
    amp_dtype = torch.bfloat16 if prec == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(prec == "fp16" and device.startswith("cuda")))
    autocast = (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)) if use_amp else nullcontext

    ckm = CheckpointManager(run.dir / "checkpoints", keep=int(ck.get("keep", 3)), keep_best=ck.get("save_best", True),
                            clean_leftovers=True)
    step, batch_index, best_val = 0, 0, float("inf")
    elapsed_before = 0.0
    tokens_seen = 0
    resume = c.get("resume_from")
    if resume:
        rdir = run.dir / "checkpoints" / resume if not Path(resume).is_absolute() else Path(resume)
        from safetensors.torch import load_file
        sd = load_file(str(rdir / "model.safetensors"))
        if c["method"] in ("from_scratch", "full"):
            if mcfg.tie_weights:
                sd.pop("lm_head.weight", None)
            model.load_state_dict(sd, strict=False)
        else:
            from ..model.peft import load_trainable
            load_trainable(model, sd)
        st = torch.load(rdir / "state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        step, batch_index = st["step"], st["batch_index"]
        best_val = st.get("best_val", best_val)
        elapsed_before = st.get("elapsed", 0.0)
        tokens_seen = st.get("tokens_seen", 0)
        _set_rng_state(st["rng"])
        run.log(f"resumed from {rdir.name}: step {step}, optimizer/scheduler/RNG state restored")

    def write_ckpt(tmp: Path, snapshot_step: int, snapshot_batch: int):
        from safetensors.torch import save_file
        if c["method"] in ("from_scratch", "full"):
            sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
            if mcfg.tie_weights:
                sd.pop("lm_head.weight", None)
        else:
            sd = trainable_state_dict(model)
        save_file(sd, str(tmp / "model.safetensors"), metadata={"makeai.run": c["run_id"], "step": str(snapshot_step)})
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None, "step": snapshot_step,
                    "batch_index": snapshot_batch, "best_val": best_val, "rng": _rng_state(),
                    "elapsed": elapsed_before + active_time(), "tokens_seen": tokens_seen}, tmp / "state.pt")

    def checkpoint(kind: str, extra: dict | None = None):
        run.set_status(message=f"saving {kind} checkpoint")
        e = ckm.save(step, kind, lambda tmp: write_ckpt(tmp, step, batch_index), extra or {})
        run.log(f"checkpoint {e['name']} ({e['bytes'] / 2**20:.1f} MB in {e['seconds']}s)")
        run.set_status(last_checkpoint=e["name"], message="")
        run.metric({"type": "checkpoint", "step": step, **e})
        return e

    @torch.no_grad()
    def evaluate(max_batches: int) -> dict[str, float] | None:
        model.eval()
        tot_loss = tot_tok = correct = 0.0
        for vs in val_splits:
            for x, y in val_batches(vs, micro, max_batches):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast():
                    logits, _, _ = model(x)
                logits = logits.float()
                valid = y != -100
                n = valid.sum().item()
                if not n:
                    continue
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                                         ignore_index=-100, reduction="sum").item()
                tot_loss += loss
                tot_tok += n
                correct += ((logits.argmax(-1) == y) & valid).sum().item()
        model.train()
        if not tot_tok:
            return None
        vl = tot_loss / tot_tok
        return {"val_loss": vl, "val_ppl": math.exp(min(vl, 50)), "token_accuracy": correct / tot_tok,
                "val_tokens": int(tot_tok)}

    # ------------------------------------------------------------ main loop
    t_active_start = time.time()
    paused_total = 0.0

    def active_time() -> float:
        return time.time() - t_active_start - paused_total

    def handle(cmds: list[str], mid_step: bool) -> None:
        nonlocal paused_total
        for cmd in cmds:
            if cmd == "stop":
                raise _StopRequested()
            if cmd == "checkpoint" and not mid_step:
                checkpoint("manual")
            if cmd == "pause":
                if mid_step:
                    raise _PauseRequested()
                pause_loop()

    def pause_loop():
        nonlocal paused_total
        t0 = time.time()
        run.set_status(state="paused", message="paused", paused_at=t0)
        run.log(f"paused at step {step}")
        run.metric({"type": "event", "event": "paused", "step": step})
        if ck.get("checkpoint_on_pause", True):
            checkpoint("pause")
            run.set_status(state="paused", message="paused (state saved)")
        while True:
            cmds = run.poll_control()
            if "stop" in cmds:
                paused_total += time.time() - t0
                raise _StopRequested()
            if "checkpoint" in cmds:
                checkpoint("manual")
                run.set_status(state="paused", message="paused")
            if "resume" in cmds:
                break
            run.set_status()  # heartbeat
            time.sleep(0.25)
        paused_total += time.time() - t0
        run.set_status(state="running", message="", paused_at=None)
        run.log(f"resumed at step {step}")
        run.metric({"type": "event", "event": "resumed", "step": step})

    model.train()
    pre = Prefetcher(src, batch_index, int(tr.get("dataloader_workers", 2)), pin=device.startswith("cuda"))
    eval_every = int(tr.get("eval_every") or 0)
    save_every = int(ck.get("save_every") or tr.get("save_every") or 0)
    eval_batches = int(c.get("eval", {}).get("max_batches", 50))
    clip = float(tr.get("grad_clip") or 0)
    nan_streak = 0
    errors = 0
    ema_sps = None
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    run.set_status(state="running", message="", total_steps=total_steps, total_epochs=total_epochs,
                   params=n_params, trainable_params=n_train)
    run.log("training started")
    run.metric({"type": "event", "event": "started", "step": step, "total_steps": total_steps})
    try:
        while step < total_steps:
            handle(run.poll_control(), mid_step=False)
            if run.pending_set:
                ps, run.pending_set = run.pending_set, {}
                if "dataloader_workers" in ps:
                    tr["dataloader_workers"] = max(1, min(32, int(ps["dataloader_workers"])))
                    pre.close()
                    pre = Prefetcher(src, batch_index, tr["dataloader_workers"], pin=device.startswith("cuda"))
                if "eval_every" in ps:
                    eval_every = int(ps["eval_every"])
                if "save_every" in ps:
                    save_every = int(ps["save_every"])
                run.log(f"settings changed while running: {ps}")
                run.metric({"type": "event", "event": "settings_changed", "step": step, "params": ps})
            t0 = time.perf_counter()
            wait0 = pre.wait_s
            loss_sum = torch.zeros((), device=device)
            tok_step = 0
            start_batch = batch_index
            try:
                for _ in range(accum):
                    cmds = run.poll_control()
                    if cmds:
                        handle(cmds, mid_step=True)
                    b, x, y = pre.get()
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with autocast():
                        _, loss, _ = model(x, labels=y)
                    scaler.scale(loss / accum).backward()
                    loss_sum += loss.detach()
                    tok_step += x.numel()
                    batch_index = b + 1
            except _PauseRequested:
                # discard the partial accumulation: the model is unchanged since the last optimizer step
                opt.zero_grad(set_to_none=True)
                pre.close()
                batch_index = start_batch
                pause_loop()
                pre = Prefetcher(src, batch_index, int(tr.get("dataloader_workers", 2)), pin=device.startswith("cuda"))
                continue
            except _StopRequested:
                opt.zero_grad(set_to_none=True)
                batch_index = start_batch
                raise
            if scaler.is_enabled():
                scaler.unscale_(opt)
            params = [p for p in model.parameters() if p.requires_grad]
            gn = torch.nn.utils.clip_grad_norm_(params, clip if clip > 0 else float("inf"))
            loss_val = (loss_sum / accum).item()
            gn_val = float(gn)
            finite = math.isfinite(loss_val) and math.isfinite(gn_val)
            if finite or scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            opt.zero_grad(set_to_none=True)
            lr = sched.get_last_lr()[0]
            sched.step()
            step += 1
            tokens_seen += tok_step
            if not finite:
                nan_streak += 1
                errors += 1
                run.log(f"non-finite loss/grad at step {step} (loss={loss_val}, grad_norm={gn_val}); step skipped")
                if nan_streak >= 10 and not scaler.is_enabled():
                    raise RuntimeError("training diverged: 10 consecutive non-finite steps - lower the learning rate")
            else:
                nan_streak = 0
            dt = time.perf_counter() - t0
            wait = pre.wait_s - wait0
            sps = 1.0 / dt
            ema_sps = sps if ema_sps is None else 0.9 * ema_sps + 0.1 * sps
            rec = {"type": "step", "step": step, "total_steps": total_steps,
                   "epoch": min(total_epochs, 1 + int(batch_index * micro / max(1, src.total_rows))),
                   "epoch_progress": (batch_index * micro / max(1, src.total_rows)),
                   "loss": loss_val, "ppl": math.exp(min(loss_val, 50)) if finite else None,
                   "lr": lr, "grad_norm": gn_val if math.isfinite(gn_val) else None,
                   "step_time": dt, "data_wait": wait, "data_wait_frac": min(1.0, wait / dt),
                   "tokens_per_s": tok_step / dt, "samples_per_s": micro * accum / dt, "steps_per_s": sps,
                   "tokens_seen": tokens_seen, "elapsed": elapsed_before + active_time(),
                   "eta_s": (total_steps - step) / ema_sps, "errors": errors,
                   "loss_scale": scaler.get_scale() if scaler.is_enabled() else None}
            if device.startswith("cuda"):
                rec.update({"mem_alloc_mb": torch.cuda.memory_allocated() / 2**20,
                            "mem_reserved_mb": torch.cuda.memory_reserved() / 2**20,
                            "mem_peak_mb": torch.cuda.max_memory_allocated() / 2**20})
            run.metric(rec)
            if step == 1 or step % 10 == 0 or step == total_steps:
                run.set_status(step=step, epoch=rec["epoch"], last_loss=loss_val)
            if step % 50 == 0 or step == 1:
                run.log(f"step {step}/{total_steps} loss {loss_val:.4f} lr {lr:.2e} "
                        f"{rec['tokens_per_s']:,.0f} tok/s")
            if eval_every and (step % eval_every == 0 or step == total_steps):
                ev = evaluate(eval_batches)
                if ev:
                    run.metric({"type": "eval", "step": step, **ev})
                    run.log(f"eval step {step}: val_loss {ev['val_loss']:.4f} ppl {ev['val_ppl']:.2f} "
                            f"token_acc {ev['token_accuracy']:.3f}")
                    run.set_status(last_val_loss=ev["val_loss"], last_eval=ev, last_eval_step=step)
                    if ev["val_loss"] < best_val:
                        best_val = ev["val_loss"]
                        run.set_status(best_val_loss=best_val)
                        if ck.get("save_best", True):
                            checkpoint("best", {"val_loss": ev["val_loss"]})
            if save_every and step % save_every == 0 and step != total_steps:
                checkpoint("auto")
    except _StopRequested:
        pre.close()
        run.set_status(state="stopping", message="stopping - saving latest safe checkpoint")
        run.log(f"stop requested by user at step {step}")
        saved = None
        save_error = None
        if ck.get("save_on_kill", True) and step > 0:
            try:
                saved = checkpoint("kill")
            except Exception as e:       # the stop itself must still complete
                save_error = str(e)
                run.log(f"kill checkpoint could not be saved: {e}")
        try:
            _finish_model(run, model, mcfg, c, step, final=False)
        except Exception as e:
            run.log(f"could not write partial weights to My AIs: {e}")
        del model, opt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        run.set_status(state="terminated", ended_at=time.time(),
                       message="TERMINATED BY USER" + (f" (checkpoint not saved: {save_error})" if save_error else ""),
                       last_checkpoint=saved["name"] if saved else run.status.get("last_checkpoint"))
        run.metric({"type": "event", "event": "terminated", "step": step})
        run.log("TERMINATED BY USER - GPU memory released")
        raise Stop()
    pre.close()
    last_eval = run.status.get("last_eval_step")
    ev = evaluate(eval_batches) if last_eval != step else run.status.get("last_eval")
    if ev and last_eval != step:
        run.metric({"type": "eval", "step": step, "final": True, **ev})
        run.log(f"final eval: val_loss {ev['val_loss']:.4f} ppl {ev['val_ppl']:.2f} token_acc {ev['token_accuracy']:.3f}")
    final = checkpoint("final", {"val_loss": ev["val_loss"] if ev else None})
    _finish_model(run, model, mcfg, c, step, final=True, final_eval=ev)
    run.set_status(state="completed", message="training complete", ended_at=time.time(), step=step,
                   final_eval=ev, last_checkpoint=final["name"])
    run.metric({"type": "event", "event": "completed", "step": step})
    run.log("training complete")
    return 0


class _StopRequested(Exception):
    pass


class _PauseRequested(Exception):
    pass


def _finish_model(run: Run, model, mcfg, c: dict, step: int, final: bool, final_eval=None):
    """Write the trained weights into the AI's registry entry (My AIs)."""
    import torch

    from .. import registry
    from ..model.peft import trainable_state_dict
    if step == 0:
        return
    uid = c["model_uid"]
    m = registry.load_manifest(uid)
    if c["method"] in ("from_scratch", "full"):
        sd = model.state_dict()
        if mcfg.tie_weights:
            sd = {k: v for k, v in sd.items() if k != "lm_head.weight"}
        dtype = torch.float32 if mcfg.param_count() < 2e8 else torch.bfloat16
        sd = {k: v.detach().to(dtype).cpu() for k, v in sd.items()}
        registry.save_weights(uid, sd, {"makeai.run": c["run_id"], "makeai.step": str(step)})
    else:
        from safetensors.torch import save_file
        tr = c["training"]
        adir = registry.adapter_dir(uid, "default")
        adir.mkdir(parents=True, exist_ok=True)
        meta = registry.attribution_metadata(m)
        sd = trainable_state_dict(model)
        tmp = adir / "adapter.safetensors.tmp"
        save_file(sd, str(tmp), metadata=meta)
        tmp.replace(adir / "adapter.safetensors")
        entry = {"name": "default", "kind": c["method"], "path": "adapters/default/adapter.safetensors",
                 "active": True, "run": c["run_id"], "step": step}
        if c["method"] in ("lora", "qlora"):
            lc = tr["lora"]
            entry.update({"rank": int(lc["rank"]), "alpha": float(lc["alpha"]),
                          "target_modules": lc.get("target_modules"), "bias": lc.get("bias", "none"),
                          "base_quant": tr.get("base_quant")})
        else:
            entry["bottleneck"] = int(tr["adapter"]["bottleneck"])
        store.write_json(adir / "adapter_config.json", entry)
        m["adapters"] = [entry]
    m["status"] = "trained" if final else "partially_trained"
    m["runnable"] = True
    runs = m.setdefault("training_runs", [])
    if c["run_id"] not in runs:
        runs.append(c["run_id"])
    if final_eval:
        m["last_eval"] = {**final_eval, "run": c["run_id"], "step": step}
    m.setdefault("provenance", []).append({"event": "trained" if final else "trained_partial", "run": c["run_id"],
                                           "steps": step, "at": store.now_iso(), "method": c["method"]})
    registry.save_manifest(m)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
