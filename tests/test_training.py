"""Training engine acceptance tests - real training processes on real data."""
import json
import shutil
import time

import psutil
import pytest
import torch

from conftest import tiny_config, wait_state
from makeai import registry
from makeai.train.runs import RunManager, read_metrics, read_status

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_training_loss_updates_and_decreases(trained):
    recs, _ = read_metrics(trained["run_id"])
    steps = [r for r in recs if r["type"] == "step"]
    evals = [r for r in recs if r["type"] == "eval"]
    assert len(steps) == 150
    first = sum(s["loss"] for s in steps[:10]) / 10
    last = sum(s["loss"] for s in steps[-10:]) / 10
    assert last < first - 1.0, (first, last)
    assert len(evals) >= 3 and evals[-1]["val_loss"] < evals[0]["val_loss"]
    assert all(0 <= e["token_accuracy"] <= 1 for e in evals)
    m = registry.load_manifest(trained["uid"])
    assert m["status"] == "trained" and m["runnable"] and registry.weights_path(trained["uid"]).exists()


def test_eta_is_measured_and_updates(trained):
    recs, _ = read_metrics(trained["run_id"])
    steps = [r for r in recs if r["type"] == "step"]
    etas = [s["eta_s"] for s in steps]
    assert all(e >= 0 for e in etas)
    assert etas[10] > etas[-2] > etas[-1] == 0          # counts down as steps complete
    s = steps[50]
    assert abs(s["eta_s"] - (150 - 50) / s["steps_per_s"]) / s["eta_s"] < 1.0    # from measured throughput
    assert s["tokens_per_s"] > 0 and s["samples_per_s"] > 0 and s["elapsed"] > 0


def test_checkpoints_best_cleanup_and_integrity(trained):
    from safetensors.torch import load_file
    idx = json.load(open(registry.store.runs_dir() / trained["run_id"] / "checkpoints" / "index.json"))
    kinds = [c["kind"] for c in idx]
    assert "final" in kinds and "best" in kinds
    assert sum(k == "auto" for k in kinds) <= 2                                   # max checkpoint count
    for c in idx:
        sd = load_file(str(registry.store.runs_dir() / trained["run_id"] / "checkpoints" / c["name"] / "model.safetensors"))
        assert all(torch.isfinite(v).all() for v in sd.values())


def _start_long_run(corpus, name, **tr):
    from makeai.data.tokenizer import load_tokenizer
    tok = load_tokenizer(corpus["tok_dir"])
    cfg = tiny_config(tok.vocab_size, n_layers=4, hidden_size=256, intermediate_size=704, n_heads=4, n_kv_heads=4, head_dim=64)
    m = registry.create_model(name=name, creator_name="Eron", creator_username="eron", config=cfg)
    shutil.copytree(corpus["tok_dir"], registry.tokenizer_dir(m["uid"]))
    rm = RunManager()
    training = {"precision": "bf16" if torch.cuda.is_available() else "fp32", "optimizer": "adamw", "scheduler": "cosine",
                "learning_rate": 1e-3, "warmup_steps": 5, "micro_batch_size": 16, "gradient_accumulation": 2,
                "context_length": 128, "max_steps": 100000, "eval_every": 0, "seed": 1, **tr}
    rid = rm.create({"model_uid": m["uid"], "model_name": name, "method": "from_scratch", "model_config": cfg.to_dict(),
                     "tokenizer_dir": str(registry.tokenizer_dir(m["uid"])), "datasets": [{"id": corpus["ds"], "weight": 1}],
                     "training": training, "checkpoint": {"save_every": 0}})
    rm.start(rid)
    return rm, rid


def _vram_used():
    import pynvml
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used / 2**20


@cuda
def test_pause_preserves_state_and_resume_continues(corpus):
    rm, rid = _start_long_run(corpus, "PauseTest")
    wait_state(rid, ("running",), pred=lambda s: s.get("step", 0) >= 20)
    rm.pause(rid)
    s = wait_state(rid, ("paused",))
    step_paused = s["step"]
    time.sleep(1.5)
    assert read_status(rid)["state"] == "paused"
    recs, _ = read_metrics(rid)
    paused_at = max(r["step"] for r in recs if r["type"] == "step")
    time.sleep(1.0)
    recs2, _ = read_metrics(rid)
    assert max(r["step"] for r in recs2 if r["type"] == "step") == paused_at          # no steps while paused
    rm.resume(rid)
    wait_state(rid, ("running",), pred=lambda s: s.get("step", 0) >= paused_at + 10)
    recs3, _ = read_metrics(rid)
    steps = [r["step"] for r in recs3 if r["type"] == "step"]
    assert steps == list(range(1, len(steps) + 1))                                     # step counter preserved
    rm.kill(rid, instant=True)
    rm.shutdown()


@cuda
def test_kill_switch_terminates_saves_and_releases_resources(corpus):
    base = _vram_used()
    rm, rid = _start_long_run(corpus, "KillTest")
    s = wait_state(rid, ("running",), pred=lambda s: s.get("step", 0) >= 15)
    pid = s["pid"]
    during = _vram_used()
    assert during > base + 200                                                         # the run really holds VRAM
    t0 = time.time()
    rm.kill(rid)
    s = wait_state(rid, ("terminated",), timeout=120)
    assert s["message"] == "TERMINATED BY USER"
    while psutil.pid_exists(pid) and time.time() - t0 < 60:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    time.sleep(1.0)
    assert _vram_used() < base + 150                                                   # GPU memory released
    ck = s["last_checkpoint"]
    assert ck and ck.endswith("-kill")
    # resume from the kill checkpoint: optimizer/scheduler/step restored
    killed_step = int(ck.split("-")[1])
    rm.start(rid, resume_from=ck)
    wait_state(rid, ("running",), pred=lambda s: s.get("step", 0) >= killed_step + 5)
    recs, _ = read_metrics(rid)
    steps = [r["step"] for r in recs if r["type"] == "step"]
    assert killed_step + 1 in steps[killed_step:]
    # hard kill while running: marked interrupted, existing checkpoint untouched
    rm.kill(rid, instant=True)
    s = read_status(rid)
    assert s["state"] == "interrupted" and "TERMINATED BY USER" in s["message"]
    from safetensors.torch import load_file
    load_file(str(registry.store.runs_dir() / rid / "checkpoints" / ck / "model.safetensors"))
    rm.shutdown()


@cuda
def test_vram_cap_prevents_shared_memory_spill(corpus):
    rm, rid = _start_long_run(corpus, "CapTest")
    wait_state(rid, ("running",), pred=lambda s: s.get("step", 0) >= 2)
    log = (registry.store.runs_dir() / rid / "log.txt").read_text(encoding="utf-8")
    assert "capped at" in log
    rm.kill(rid, instant=True)
    rm.shutdown()


def test_optimizers_and_schedulers_train():
    from makeai.train.optim import OPTIMIZERS, SCHEDULERS, build_optimizer, build_scheduler, lr_lambda
    torch.manual_seed(0)
    for name in OPTIMIZERS:
        if name == "adamw8bit" and not torch.cuda.is_available():
            continue
        dev = "cuda" if name == "adamw8bit" else "cpu"
        from makeai.model import MakeAIForCausalLM
        m = MakeAIForCausalLM(tiny_config(64, n_layers=1, hidden_size=64, intermediate_size=128, n_heads=2, n_kv_heads=2, head_dim=32)).to(dev)
        opt = build_optimizer(name, m, 3e-3 if name != "sgd" else 0.3, 0.0)
        x = torch.randint(0, 64, (8, 16), device=dev)
        l0 = None
        for _ in range(30):
            _, loss, _ = m(x, labels=x)
            l0 = l0 or loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
        assert loss.item() < l0, name
    for s in SCHEDULERS:
        f = lr_lambda(s, 100, 10)
        assert 0 < f(0) <= 1 and f(50) > 0 and f(99) <= f(10) + 1e-9, s


def test_checkpoint_listing_never_deletes_a_save_in_progress(tmp_path):
    """Regression: the UI lists checkpoints while the trainer writes one; listing must not touch temp dirs."""
    import os
    from makeai.train.checkpoint import CheckpointManager
    live = tmp_path / f".tmp-step-00000009-kill-{os.getppid()}"      # written by a process that is alive
    dead = tmp_path / ".tmp-step-00000001-auto-999999"               # left behind by a process that is gone
    live.mkdir()
    dead.mkdir()
    CheckpointManager(tmp_path).index()                               # what the server does on every poll
    assert live.exists() and dead.exists()
    CheckpointManager(tmp_path, clean_leftovers=True)                 # what a new trainer does at start
    assert live.exists() and not dead.exists()
