"""Shared fixtures. Every test session uses its own temporary MAKEAI_HOME."""
from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STDLIB = Path(os.__file__).parent          # real text corpus: the Python standard library source
QWEN = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than ~30 s or needs a large model")


@pytest.fixture(scope="session")
def home():
    d = Path(tempfile.mkdtemp(prefix="makeai-test-"))
    old = os.environ.get("MAKEAI_HOME")
    os.environ["MAKEAI_HOME"] = str(d)
    yield d
    if old is None:
        os.environ.pop("MAKEAI_HOME", None)
    else:
        os.environ["MAKEAI_HOME"] = old
    shutil.rmtree(d, ignore_errors=True)


def switch_home(path: Path) -> None:
    os.environ["MAKEAI_HOME"] = str(path)


@pytest.fixture(scope="session")
def corpus(home):
    """Dataset of real Python source files and a byte-level BPE tokenizer trained on it."""
    from makeai.data import datasets as D
    from makeai.data.tokenizer import train_tokenizer
    files = sorted(STDLIB.glob("*.py"))[:60]
    ds = D.create_dataset("stdlib-sample", [str(f) for f in files], {})
    tok_dir = home / "tokenizers" / "tok-test"
    train_tokenizer("bpe", D.iter_texts([ds["id"]]), 2048, tok_dir)
    return {"ds": ds["id"], "tok_dir": tok_dir}


def tiny_config(vocab: int, **kw):
    from makeai.model.config import ModelConfig
    base = dict(vocab_size=vocab, n_layers=2, hidden_size=96, intermediate_size=256, n_heads=4, n_kv_heads=2,
                head_dim=24, context_length=128)
    base.update(kw)
    return ModelConfig(**base)


def wait_state(run_id, states, timeout=300, pred=None):
    from makeai.train.runs import read_status
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = read_status(run_id)
        if s.get("state") in states and (pred is None or pred(s)):
            return s
        if s.get("state") in ("failed", "crashed") and "failed" not in states:
            raise AssertionError(f"run {s.get('state')}: {s.get('message')} {s.get('stdout_tail', '')[-800:]}")
        time.sleep(0.2)
    raise AssertionError(f"timeout waiting for {states}; last status {read_status(run_id)}")


@pytest.fixture(scope="session")
def trained(home, corpus):
    """A tiny AI created from scratch and trained for real on the stdlib dataset."""
    import shutil as sh

    from makeai import registry
    from makeai.data.tokenizer import load_tokenizer
    from makeai.train.runs import RunManager
    tok = load_tokenizer(corpus["tok_dir"])
    cfg = tiny_config(tok.vocab_size)
    m = registry.create_model(name="EronAI", creator_name="Eron", creator_username="eron",
                              description="A general-purpose AI focused on coding and reasoning.", version="1.0",
                              tags=["coding"], config=cfg)
    sh.copytree(corpus["tok_dir"], registry.tokenizer_dir(m["uid"]))
    rm = RunManager()
    run_id = rm.create({"model_uid": m["uid"], "model_name": m["name"], "method": "from_scratch",
                        "model_config": cfg.to_dict(), "tokenizer_dir": str(registry.tokenizer_dir(m["uid"])),
                        "datasets": [{"id": corpus["ds"], "weight": 1}],
                        "training": {"precision": "bf16", "optimizer": "adamw", "scheduler": "cosine",
                                     "learning_rate": 2e-3, "warmup_steps": 10, "weight_decay": 0.1, "grad_clip": 1.0,
                                     "micro_batch_size": 16, "gradient_accumulation": 1, "context_length": 128,
                                     "max_steps": 150, "eval_every": 50, "dataloader_workers": 2, "seed": 7},
                        "checkpoint": {"save_every": 50, "keep": 2}, "eval": {"max_batches": 10}})
    rm.start(run_id)
    wait_state(run_id, ("completed",), timeout=600)
    rm.shutdown()
    return {"uid": m["uid"], "run_id": run_id, "tok_dir": corpus["tok_dir"], "ds": corpus["ds"]}
