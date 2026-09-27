"""Crash-safe checkpoints.

Each checkpoint is written to a hidden temp directory and renamed into place
only once every file is complete and fsynced, so an interrupted save (power
loss, hard kill) never damages an existing checkpoint. ``index.json`` is
updated atomically after the rename.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from .. import store


REQUIRED = ("model.safetensors", "state.pt")


class CheckpointManager:
    def __init__(self, root: Path, keep: int = 3, keep_best: bool = True, clean_leftovers: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep = max(1, keep)
        self.keep_best = keep_best
        if clean_leftovers:
            self.clean_leftovers()

    def clean_leftovers(self) -> None:
        """Remove temp dirs of saves that were interrupted - only if the process that wrote them is gone."""
        import psutil
        for p in self.root.glob(".tmp-*"):
            try:
                pid = int(p.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                pid = None
            if pid and pid != os.getpid() and psutil.pid_exists(pid):
                continue
            shutil.rmtree(p, ignore_errors=True)

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def index(self) -> list[dict[str, Any]]:
        """Registered checkpoints that are complete on disk."""
        idx = store.read_json(self.index_path, []) or []
        return [c for c in idx if all((self.root / c["name"] / f).is_file() for f in REQUIRED)]

    def save(self, step: int, kind: str, writer, info: dict[str, Any]) -> dict[str, Any]:
        """``writer(tmp_dir)`` writes the files. ``kind``: auto | best | manual | pause | kill | final."""
        name = f"step-{step:08d}-{kind}"
        tmp = self.root / f".tmp-{name}-{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        t0 = time.time()
        writer(tmp)
        missing = [f for f in REQUIRED if not (tmp / f).is_file()]
        if missing:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError(f"checkpoint incomplete, missing {missing}; nothing was registered")
        for f in tmp.iterdir():
            with open(f, "rb+") as fh:
                os.fsync(fh.fileno())
        dst = self.root / name
        if dst.exists():
            shutil.rmtree(dst)
        store._replace_retry(str(tmp), dst)
        size = sum(f.stat().st_size for f in dst.iterdir())
        entry = {"name": name, "step": step, "kind": kind, "at": store.now_iso(), "bytes": size,
                 "seconds": round(time.time() - t0, 2), **info}
        idx = [c for c in self.index() if c["name"] != name] + [entry]
        store.write_json(self.index_path, idx)
        self.cleanup()
        return entry

    def cleanup(self) -> list[str]:
        """Keep the newest ``keep`` automatic checkpoints plus best/manual/kill/final ones."""
        idx = self.index()
        removable = [c for c in idx if c["kind"] in ("auto", "pause")]
        removable.sort(key=lambda c: c["step"])
        best = self.best()
        drop = removable[:-self.keep] if len(removable) > self.keep else []
        removed = []
        for c in drop:
            if self.keep_best and best and c["name"] == best["name"]:
                continue
            shutil.rmtree(self.root / c["name"], ignore_errors=True)
            removed.append(c["name"])
        if removed:
            store.write_json(self.index_path, [c for c in idx if c["name"] not in removed])
        # only one 'best' checkpoint is kept
        bests = sorted([c for c in self.index() if c["kind"] == "best"], key=lambda c: c["step"])
        for c in bests[:-1]:
            shutil.rmtree(self.root / c["name"], ignore_errors=True)
            removed.append(c["name"])
        if len(bests) > 1:
            store.write_json(self.index_path, [c for c in self.index()])
        return removed

    def latest(self) -> dict[str, Any] | None:
        idx = self.index()
        return max(idx, key=lambda c: (c["step"], c["at"])) if idx else None

    def best(self) -> dict[str, Any] | None:
        idx = [c for c in self.index() if c.get("val_loss") is not None]
        return min(idx, key=lambda c: c["val_loss"]) if idx else None

    def backup(self, name: str) -> Path:
        src = self.root / name
        if not src.exists():
            raise FileNotFoundError(name)
        dst = store.backups_dir() / f"{self.root.parent.name}-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copytree(src, dst)
        return dst
