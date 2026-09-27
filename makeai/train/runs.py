"""Server-side management of training runs (each run = one worker process)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from .. import store

TERMINAL = ("completed", "terminated", "interrupted", "failed", "crashed")
ACTIVE = ("starting", "running", "paused", "stopping")


def run_dir(run_id: str) -> Path:
    d = store.runs_dir() / run_id
    if not (d / "run.json").exists():
        raise FileNotFoundError(f"run {run_id} not found")
    return d


def read_status(run_id: str) -> dict[str, Any]:
    return store.read_json(run_dir(run_id) / "status.json", {}) or {}


def read_config(run_id: str) -> dict[str, Any]:
    return store.read_json(run_dir(run_id) / "run.json")


def list_runs() -> list[dict[str, Any]]:
    out = []
    for d in store.runs_dir().iterdir():
        cfg = store.read_json(d / "run.json")
        if not cfg:
            continue
        st = store.read_json(d / "status.json", {}) or {}
        out.append({"run_id": cfg["run_id"], "model_uid": cfg.get("model_uid"), "model_name": cfg.get("model_name"),
                    "method": cfg.get("method"), "created": cfg.get("created"), **{k: st.get(k) for k in (
                        "state", "message", "step", "total_steps", "last_loss", "best_val_loss", "started_at",
                        "ended_at", "last_checkpoint")}})
    out.sort(key=lambda r: r.get("created") or "", reverse=True)
    return out


def read_metrics(run_id: str, since_line: int = 0, kinds: tuple[str, ...] | None = None) -> tuple[list[dict], int]:
    p = run_dir(run_id) / "metrics.jsonl"
    if not p.exists():
        return [], 0
    out, n = [], 0
    with open(p, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if n <= since_line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break   # partially written last line; read it next time
            if kinds is None or r.get("type") in kinds:
                out.append(r)
    return out, n if out or n else since_line


def tail_log(run_id: str, lines: int = 200) -> list[str]:
    p = run_dir(run_id) / "log.txt"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()[-lines:]


class RunManager:
    def __init__(self, telemetry=None):
        self.procs: dict[str, subprocess.Popen] = {}
        self.telemetry = telemetry
        self.kill_deadlines: dict[str, float] = {}
        self.baseline_util: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.recover()
        threading.Thread(target=self._monitor, name="run-monitor", daemon=True).start()

    # --------------------------------------------------------------- launch
    def create(self, cfg: dict[str, Any]) -> str:
        run_id = cfg.get("run_id") or store.new_id("run")
        cfg["run_id"] = run_id
        cfg["created"] = store.now_iso()
        d = store.runs_dir() / run_id
        d.mkdir(parents=True, exist_ok=True)
        store.write_json(d / "run.json", cfg)
        store.write_json(d / "status.json", {"state": "created", "message": "", "step": 0})
        store.write_json(d / "control.json", {"commands": []})
        return run_id

    def start(self, run_id: str, resume_from: str | None = None) -> dict[str, Any]:
        d = run_dir(run_id)
        st = read_status(run_id)
        if st.get("state") in ACTIVE and self._alive(run_id):
            raise RuntimeError("run is already active")
        cfg = read_config(run_id)
        cfg["resume_from"] = resume_from
        store.write_json(d / "run.json", cfg)
        # keep command sequence monotonic across restarts
        ctrl = store.read_json(d / "control.json", {"commands": []})
        st.update({"state": "starting", "message": "launching worker", "ended_at": None,
                   "control_seq": max([c.get("seq", 0) for c in ctrl.get("commands", [])] + [0])})
        store.write_json(d / "status.json", st)
        if self.telemetry is not None:
            win = self.telemetry.window(5)
            utils = [s["gpus"][0]["util_pct"] for s in win if s.get("gpus") and s["gpus"][0].get("util_pct") is not None]
            if utils:
                self.baseline_util[run_id] = sum(utils) / len(utils)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["MAKEAI_HOME"] = str(store.home())
        pkg_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = open(d / "stdout.txt", "ab")
        p = subprocess.Popen([sys.executable, "-m", "makeai.train.worker", str(d)], cwd=pkg_root, env=env,
                             stdout=out, stderr=subprocess.STDOUT, creationflags=flags,
                             start_new_session=(os.name != "nt"))
        with self._lock:
            self.procs[run_id] = p
        if self.telemetry is not None:
            self.telemetry.proc_watch[p.pid] = run_id
        st["pid"] = p.pid
        store.write_json(d / "status.json", st)
        return st

    # -------------------------------------------------------------- control
    def send(self, run_id: str, cmd: str, params: dict | None = None) -> int:
        d = run_dir(run_id)
        ctrl = store.read_json(d / "control.json", {"commands": []}) or {"commands": []}
        seq = max([c.get("seq", 0) for c in ctrl["commands"]] + [int(read_status(run_id).get("control_seq", 0))]) + 1
        entry = {"seq": seq, "cmd": cmd, "at": time.time()}
        if params:
            entry["params"] = params
        ctrl["commands"] = (ctrl["commands"] + [entry])[-50:]
        store.write_json(d / "control.json", ctrl)
        return seq

    def pause(self, run_id: str):
        self._require_active(run_id)
        return self.send(run_id, "pause")

    def resume(self, run_id: str):
        st = read_status(run_id)
        if st.get("state") == "paused" and self._alive(run_id):
            return self.send(run_id, "resume")
        raise RuntimeError(f"run is {st.get('state')}, not paused")

    def checkpoint(self, run_id: str):
        self._require_active(run_id)
        return self.send(run_id, "checkpoint")

    def kill(self, run_id: str, instant: bool = False, grace_s: float = 90.0) -> dict[str, Any]:
        """Kill Switch. Graceful: stop new steps, save a safe checkpoint, release GPU, exit.
        ``instant`` (or the grace period running out) terminates the process tree immediately."""
        st = read_status(run_id)
        if st.get("state") in TERMINAL and not self._alive(run_id):
            return st
        if instant:
            return self._hard_kill(run_id, "instant kill requested")
        self.send(run_id, "stop")
        self.kill_deadlines[run_id] = time.time() + grace_s
        st["message"] = "kill requested - stopping"
        return st

    def _hard_kill(self, run_id: str, why: str) -> dict[str, Any]:
        pid = self._pid(run_id)
        killed = []
        if pid:
            try:
                parent = psutil.Process(pid)
                procs = parent.children(recursive=True) + [parent]
                for p in procs:
                    try:
                        p.kill()
                        killed.append(p.pid)
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(procs, timeout=10)
            except psutil.NoSuchProcess:
                pass
        self.kill_deadlines.pop(run_id, None)
        st = read_status(run_id)
        st.update({"state": "interrupted", "message": f"TERMINATED BY USER - interrupted ({why}); "
                   f"last safe checkpoint: {st.get('last_checkpoint') or 'none'}", "ended_at": time.time(),
                   "hard_killed_pids": killed})
        store.write_json(run_dir(run_id) / "status.json", st)
        self._settle(run_id, "interrupted")
        with open(run_dir(run_id) / "log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] HARD KILL ({why}): process tree {killed} terminated; "
                    f"the OS released its GPU and RAM. Existing checkpoints are untouched.\n")
        return st

    @staticmethod
    def _settle(run_id: str, state: str) -> None:
        from .. import registry
        try:
            registry.settle_status(read_config(run_id)["model_uid"], state)
        except Exception:
            pass

    # ------------------------------------------------------------- monitor
    def _pid(self, run_id: str) -> int | None:
        p = self.procs.get(run_id)
        if p is not None:
            return p.pid
        return read_status(run_id).get("pid")

    def _alive(self, run_id: str) -> bool:
        p = self.procs.get(run_id)
        if p is not None:
            return p.poll() is None
        pid = read_status(run_id).get("pid")
        if not pid:
            return False
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and "makeai.train.worker" in " ".join(proc.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def _require_active(self, run_id: str):
        st = read_status(run_id)
        if st.get("state") not in ("running", "paused", "starting") or not self._alive(run_id):
            raise RuntimeError(f"run is {st.get('state')}")

    def _monitor(self):
        while not self._stop.is_set():
            try:
                for run_id, deadline in list(self.kill_deadlines.items()):
                    if not self._alive(run_id):
                        self.kill_deadlines.pop(run_id, None)
                    elif time.time() > deadline:
                        self._hard_kill(run_id, "graceful stop exceeded grace period")
                with self._lock:
                    items = list(self.procs.items())
                for run_id, p in items:
                    code = p.poll()
                    if code is None:
                        continue
                    with self._lock:
                        self.procs.pop(run_id, None)
                    if self.telemetry is not None:
                        self.telemetry.proc_watch.pop(p.pid, None)
                    st = read_status(run_id)
                    if st.get("state") not in TERMINAL:
                        tail = ""
                        try:
                            tail = (run_dir(run_id) / "stdout.txt").read_text(encoding="utf-8", errors="replace")[-1500:]
                        except OSError:
                            pass
                        st.update({"state": "crashed", "message": f"worker exited with code {code}", "exit_code": code,
                                   "ended_at": time.time(), "stdout_tail": tail})
                        store.write_json(run_dir(run_id) / "status.json", st)
                    if st.get("state") in ("crashed", "interrupted", "failed"):
                        self._settle(run_id, st["state"])
            except Exception:
                pass
            self._stop.wait(0.5)

    def recover(self):
        """Crash recovery at startup: runs marked active whose process is gone become 'crashed'."""
        for d in store.runs_dir().iterdir():
            st = store.read_json(d / "status.json", {}) or {}
            if st.get("state") in ACTIVE:
                pid = st.get("pid")
                alive = False
                if pid:
                    try:
                        proc = psutil.Process(pid)
                        alive = "makeai.train.worker" in " ".join(proc.cmdline())
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        alive = False
                if not alive:
                    st.update({"state": "crashed", "message": "MakeAI was closed or the worker died while this run "
                               "was active - resume from the latest checkpoint", "ended_at": st.get("heartbeat")})
                    store.write_json(d / "status.json", st)
                    self._settle(d.name, "crashed")

    def shutdown(self):
        self._stop.set()
