"""Claude Mode - Claude works in MakeAI, the user watches.

The user talks to Claude in the Claude Code app. Claude reaches MakeAI through
the MCP server in ``mcp_server.py`` (normal MakeAI API, requests marked
``X-MakeAI-Agent: claude``). This module keeps what the watch-only Claude Mode
view shows:

* the session (goal, connected / last seen),
* an activity list (what Claude did, plus checkpoints and training results),
* the "focus" - the thing Claude is working on (for example a training run).

Like ``makeai.devagent`` this package is optional: the runtime never needs it,
and deleting it only hides Claude Mode.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import Body, HTTPException, Request

from .. import store

MCP_SCRIPT = Path(__file__).resolve().parent / "mcp_server.py"


def console_python() -> str:
    """The MCP server talks over stdin/stdout, so it must run with python.exe, not pythonw.exe."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)
CONNECTED_WINDOW_S = 90          # Claude counts as connected if it called within this window

PROMPT_TEMPLATE = """Work inside MakeAI for me using the "makeai" tools. I watch what you do live in MakeAI's Claude Mode.

Goal: {goal}

1. Start with makeai_session_start (pass the goal).
2. Look at the hardware and data, choose sensible settings, create the AI and start training.
3. Keep an eye on the run: use makeai_wait to wait for checkpoints and results, and makeai_run_status to check
   loss, validation loss, speed, VRAM and temperature. React if something goes wrong.
4. When training is done, evaluate the AI and try it with makeai_chat, then tell me the results here.
5. Don't delete anything and don't make an AI public unless I ask. Finish with makeai_session_end."""


class ClaudeLink:
    def __init__(self, list_runs: Callable[[], list[dict]], read_status: Callable[[str], dict]):
        self.dir = store.sub("claudemode")
        self._list_runs = list_runs
        self._read_status = read_status
        self.cond = threading.Condition()
        self.session: dict[str, Any] = store.read_json(self.dir / "session.json", {}) or {}
        self.events: list[dict[str, Any]] = []
        self.focus: dict[str, Any] | None = self.session.get("focus")
        self.last_seen = 0.0
        self.ui_seen = 0.0
        self._next = 1
        self._run_states: dict[str, str] = {}
        self._seen_ckpts: dict[str, set[str]] = {}
        self._pending_run_events: list[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        p = self.dir / "activity.jsonl"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f.readlines()[-500:]:
                    try:
                        self.events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if self.events:
            self._next = max(e["id"] for e in self.events) + 1
        for rid in self.session.get("runs", []):
            self._remember_run(rid)

    def _remember_run(self, rid: str) -> None:
        try:
            self._run_states[rid] = self._read_status(rid).get("state", "")
        except Exception:
            self._run_states[rid] = ""
        self._seen_ckpts[rid] = {c["name"] for c in self._checkpoints(rid)}

    def _checkpoints(self, rid: str) -> list[dict]:
        return store.read_json(store.runs_dir() / rid / "checkpoints" / "index.json", []) or []

    def _save_session(self) -> None:
        self.session["focus"] = self.focus
        store.write_json(self.dir / "session.json", self.session)

    def add(self, kind: str, text: str, **detail) -> dict[str, Any]:
        with self.cond:
            e = {"id": self._next, "t": time.time(), "kind": kind, "text": text, **detail}
            self._next += 1
            self.events.append(e)
            del self.events[:-1000]
            with open(self.dir / "activity.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            self.cond.notify_all()
            return e

    # ------------------------------------------------------------------ session
    @property
    def connected(self) -> bool:
        return time.time() - self.last_seen < CONNECTED_WINDOW_S

    def seen(self) -> None:
        self.last_seen = time.time()

    def start(self, goal: str) -> dict[str, Any]:
        self.seen()
        self.session = {"active": True, "goal": goal, "started_at": time.time(), "runs": [], "focus": None}
        self.focus = None
        self._save_session()
        self.add("system", "Claude started working" + (f": {goal}" if goal else ""), icon="spark")
        return {**self.session, "ui_open": time.time() - self.ui_seen < 5}

    def end(self, summary: str) -> dict[str, Any]:
        self.seen()
        self.session["active"] = False
        self.session["ended_at"] = time.time()
        self._save_session()
        self.add("system", "Claude finished" + (f": {summary}" if summary else ""), icon="done")
        return self.session

    def set_focus(self, view: str, ref: str | None, title: str | None) -> None:
        self.focus = {"view": view, "id": ref, "title": title or view, "t": time.time()}
        self._save_session()

    def track_run(self, run_id: str) -> None:
        runs = self.session.setdefault("runs", [])
        if run_id not in runs:
            runs.append(run_id)
            self._save_session()
        self._remember_run(run_id)

    # ------------------------------------------------------------------ run events
    def poll_runs(self) -> None:
        """Turn checkpoints and final states of Claude's runs into activity items (and events for Claude)."""
        for rid in list(self.session.get("runs", [])):
            for c in self._checkpoints(rid):
                seen = self._seen_ckpts.setdefault(rid, set())
                if c["name"] not in seen:
                    seen.add(c["name"])
                    label = {"best": "Best checkpoint", "final": "Final checkpoint", "kill": "Safe checkpoint",
                             "pause": "Checkpoint (paused)", "manual": "Checkpoint"}.get(c["kind"], "Checkpoint")
                    extra = f" · val loss {c['val_loss']:.3f}" if c.get("val_loss") is not None else ""
                    self.add("event", f"{label} saved at step {c['step']:,}{extra}", icon="check", run_id=rid)
            try:
                st = self._read_status(rid)
            except Exception:
                continue
            state = st.get("state", "")
            if state != self._run_states.get(rid) and state in ("completed", "failed", "crashed", "terminated",
                                                                "interrupted", "paused"):
                self._run_states[rid] = state
                ev = {"run_id": rid, "state": state, "message": st.get("message"), "step": st.get("step"),
                      "best_val_loss": st.get("best_val_loss"), "final_eval": st.get("final_eval")}
                self._pending_run_events.append(ev)
                text = {"completed": "Training finished", "paused": "Training paused", "failed": "Training failed",
                        "crashed": "Training stopped unexpectedly", "terminated": "Training stopped",
                        "interrupted": "Training interrupted"}[state]
                if state == "completed" and st.get("best_val_loss") is not None:
                    text += f" · best val loss {st['best_val_loss']:.3f}"
                elif st.get("message") and state in ("failed", "crashed"):
                    text += f" · {st['message'][:120]}"
                self.add("event", text, icon="done" if state == "completed" else "stop", run_id=rid, state=state)
            elif state:
                self._run_states[rid] = state

    def wait(self, timeout: float) -> dict[str, Any]:
        """Long-poll for Claude: returns as soon as a checkpoint or final state of its runs happens."""
        self.seen()
        deadline = time.time() + max(1.0, min(timeout, 110.0))
        while True:
            with self.cond:
                self.poll_runs()
                if self._pending_run_events or time.time() >= deadline:
                    events, self._pending_run_events = self._pending_run_events, []
                    self.seen()
                    return {"run_events": events,
                            "active_runs": [r for r in self._list_runs()
                                            if r.get("state") in ("starting", "running", "paused", "stopping")]}
                self.cond.wait(timeout=min(1.0, max(0.05, deadline - time.time())))
                self.seen()

    def state(self, since: int = 0) -> dict[str, Any]:
        self.ui_seen = time.time()
        with self.cond:
            self.poll_runs()
        return {"session": self.session, "connected": self.connected, "last_seen": self.last_seen or None,
                "focus": self.focus, "events": [e for e in self.events if e["id"] > since][-400:]}


# ---------------------------------------------------------------------- setup
def claude_cli() -> str | None:
    return shutil.which("claude")


def setup_info(base_url: str) -> dict[str, Any]:
    cmd = ["claude", "mcp", "add", "--scope", "user", "makeai", "-e", f"MAKEAI_URL={base_url}", "--",
           console_python(), str(MCP_SCRIPT)]
    registered = None
    cli = claude_cli()
    if cli:
        try:
            p = subprocess.run([cli, "mcp", "get", "makeai"], capture_output=True, text=True, timeout=30,
                               encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            norm = lambda x: x.replace("\\", "/").lower()
            registered = p.returncode == 0 and norm(str(MCP_SCRIPT)) in norm(p.stdout)
        except Exception:
            registered = None
    return {"cli": cli, "registered": registered, "prompt_template": PROMPT_TEMPLATE,
            "add_command": " ".join(f'"{c}"' if " " in c else c for c in cmd)}


def register(base_url: str) -> dict[str, Any]:
    cli = claude_cli()
    if not cli:
        raise RuntimeError("Claude Code is not installed on this computer")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run([cli, "mcp", "remove", "--scope", "user", "makeai"], capture_output=True, text=True, timeout=60,
                   creationflags=flags)
    p = subprocess.run([cli, "mcp", "add", "--scope", "user", "makeai", "-e", f"MAKEAI_URL={base_url}", "--",
                        console_python(), str(MCP_SCRIPT)], capture_output=True, text=True, timeout=60,
                       encoding="utf-8", errors="replace", creationflags=flags)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip()[-500:])
    return {"ok": True}


# ---------------------------------------------------------------------- routes
def install(app, S, list_runs, read_status) -> ClaudeLink:
    link = ClaudeLink(list_runs, read_status)
    S.claude = link

    def base(request: Request) -> str:
        return str(request.base_url).rstrip("/").replace("://localhost", "://127.0.0.1")

    @app.get("/api/claude/state")
    def claude_state(since: int = 0):
        return link.state(since)

    @app.post("/api/claude/session")
    def claude_session(body: dict = Body(...)):
        if body.get("action") == "end":
            return link.end(body.get("summary", ""))
        return link.start(body.get("goal", ""))

    @app.post("/api/claude/event")
    def claude_event(body: dict = Body(...)):
        link.seen()
        kind = body.get("kind") if body.get("kind") in ("say", "action") else "say"
        start = link.session.get("started_at") or 0
        same = next((e for e in reversed(link.events[-12:]) if e["kind"] == kind and e["text"] == body.get("text")
                     and e["t"] >= start and kind == "action"), None)
        if same:                             # repeated check (e.g. watching a run) -> refresh, don't list again
            same["detail"] = body.get("detail") or same.get("detail")
            return same
        return link.add(kind, str(body.get("text", ""))[:2000], tool=body.get("tool"), status=body.get("status"),
                        icon=body.get("icon"), detail=body.get("detail"))

    @app.post("/api/claude/focus")
    def claude_focus(body: dict = Body(...)):
        link.seen()
        link.set_focus(body.get("view", "dashboard"), body.get("id"), body.get("title"))
        return {"ok": True}

    @app.post("/api/claude/track")
    def claude_track(body: dict = Body(...)):
        link.track_run(body["run_id"])
        return {"ok": True}

    @app.get("/api/claude/wait")
    def claude_wait(timeout: float = 45):
        return link.wait(timeout)

    @app.get("/api/claude/setup")
    def claude_setup(request: Request):
        return setup_info(base(request))

    @app.post("/api/claude/register")
    def claude_register(request: Request):
        try:
            return register(base(request))
        except Exception as e:
            raise HTTPException(400, str(e))

    return link
