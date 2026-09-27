"""Optional Claude development agent - develops MakeAI itself.

This package is NOT part of the MakeAI runtime. Nothing in training, inference,
datasets, import/export, sharing or projects imports it. The server loads it
with a guarded import; deleting this folder (or not having the ``claude`` CLI)
simply hides the Development panel.

Workflow: plan (read-only, ``--permission-mode plan``) -> user approves or edits
-> execute (``--permission-mode acceptEdits``, resuming the planning session)
-> the agent writes code and runs tests in the MakeAI source tree.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

SOURCE_ROOT = Path(__file__).resolve().parents[2]

PLAN_PROMPT = """You are the development agent for MakeAI (by Convergent), a local AI model
creation/training/inference application whose source is in the current directory.
Analyze the project and produce a concise numbered implementation plan for this request.
Do not modify any files in this step.

Request: {task}

Answer with: a one-line summary, then the numbered plan (files to change, tests to add/run)."""

EXEC_PROMPT = """The plan below is APPROVED by the user (it may have been edited). Implement it now in
this repository, keep the MakeAI runtime independent of Claude, then run the relevant tests
(`python -m pytest -q` from the project root, using the project's virtual environment if present)
and report what changed and the test results.

Approved plan:
{plan}"""


def claude_path() -> str | None:
    return shutil.which("claude")


def status() -> dict[str, Any]:
    p = claude_path()
    return {"available": p is not None, "cli": p, "source_root": str(SOURCE_ROOT),
            "reason": None if p else "Claude Code CLI ('claude') not found on PATH"}


def _run(args: list[str], on_line: Callable[[str], None] | None, cancel, timeout: float) -> tuple[int, list[str]]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(args, cwd=str(SOURCE_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", creationflags=flags)
    lines: list[str] = []
    t0 = time.time()
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        if on_line:
            on_line(line)
        if (cancel is not None and cancel.is_set()) or time.time() - t0 > timeout:
            proc.kill()
            break
    return proc.wait(), lines


def plan(task: str, on_line=None, cancel=None, timeout: float = 900) -> dict[str, Any]:
    cli = claude_path()
    if not cli:
        raise RuntimeError(status()["reason"])
    code, lines = _run([cli, "-p", PLAN_PROMPT.format(task=task), "--permission-mode", "plan",
                        "--output-format", "json"], None, cancel, timeout)
    raw = "\n".join(lines)
    try:
        data = json.loads(raw[raw.index("{"):])
    except (ValueError, json.JSONDecodeError):
        raise RuntimeError(f"claude exited with {code}: {raw[-800:]}")
    if data.get("is_error"):
        raise RuntimeError(data.get("result") or "claude reported an error")
    return {"plan": data.get("result", ""), "session_id": data.get("session_id"),
            "cost_usd": data.get("total_cost_usd"), "duration_ms": data.get("duration_ms")}


def execute(plan_text: str, session_id: str | None, on_line=None, cancel=None, timeout: float = 3600) -> dict[str, Any]:
    cli = claude_path()
    if not cli:
        raise RuntimeError(status()["reason"])
    args = [cli, "-p", EXEC_PROMPT.format(plan=plan_text), "--permission-mode", "acceptEdits",
            "--output-format", "stream-json", "--verbose"]
    if session_id:
        args += ["--resume", session_id]
    events: list[dict] = []

    def handle(line: str):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            if on_line:
                on_line(line)
            return
        events.append(ev)
        if on_line:
            on_line(_describe(ev))

    code, _ = _run(args, handle, cancel, timeout)
    final = next((e for e in reversed(events) if e.get("type") == "result"), {})
    return {"exit_code": code, "result": final.get("result"), "cost_usd": final.get("total_cost_usd"),
            "session_id": final.get("session_id") or session_id, "events": len(events)}


def _describe(ev: dict) -> str:
    t = ev.get("type")
    if t == "assistant":
        parts = []
        for c in (ev.get("message") or {}).get("content", []):
            if c.get("type") == "text":
                parts.append(c["text"])
            elif c.get("type") == "tool_use":
                inp = c.get("input") or {}
                target = inp.get("file_path") or inp.get("command") or inp.get("pattern") or ""
                parts.append(f"→ {c.get('name')} {str(target)[:160]}")
        return "\n".join(parts)
    if t == "user":
        for c in (ev.get("message") or {}).get("content", []):
            if isinstance(c, dict) and c.get("type") == "tool_result":
                txt = c.get("content")
                if isinstance(txt, list):
                    txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
                return f"  ✓ {str(txt)[:200]}"
    if t == "result":
        return f"done: {ev.get('subtype')} ({ev.get('duration_ms', 0) / 1000:.0f}s)"
    return ""
