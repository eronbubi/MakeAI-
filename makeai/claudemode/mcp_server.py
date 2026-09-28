"""MakeAI MCP server (stdio) - lets a coding agent (Claude Code, Codex, Cursor, ...) work inside MakeAI.

Register once, e.g. for Claude Code:
    claude mcp add --scope user makeai -e MAKEAI_URL=http://127.0.0.1:7860 -- <python> <path>/mcp_server.py
(MakeAI writes the matching config for the other agents, see ``agents.py``.)

It speaks JSON-RPC 2.0 over stdin/stdout (Model Context Protocol) and only uses
the standard library, so it starts instantly. Every tool calls the local MakeAI
HTTP API with ``X-MakeAI-Agent: <agent>`` and records what it did in the
agent activity feed, which the user watches in the MakeAI window. If MakeAI is
not running, the first tool call starts it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

BASE = os.environ.get("MAKEAI_URL", "http://127.0.0.1:7860").rstrip("/")
ROOT = Path(__file__).resolve().parents[2]          # folder containing run.py
SERVER_NAME = "makeai"
VERSION = "1.1.0"
SUPPORTED_PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

# which agent runs this server: MAKEAI_AGENT (set by the config MakeAI writes) or the MCP clientInfo name
AGENTS = {"claude": "Claude", "codex": "Codex", "cursor": "Cursor", "opencode": "OpenCode", "antigravity": "Antigravity",
          "devin": "Devin", "windsurf": "Devin", "cline": "Cline", "aider": "Aider", "zed": "Zed", "kiro": "Kiro",
          "junie": "Junie", "jetbrains": "Junie"}
AGENT = {"id": "claude", "name": "Claude", "client": None}


def identify(env_id: str | None, client: dict | None) -> dict:
    cname = str((client or {}).get("name") or "")
    for key in ([env_id] if env_id else []) + [cname.lower()]:
        for k, name in AGENTS.items():
            if key and k in key.lower():
                aid = {"windsurf": "devin", "jetbrains": "junie"}.get(k, k)
                return {"id": aid, "name": name, "client": cname or None}
    return {"id": (env_id or cname or "agent").lower()[:32], "name": cname[:40] or "Agent", "client": cname or None}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ====================================================================== HTTP
class ApiError(Exception):
    pass


def http(method: str, path: str, body=None, timeout: float = 120.0):
    data = None
    headers = {"X-MakeAI-Client": "1", "X-MakeAI-Agent": AGENT["id"]}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return json.loads(raw) if raw and r.headers.get("content-type", "").startswith("application/json") else raw
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8")).get("detail")
        except Exception:
            detail = e.reason
        raise ApiError(f"{e.code}: {detail}")


def running() -> bool:
    try:
        http("GET", "/api/health", timeout=3)
        return True
    except Exception:
        return False


def ensure_running() -> str:
    """Start MakeAI in the background if it is not reachable."""
    if running():
        return "already running"
    run_py = ROOT / "run.py"
    port = urllib.parse.urlparse(BASE).port or 7860
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0x8) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen([sys.executable, str(run_py), "--no-browser", "--port", str(port)], cwd=str(ROOT),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=flags, start_new_session=(os.name != "nt"))
    for _ in range(120):
        time.sleep(0.5)
        if running():
            return "started"
    raise ApiError(f"MakeAI did not start at {BASE}")


def event(kind: str, text: str, **kw):
    try:
        http("POST", "/api/claude/event", {"kind": kind, "text": text, **kw}, timeout=10)
    except Exception as e:
        log("event failed", e)


def hello():
    """Tell MakeAI (if it is running) which agent just loaded these tools - shown as 'connected' in the app."""
    def go():
        try:
            http("POST", "/api/claude/hello", {"agent": AGENT}, timeout=2)
        except Exception:
            pass
    import threading
    threading.Thread(target=go, daemon=True).start()


def focus(view: str, ref: str | None = None, title: str | None = None):
    try:
        http("POST", "/api/claude/focus", {"view": view, "id": ref, "title": title}, timeout=10)
    except Exception:
        pass


def wait_job(job: dict, timeout: float = 3600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = http("GET", f"/api/jobs/{job['id']}")
        if j["state"] == "done":
            return j["result"]
        if j["state"] in ("failed", "cancelled"):
            raise ApiError(f"{j['title']} failed: {j.get('error')}")
        time.sleep(1.0)
    raise ApiError(f"{job['title']} still running after {int(timeout)} s (job {job['id']})")


def profile() -> dict:
    return http("GET", "/api/settings")["profile"]


def fmt_params(n):
    if not n:
        return "?"
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


# ====================================================================== tools
TOOLS: dict[str, dict] = {}


def tool(name: str, description: str, props: dict | None = None, required: list[str] | None = None, log_as=None):
    def deco(fn):
        TOOLS[name] = {"fn": fn, "log": log_as, "schema": {
            "name": name, "description": description,
            "inputSchema": {"type": "object", "properties": props or {}, "required": required or []}}}
        return fn
    return deco


S = {"type": "string"}
I = {"type": "integer"}
N = {"type": "number"}
B = {"type": "boolean"}
O = {"type": "object"}
SA = {"type": "array", "items": {"type": "string"}}


@tool("makeai_session_start", "Start a session. Starts MakeAI if needed, opens its live view for the user, "
      "and returns an overview of hardware, datasets, AIs and runs. Call this first.",
      {"goal": {**S, "description": "What the user wants you to achieve"}, "open_window": {**B, "description": "open the MakeAI window (default true)"}})
def t_session_start(goal: str = "", open_window: bool = True):
    how = ensure_running()
    sess = http("POST", "/api/claude/session", {"action": "start", "goal": goal, "agent": AGENT})
    if open_window and not sess.get("ui_open"):      # the user already has MakeAI open -> don't open another tab
        try:
            webbrowser.open(BASE.replace("127.0.0.1", "localhost") + "/?claude=1")
        except Exception:
            pass
    focus("dashboard", None, "Dashboard")
    ov = t_overview()
    hw = ov["hardware"]
    event("action", f"Scanned hardware — {hw.get('gpu') or hw.get('cpu')}"
          + (f", {round((hw.get('vram_total_mb') or 0) / 1024)} GB VRAM" if hw.get("vram_total_mb") else ""),
          tool="makeai_overview", status="ok", icon="chip")
    return {"makeai": how, "url": BASE, "overview": ov, "profile": profile(),
            "note": "The user watches your actions live in MakeAI (view only) and talks to you here."}


@tool("makeai_session_end", "End the session with a short summary for the user.", {"summary": S})
def t_session_end(summary: str = ""):
    return http("POST", "/api/claude/session", {"action": "end", "summary": summary})


@tool("makeai_say", "Add a short line to the activity list the user watches in MakeAI (e.g. 'Lowering the learning rate'). Keep it under 60 characters.",
      {"text": S}, ["text"])
def t_say(text: str):
    http("POST", "/api/claude/event", {"kind": "say", "text": text, "icon": "spark"})
    return {"ok": True}


@tool("makeai_wait", "Wait while a training run is going: returns as soon as one of your runs saves a checkpoint, "
      "finishes, fails or is stopped, or after timeout_s. Use it instead of polling.",
      {"timeout_s": {**N, "description": "seconds to wait, max 110 (default 50)"}})
def t_wait(timeout_s: float = 50):
    r = http("GET", f"/api/claude/wait?timeout={min(110, max(1, timeout_s))}", timeout=130)
    if not r["run_events"]:
        r["note"] = "nothing new yet - call makeai_wait again or check makeai_run_status"
    return r


@tool("makeai_focus", "Show a page to the user in the MakeAI live view.",
      {"view": {**S, "enum": ["dashboard", "hardware", "models", "ai", "run", "dataset", "datasets", "training", "evaluate"]},
       "id": S, "title": S}, ["view"])
def t_focus(view: str, id: str | None = None, title: str | None = None):
    focus(view, id, title)
    return {"ok": True}


@tool("makeai_overview", "Hardware summary, datasets, AIs, tokenizers and training runs.", log_as="Looked at the overview")
def t_overview():
    hw = http("GET", "/api/hardware")
    g = (hw.get("gpus") or [{}])[0]
    return {
        "hardware": {"gpu": g.get("name"), "vram_total_mb": g.get("vram_total_mb"), "vram_used_mb": g.get("vram_used_mb"),
                     "compute_capability": g.get("compute_capability"), "bf16": g.get("bf16"), "cpu": hw["cpu"]["name"],
                     "ram_total_mb": hw["ram"]["total_mb"], "ram_used_mb": hw["ram"]["used_mb"],
                     "benchmark_tflops": (hw.get("benchmark") or {}).get("tflops")},
        "datasets": [{"id": d["id"], "name": d["name"], "samples": d.get("stats", {}).get("samples"),
                      "tokens": d.get("stats", {}).get("tokens"), "kinds": d.get("stats", {}).get("kinds")}
                     for d in http("GET", "/api/datasets")],
        "ais": [{"uid": m["uid"], "id": m["id"], "name": m["name"], "status": m.get("status"), "method": m.get("method"),
                 "params": m.get("param_count"), "runnable": m.get("runnable"), "backend": m.get("backend") or "native"}
                for m in http("GET", "/api/models")],
        "tokenizers": [{"id": t["id"], "name": t["name"], "kind": t["kind"], "vocab_size": t["vocab_size"]}
                       for t in http("GET", "/api/tokenizers")],
        "runs": http("GET", "/api/runs")[:15],
    }


@tool("makeai_hardware", "Full hardware scan plus current live telemetry (GPU util, VRAM, temperature, power, CPU, RAM).",
      log_as="Scanned hardware")
def t_hardware():
    focus("hardware", None, "Hardware")
    return {"scan": http("GET", "/api/hardware"), "live": http("GET", "/api/hardware/live?seconds=5")["latest"]}


@tool("makeai_benchmark_gpu", "Measure the GPU's real matmul TFLOPS and bandwidth (improves time estimates).",
      log_as="Measured GPU throughput")
def t_bench():
    return wait_job(http("POST", "/api/hardware/benchmark"), 300)


@tool("makeai_get_ai", "Details of one AI (manifest, architecture, evaluations).", {"uid": S}, ["uid"])
def t_get_ai(uid: str):
    focus("ai", uid, uid)
    return http("GET", f"/api/models/{uid}")


@tool("makeai_import_dataset", "Import files or folders (JSON, JSONL, TXT, CSV, Parquet, folders) from this computer as a dataset.",
      {"name": S, "paths": SA, "options": {**O, "description": "txt_mode (file|paragraph|line), extensions ['.py'], text_field"}},
      ["name", "paths"])
def t_import_dataset(name: str, paths: list, options: dict | None = None):
    event("action", f"Importing dataset “{name}” from {len(paths)} path(s)", tool="makeai_import_dataset", status="running")
    m = wait_job(http("POST", "/api/datasets", {"name": name, "paths": paths, "options": options or {}}))
    focus("dataset", m["id"], m["name"])
    return m


@tool("makeai_dataset_preview", "First records of a dataset and its statistics.", {"id": S, "offset": I, "limit": I}, ["id"])
def t_ds_preview(id: str, offset: int = 0, limit: int = 5):
    focus("dataset", id, id)
    meta = http("GET", f"/api/datasets/{id}")
    rows = http("GET", f"/api/datasets/{id}/preview?offset={offset}&limit={min(limit, 20)}")
    for r in rows:
        if "text" in r:
            r["text"] = r["text"][:1500]
    return {"meta": meta, "records": rows}


@tool("makeai_dataset_tool", "Run a dataset tool: clean, filter (min_chars, max_chars, min_words, max_words, include_regex, "
      "exclude_regex), dedup (near: bool), shuffle (seed), stats (tokenizer id or 'model:<uid>').",
      {"id": S, "op": {**S, "enum": ["clean", "filter", "dedup", "shuffle", "stats"]}, "options": O}, ["id", "op"])
def t_ds_tool(id: str, op: str, options: dict | None = None):
    focus("dataset", id, id)
    return wait_job(http("POST", f"/api/datasets/{id}/{op}", options or {}))


@tool("makeai_train_tokenizer", "Train a tokenizer (bpe = byte-level BPE, wordpiece, unigram, sentencepiece) on datasets.",
      {"name": S, "kind": {**S, "enum": ["bpe", "wordpiece", "unigram", "sentencepiece"]}, "vocab_size": I, "dataset_ids": SA},
      ["name", "dataset_ids"])
def t_train_tok(name: str, dataset_ids: list, kind: str = "bpe", vocab_size: int = 8192):
    event("action", f"Training {kind} tokenizer “{name}” (vocab {vocab_size})", tool="makeai_train_tokenizer", status="running")
    return wait_job(http("POST", "/api/tokenizers/train", {"name": name, "kind": kind, "vocab_size": vocab_size,
                                                          "dataset_ids": dataset_ids}))


@tool("makeai_recommend", "Hardware-aware recommendation (architecture, training settings, VRAM/time estimates) for a "
      "complexity level 1/4/16/64/256 and method from_scratch/full/lora/qlora/adapter.",
      {"complexity": I, "method": S, "base_model": S, "dataset_ids": SA, "vocab_size": I})
def t_recommend(complexity: int = 4, method: str = "from_scratch", base_model: str | None = None,
                dataset_ids: list | None = None, vocab_size: int | None = None):
    return http("POST", "/api/recommend", {"complexity": complexity, "method": method, "base_model": base_model,
                                           "dataset_ids": dataset_ids or [], "vocab_size": vocab_size})


@tool("makeai_create_ai", "Create a new AI in My AIs (creator = the user's profile). method from_scratch needs dataset_ids "
      "(a byte-level BPE tokenizer is trained on them unless tokenizer is given); other methods need base_model. "
      "config optionally overrides the recommended architecture.",
      {"name": S, "description": S, "version": S, "tags": SA, "icon": S,
       "method": {**S, "enum": ["from_scratch", "full", "lora", "qlora", "adapter"]}, "base_model": S,
       "complexity": I, "dataset_ids": SA, "tokenizer": S, "vocab_size": I, "config": O}, ["name", "method"])
def t_create_ai(name: str, method: str, description: str = "", version: str = "1.0", tags: list | None = None,
                icon: str = "", base_model: str | None = None, complexity: int = 4, dataset_ids: list | None = None,
                tokenizer: str | None = None, vocab_size: int | None = None, config: dict | None = None):
    me = profile()
    body = {"name": name, "creator_name": me["name"], "creator_username": me["username"], "description": description,
            "version": version, "tags": tags or [], "icon": icon, "method": method}
    if method == "from_scratch":
        if not dataset_ids and not tokenizer:
            raise ApiError("from_scratch needs dataset_ids (to train the tokenizer) or an existing tokenizer id")
        rec = t_recommend(complexity, "from_scratch", None, dataset_ids, vocab_size)
        cfg = {**rec["model"], **(config or {})}
        if not tokenizer:
            event("action", f"Training BPE tokenizer for {name} (vocab {cfg['vocab_size']})", tool="makeai_create_ai", status="running")
            tok = wait_job(http("POST", "/api/tokenizers/train", {"name": f"{name} bpe", "kind": "bpe",
                                                                  "vocab_size": cfg["vocab_size"], "dataset_ids": dataset_ids}))
            tokenizer = tok["id"]
        body.update({"config": cfg, "tokenizer": tokenizer})
    else:
        if not base_model:
            raise ApiError(f"{method} needs base_model (uid of an AI in My AIs)")
        body["base_model"] = base_model
    m = http("POST", "/api/models", body)
    focus("ai", m["uid"], m["name"])
    return {"uid": m["uid"], "id": m["id"], "params": m.get("param_count"), "status": m["status"]}


@tool("makeai_start_training", "Start training an AI. Settings come from the hardware recommendation for the complexity level; "
      "'training' overrides any field (e.g. learning_rate, micro_batch_size, gradient_accumulation, context_length, "
      "precision, optimizer, scheduler, eval_every, lora {rank, alpha}); 'steps' sets max_steps. Returns the run id; "
      "the user sees the live dashboard.",
      {"model_uid": S, "dataset_ids": SA, "complexity": I, "steps": I, "training": O, "keep_checkpoints": I},
      ["model_uid", "dataset_ids"])
def t_start_training(model_uid: str, dataset_ids: list, complexity: int = 4, steps: int | None = None,
                     training: dict | None = None, keep_checkpoints: int = 3):
    m = http("GET", f"/api/models/{model_uid}")
    method = m["method"]
    rec_method = "full" if (method in ("from_scratch", "imported") and m.get("runnable")) else method
    if rec_method == "imported":
        rec_method = "full"
    rec = t_recommend(complexity, rec_method, None if rec_method == "from_scratch" else (m.get("base_model") or model_uid),
                      dataset_ids, (m.get("config") or {}).get("vocab_size") if rec_method == "from_scratch" else None)
    tr = {**rec["training"], **(training or {})}
    if steps:
        tr["max_steps"] = int(steps)
    event("action", f"Picked settings: {fmt_params(rec['estimates'].get('params'))}, {str(tr.get('precision')).upper()}, "
          f"batch {tr.get('micro_batch_size')}" + (f"×{tr.get('gradient_accumulation')}" if tr.get("gradient_accumulation", 1) > 1 else ""),
          tool="makeai_start_training", status="ok", icon="sliders")
    r = http("POST", "/api/runs", {"model_uid": model_uid, "datasets": [{"id": d, "weight": 1} for d in dataset_ids],
                                   "training": tr, "checkpoint": {"save_every": tr.get("save_every"), "keep": keep_checkpoints},
                                   "eval": {"max_batches": 50}})
    http("POST", "/api/claude/track", {"run_id": r["run_id"]})
    focus("run", r["run_id"], m["name"])
    event("action", f"Started training {m['name']}", tool="makeai_start_training", status="ok", icon="play",
          detail=f"{tr.get('max_steps')} steps · lr {tr.get('learning_rate')}")
    return {"run_id": r["run_id"], "settings": {k: tr.get(k) for k in ("precision", "optimizer", "learning_rate", "scheduler",
            "micro_batch_size", "gradient_accumulation", "context_length", "max_steps", "epochs", "gradient_checkpointing")},
            "estimates": rec["estimates"], "notes": rec.get("notes")}


@tool("makeai_run_status", "Live status of a training run: state, step, loss, validation, throughput, ETA, memory, "
      "performance score and the last log lines.", {"run_id": S, "log_lines": I}, ["run_id"])
def t_run_status(run_id: str, log_lines: int = 15):
    info = http("GET", f"/api/runs/{run_id}")
    recs = http("GET", f"/api/runs/{run_id}/metrics?since=0")["records"]
    steps = [r for r in recs if r["type"] == "step"]
    evals = [r for r in recs if r["type"] == "eval"]
    last = steps[-1] if steps else {}
    keep = ("step", "total_steps", "epoch", "loss", "lr", "grad_norm", "tokens_per_s", "steps_per_s", "eta_s", "elapsed",
            "mem_peak_mb", "data_wait_frac")
    first10 = sum(s["loss"] for s in steps[:10]) / len(steps[:10]) if steps else None
    health = http("GET", f"/api/runs/{run_id}/health")
    return {"state": info["status"].get("state"), "message": info["status"].get("message"),
            "last_step": {k: last.get(k) for k in keep}, "loss_first_10_avg": first10,
            "evals": [{k: e.get(k) for k in ("step", "val_loss", "val_ppl", "token_accuracy")} for e in evals[-5:]],
            "best_val_loss": info["status"].get("best_val_loss"),
            "health": {k: health.get(k) for k in ("score", "status_label", "reason", "recommendation", "components")},
            "log": http("GET", f"/api/runs/{run_id}/log?lines={log_lines}")["lines"]}


@tool("makeai_run_control", "Control a training run: pause, resume, checkpoint, kill (graceful: saves a checkpoint), "
      "restart (resume from the latest checkpoint).",
      {"run_id": S, "action": {**S, "enum": ["pause", "resume", "checkpoint", "kill", "restart"]}}, ["run_id", "action"])
def t_run_control(run_id: str, action: str):
    focus("run", run_id, "Run")
    return http("POST", f"/api/runs/{run_id}/{action}", {})


@tool("makeai_evaluate", "Evaluate an AI on a dataset: loss, perplexity, token accuracy.",
      {"uid": S, "dataset_id": S, "max_batches": I}, ["uid", "dataset_id"])
def t_eval(uid: str, dataset_id: str, max_batches: int = 50):
    focus("ai", uid, uid)
    return wait_job(http("POST", "/api/eval/dataset", {"uid": uid, "dataset_id": dataset_id, "max_batches": max_batches}))


@tool("makeai_benchmark_ai", "Run a custom benchmark: items [{prompt, expected, match: contains|exact|regex}] (greedy decoding).",
      {"uid": S, "items": {"type": "array", "items": O}, "name": S}, ["uid", "items"])
def t_bench_ai(uid: str, items: list, name: str = "benchmark"):
    r = wait_job(http("POST", "/api/eval/benchmark", {"uid": uid, "items": items, "name": name}))
    return {"accuracy": r.get("accuracy"), "n": r.get("n"), "results": r.get("results")}


@tool("makeai_chat", "Talk to an AI locally (chat, or raw completion with 'prompt'). Returns the reply and speed.",
      {"uid": S, "message": S, "prompt": S, "system_prompt": S, "temperature": N, "max_tokens": I}, ["uid"])
def t_chat(uid: str, message: str | None = None, prompt: str | None = None, system_prompt: str | None = None,
           temperature: float = 0.7, max_tokens: int = 200):
    body = {"uid": uid, "messages": [{"role": "user", "content": message}] if message else [], "raw_prompt": prompt,
            "system_prompt": system_prompt, "params": {"temperature": temperature, "max_tokens": max_tokens, "seed": 0}}
    raw = http("POST", "/api/playground/chat", body, timeout=600)
    text, done = "", {}
    for line in str(raw).splitlines():
        if line.startswith("data: "):
            d = json.loads(line[6:])
            if d.get("error"):
                raise ApiError(d["error"])
            text += d.get("delta", "")
            if d.get("done"):
                done = d
    return {"reply": text, "tokens": done.get("completion_tokens"), "tokens_per_s": done.get("tokens_per_s"),
            "finish_reason": done.get("finish_reason")}


@tool("makeai_export", "Export an AI: package, safetensors, hf, gguf, pytorch, onnx, lora, tokenizer, config.",
      {"uid": S, "format": S}, ["uid", "format"])
def t_export(uid: str, format: str):
    return wait_job(http("POST", f"/api/models/{uid}/export", {"format": format}))


@tool("makeai_import_model", "Inspect and import a model file/folder (.makeai, .safetensors+config, .gguf, .pt, .onnx). "
      "For files without creator metadata give creator_name/creator_username of the original author.",
      {"path": S, "creator_name": S, "creator_username": S, "name": S, "version": S, "inspect_only": B}, ["path"])
def t_import(path: str, creator_name: str | None = None, creator_username: str | None = None, name: str | None = None,
             version: str = "1.0", inspect_only: bool = False):
    info = http("POST", "/api/import/inspect", {"path": path})
    if inspect_only:
        return info
    o = {k: v for k, v in {"creator_name": creator_name, "creator_username": creator_username, "name": name,
                           "version": version}.items() if v}
    m = wait_job(http("POST", "/api/import", {"path": path, "overrides": o}))
    focus("ai", m["uid"], m["name"])
    return {"uid": m["uid"], "id": m["id"], "runnable": m.get("runnable")}


@tool("makeai_list_runs", "All training runs with state and latest loss.")
def t_list_runs():
    return http("GET", "/api/runs")


LOG_TEXT = {
    "makeai_get_ai": lambda a: f"Opened AI {a.get('uid')}",
    "makeai_dataset_preview": lambda a: f"Looked at dataset {a.get('id')}",
    "makeai_dataset_tool": lambda a: f"Dataset {a.get('id')}: {a.get('op')} {json.dumps(a.get('options') or {})}",
    "makeai_recommend": lambda a: f"Asked for a recommendation (complexity {a.get('complexity', 4)}, {a.get('method', 'from_scratch')})",
    "makeai_create_ai": lambda a: f"Created AI “{a.get('name')}” ({a.get('method')})",
    "makeai_run_status": lambda a: "Watching loss, VRAM and temperature",
    "makeai_run_control": lambda a: f"{str(a.get('action', '')).upper()} run {a.get('run_id')}",
    "makeai_evaluate": lambda a: f"Evaluated {a.get('uid')} on {a.get('dataset_id')}",
    "makeai_benchmark_ai": lambda a: f"Benchmarked {a.get('uid')} ({len(a.get('items') or [])} items)",
    "makeai_chat": lambda a: f"Chatted with {a.get('uid')}: “{(a.get('message') or a.get('prompt') or '')[:80]}”",
    "makeai_export": lambda a: f"Exported {a.get('uid')} as {a.get('format')}",
    "makeai_import_model": lambda a: f"{'Inspected' if a.get('inspect_only') else 'Imported'} {a.get('path')}",
    "makeai_import_dataset": lambda a: f"Imported dataset “{a.get('name')}”",
    "makeai_train_tokenizer": lambda a: f"Trained tokenizer “{a.get('name')}”",
    "makeai_benchmark_gpu": lambda a: "Measured GPU throughput",
}
QUIET = {"makeai_wait", "makeai_say", "makeai_session_start", "makeai_session_end", "makeai_focus", "makeai_list_runs",
         "makeai_overview", "makeai_start_training"}
ICONS = {"makeai_hardware": "chip", "makeai_benchmark_gpu": "chip", "makeai_recommend": "sliders",
         "makeai_start_training": "play", "makeai_run_status": "eye", "makeai_run_control": "control",
         "makeai_create_ai": "plus", "makeai_import_dataset": "data", "makeai_dataset_tool": "data",
         "makeai_dataset_preview": "data", "makeai_train_tokenizer": "data", "makeai_evaluate": "chart",
         "makeai_benchmark_ai": "chart", "makeai_chat": "chat", "makeai_export": "box", "makeai_import_model": "box",
         "makeai_get_ai": "eye"}


def summarize(name: str, result) -> str:
    if not isinstance(result, dict):
        return ""
    if name == "makeai_run_status":
        ls = result.get("last_step") or {}
        return (f"{result.get('state')} · step {ls.get('step')}/{ls.get('total_steps')} · loss "
                f"{ls.get('loss'):.3f}" if ls.get("loss") is not None else str(result.get("state")))
    if name == "makeai_chat":
        return f"reply: “{(result.get('reply') or '')[:160]}” ({result.get('tokens_per_s')} tok/s)"
    if name == "makeai_evaluate":
        return f"loss {result.get('loss', 0):.4f} · ppl {result.get('perplexity', 0):.2f} · token acc {100 * result.get('token_accuracy', 0):.1f}%"
    if name == "makeai_create_ai":
        return f"{result.get('id')} · {fmt_params(result.get('params'))} parameters"
    if name == "makeai_start_training":
        s = result.get("settings", {})
        return f"run {result.get('run_id')} · {s.get('max_steps')} steps · lr {s.get('learning_rate')} · {s.get('precision')}"
    if name == "makeai_export":
        return f"{result.get('file')} ({round((result.get('bytes') or 0) / 2**20, 1)} MB)"
    return ""


def call_tool(name: str, args: dict):
    t = TOOLS.get(name)
    if not t:
        raise ApiError(f"unknown tool {name}")
    if name != "makeai_session_start" and not running():
        ensure_running()
    label = t["log"] or (LOG_TEXT[name](args) if name in LOG_TEXT else None)
    try:
        result = t["fn"](**args)
    except Exception as e:
        if name not in QUIET and label:
            event("action", label, tool=name, status="error", icon=ICONS.get(name), detail=str(e)[:400])
        raise
    if name not in QUIET and label:
        event("action", label, tool=name, status="ok", icon=ICONS.get(name), detail=summarize(name, result))
    return result


# ====================================================================== JSON-RPC
def send(msg: dict):
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def handle(req: dict):
    method = req.get("method")
    rid = req.get("id")
    params = req.get("params") or {}
    if rid is None:            # notification (e.g. notifications/initialized)
        return
    try:
        if method == "initialize":
            AGENT.update(identify(os.environ.get("MAKEAI_AGENT"), params.get("clientInfo")))
            hello()
            v = params.get("protocolVersion")
            result = {"protocolVersion": v if v in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0],
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": SERVER_NAME, "version": VERSION},
                      "instructions": "Tools to work inside MakeAI (local AI training app). The user watches your "
                                      "actions live in MakeAI (view only) and talks to you in this chat. "
                                      "Start with makeai_session_start; use makeai_wait while training runs."}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": [t["schema"] for t in TOOLS.values()]}
        elif method == "tools/call":
            name = params.get("name")
            try:
                out = call_tool(name, params.get("arguments") or {})
                result = {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, default=str)[:60000]}],
                          "isError": False}
            except (ApiError, TypeError, KeyError, urllib.error.URLError) as e:
                result = {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
        elif method in ("resources/list", "prompts/list"):
            result = {"resources": []} if method == "resources/list" else {"prompts": []}
        else:
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}})
            return
        send({"jsonrpc": "2.0", "id": rid, "result": result})
    except Exception as e:
        log(traceback.format_exc())
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(e)}})


def parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def cli(argv: list[str]) -> int:
    """Command-line use for agents without MCP (Aider):  mcp_server.py [--agent aider] call <tool> [k=v ...|JSON]"""
    global BASE
    agent = os.environ.get("MAKEAI_AGENT")
    while argv and argv[0].startswith("--"):
        flag, val = argv[0], argv[1] if len(argv) > 1 else ""
        if flag == "--agent":
            agent = val
        elif flag == "--url":
            BASE = val.rstrip("/")
        argv = argv[2:]
    AGENT.update(identify(agent or "cli", {"name": "command line"}))
    if not argv or argv[0] not in ("call", "tools"):
        print("usage: mcp_server.py [--agent NAME] [--url URL] call <tool> [key=value ...] | tools", file=sys.stderr)
        return 2
    if argv[0] == "tools":
        for name, t in TOOLS.items():
            props = t["schema"]["inputSchema"].get("properties") or {}
            print(f"{name}({', '.join(props)}): {t['schema']['description']}")
        return 0
    if len(argv) < 2:
        print("call needs a tool name", file=sys.stderr)
        return 2
    name, rest = argv[1], argv[2:]
    args: dict = {}
    for a in rest:
        if a.lstrip().startswith("{"):
            args.update(json.loads(a))
        elif "=" in a:
            k, v = a.split("=", 1)
            args[k] = parse_value(v)
        else:
            print(f"cannot read argument {a!r} (use key=value)", file=sys.stderr)
            return 2
    try:
        out = call_tool(name, args)
    except (ApiError, TypeError, KeyError, urllib.error.URLError) as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:60000])
    return 0


def main():
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if args and (args[0] in ("call", "tools") or args[0].startswith("--")):
        sys.exit(cli(args))
    log(f"makeai MCP server {VERSION} -> {BASE}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        if isinstance(req, list):
            for r in req:
                handle(r)
        else:
            handle(req)


if __name__ == "__main__":
    main()
