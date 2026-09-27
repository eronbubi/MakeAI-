"""Claude Mode: the real MCP protocol over stdio against a live MakeAI server."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from conftest import ROOT, wait_state

MCP = ROOT / "makeai" / "claudemode" / "mcp_server.py"


@pytest.fixture(scope="module")
def live(trained):
    import uvicorn

    from makeai.server.app import create_app
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    base = f"http://127.0.0.1:{port}"
    req = urllib.request.Request(base + "/api/settings", method="PATCH", data=json.dumps({"profile": {"name": "Eron", "username": "eron"}}).encode(),
                                 headers={"Content-Type": "application/json", "X-MakeAI-Client": "1"})
    urllib.request.urlopen(req).read()
    yield base
    server.should_exit = True


class Mcp:
    def __init__(self, base):
        self.p = subprocess.Popen([sys.executable, str(MCP)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, encoding="utf-8", env={**os.environ, "MAKEAI_URL": base})
        self.n = 0

    def rpc(self, method, params=None):
        self.n += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        assert r["id"] == self.n
        return r

    def call(self, name, **args):
        r = self.rpc("tools/call", {"name": name, "arguments": args})["result"]
        assert not r["isError"], r["content"][0]["text"]
        return json.loads(r["content"][0]["text"])

    def close(self):
        self.p.stdin.close()
        self.p.wait(10)


def post(base, path, body):
    req = urllib.request.Request(base + path, method="POST", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-MakeAI-Client": "1"})
    return json.loads(urllib.request.urlopen(req).read())


def get(base, path):
    return json.loads(urllib.request.urlopen(base + path).read())


def test_mcp_handshake_and_tools(live):
    m = Mcp(live)
    init = m.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}})
    assert init["result"]["protocolVersion"] == "2025-06-18" and init["result"]["capabilities"]["tools"] is not None
    tools = {t["name"] for t in m.rpc("tools/list")["result"]["tools"]}
    assert {"makeai_session_start", "makeai_wait", "makeai_say", "makeai_create_ai", "makeai_start_training",
            "makeai_run_status", "makeai_run_control", "makeai_chat"} <= tools
    assert not any("delete" in t for t in tools)          # Claude gets no destructive tools
    assert m.rpc("bogus/method").get("error")
    m.close()


def test_claude_works_while_user_watches(live, trained):
    m = Mcp(live)
    m.rpc("initialize", {"protocolVersion": "2024-11-05"})
    m.call("makeai_session_start", goal="train a tiny model", open_window=False)
    run = m.call("makeai_start_training", model_uid=trained["uid"], dataset_ids=[trained["ds"]], complexity=1, steps=30,
                 training={"micro_batch_size": 8, "gradient_accumulation": 1, "context_length": 128, "eval_every": 15})
    st = get(live, "/api/claude/state?since=0")
    assert st["connected"] and st["session"]["active"] and st["focus"] == {**st["focus"], "view": "run", "id": run["run_id"]}
    texts = [e["text"] for e in st["events"]]
    assert any(t.startswith("Scanned hardware") for t in texts)
    assert any(t.startswith("Picked settings") for t in texts) and any(t.startswith("Started training") for t in texts)
    # checkpoints and the final result reach Claude through makeai_wait and appear in the activity list
    events = []
    for _ in range(40):
        events += m.call("makeai_wait", timeout_s=15)["run_events"]
        if any(e["state"] == "completed" for e in events):
            break
    assert any(e["run_id"] == run["run_id"] and e["state"] == "completed" for e in events)
    texts = [e["text"] for e in get(live, "/api/claude/state?since=0")["events"]]
    assert any("checkpoint saved at step" in t.lower() for t in texts) and any(t.startswith("Training finished") for t in texts)
    # repeated status checks are listed once
    m.call("makeai_run_status", run_id=run["run_id"])
    m.call("makeai_run_status", run_id=run["run_id"])
    texts = [e["text"] for e in get(live, "/api/claude/state?since=0")["events"]]
    assert texts.count("Watching loss, VRAM and temperature") == 1
    status = m.call("makeai_run_status", run_id=run["run_id"])
    assert status["state"] == "completed" and status["last_step"]["step"] == 30
    reply = m.call("makeai_chat", uid=trained["uid"], prompt="def ", max_tokens=8)
    assert reply["tokens"] > 0
    m.call("makeai_session_end", summary="done")
    assert not get(live, "/api/claude/state?since=0")["session"]["active"]
    m.close()


def test_app_has_no_input_to_claude(live):
    """Claude Mode is watch-only: there is no endpoint to send messages from MakeAI to Claude."""
    for path in ("/api/claude/message", "/api/claude/launch"):
        try:
            post(live, path, {"text": "x"})
            raise AssertionError(f"{path} should not exist")
        except urllib.error.HTTPError as e:
            assert e.code in (404, 405)
