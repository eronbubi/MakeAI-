"""Agent connections: config files for Codex, Cursor, OpenCode, Antigravity, Devin, Cline, Aider, Zed, Kiro, Junie."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomli

from makeai.claudemode import agents as A

BASE = "http://127.0.0.1:7860"

ZED_SETTINGS = """// Zed settings
//
// For information on how to configure Zed, see the Zed
// documentation: https://zed.dev/docs/configuring-zed
{
  "icon_theme": {
    "mode": "light",
    "light": "Zed (Default)",
  },
  "base_keymap": "JetBrains",
  /* font */ "ui_font_size": 16,
  "theme": {
    "mode": "light",
    "light": "One Light",
    "dark": "One Dark",
  },
}
"""


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "AppData" / "Roaming").mkdir(parents=True)
    (home / "AppData" / "Local" / "Programs").mkdir(parents=True)
    monkeypatch.setattr(A, "HOME", home)
    monkeypatch.setattr(A, "APPDATA", home / "AppData" / "Roaming")
    monkeypatch.setattr(A, "LOCALAPPDATA", home / "AppData" / "Local")
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(A, "codex_cli", lambda: None)          # test the file edit, not the user's CLIs
    monkeypatch.setattr(A, "opencode_cli", lambda: None)
    monkeypatch.setattr(A, "_which", lambda *n: None)
    return home


def test_jsonc_set_keeps_comments_and_roundtrips():
    entry = {"command": "C:\\py\\python.exe", "args": ["C:\\app\\mcp_server.py"], "env": {"MAKEAI_URL": BASE}}
    out = A.jsonc_set(ZED_SETTINGS, ["context_servers", "makeai"], entry)
    assert "// Zed settings" in out and "/* font */" in out
    data = A.jsonc_loads(out)
    assert data["context_servers"]["makeai"] == entry
    assert data["theme"]["dark"] == "One Dark" and data["ui_font_size"] == 16
    # replace in place, then remove again -> original content back
    entry2 = {**entry, "args": ["D:\\other.py"]}
    out2 = A.jsonc_set(out, ["context_servers", "makeai"], entry2)
    assert A.jsonc_loads(out2)["context_servers"]["makeai"] == entry2
    assert out2.count('"context_servers"') == 1
    out3 = A.jsonc_delete(out2, ["context_servers", "makeai"])
    assert A.jsonc_loads(out3)["context_servers"] == {}
    # other servers stay untouched
    other = A.jsonc_set('{"mcpServers": {"x": {"command": "a"}}}', ["mcpServers", "makeai"], entry)
    d = A.jsonc_loads(other)
    assert d["mcpServers"]["x"] == {"command": "a"} and d["mcpServers"]["makeai"] == entry
    d2 = A.jsonc_loads(A.jsonc_delete(other, ["mcpServers", "makeai"]))
    assert d2 == {"mcpServers": {"x": {"command": "a"}}}
    # last member removal takes the comma before it
    d3 = A.jsonc_delete('{\n  "a": 1,\n  "b": {"c": 2}\n}\n', ["b"])
    assert json.loads(d3) == {"a": 1}
    for empty in ("", "{}", "{\n}\n", "  \n"):
        assert A.jsonc_loads(A.jsonc_set(empty, ["mcpServers", "makeai"], entry)) == {"mcpServers": {"makeai": entry}}


@pytest.mark.parametrize("aid", ["cursor", "opencode", "antigravity", "devin", "cline", "zed", "kiro", "junie"])
def test_json_agents_connect_status_disconnect(fake_home, aid):
    a = A.get(aid)
    if aid == "zed":
        a.paths()[0].parent.mkdir(parents=True)
        a.paths()[0].write_text(ZED_SETTINGS, encoding="utf-8")
    assert not a.configured()
    where = a.connect(BASE)
    assert a.configured(), where
    for p in where.split(", "):
        data = A.jsonc_loads(Path(p).read_text(encoding="utf-8"))
        entry = A._get(data, a.keys)
        env = entry.get("env") or entry.get("environment")
        assert env["MAKEAI_AGENT"] == aid and env["MAKEAI_URL"] == BASE
        cmd = entry["command"] if isinstance(entry["command"], list) else [entry["command"], *entry["args"]]
        assert cmd == [A.console_python(), str(A.MCP_SCRIPT)]
    a.connect(BASE)                                           # again: still one entry, file still valid
    assert a.configured()
    if aid == "zed":
        text = a.paths()[0].read_text(encoding="utf-8")
        assert "// Zed settings" in text and text.count('"makeai"') == 1
        assert Path(str(a.paths()[0]) + ".makeai-backup").read_text(encoding="utf-8") == ZED_SETTINGS
    a.disconnect()
    assert not a.configured()


def test_codex_toml_edit_keeps_other_tables(fake_home):
    cfg = fake_home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    original = 'model = "gpt-5"\n\n[mcp_servers.node_repl]\ncommand = \'C:\\x\\node.exe\'\n\n[mcp_servers.node_repl.env]\nA = "1"\n\n[features]\nfoo = true\n'
    cfg.write_text(original, encoding="utf-8")
    a = A.get("codex")
    a.connect(BASE)
    d = tomli.loads(cfg.read_text(encoding="utf-8"))
    m = d["mcp_servers"]["makeai"]
    assert m["command"] == A.console_python() and m["args"] == [str(A.MCP_SCRIPT)]
    assert m["env"]["MAKEAI_AGENT"] == "codex" and d["mcp_servers"]["node_repl"]["env"] == {"A": "1"}
    assert d["features"] == {"foo": True} and d["model"] == "gpt-5"
    a.connect(BASE)
    assert cfg.read_text(encoding="utf-8").count("[mcp_servers.makeai]") == 1
    a.disconnect()
    assert "makeai" not in tomli.loads(cfg.read_text(encoding="utf-8"))["mcp_servers"]


def test_aider_read_list(fake_home, monkeypatch, tmp_path):
    monkeypatch.setattr(A.store, "sub", lambda name: tmp_path / "mk" / name)
    a = A.get("aider")
    conf = fake_home / ".aider.conf.yml"
    conf.write_text("model: sonnet\nread: CONVENTIONS.md\n", encoding="utf-8")
    a.connect(BASE)
    assert a.configured()
    import yaml
    d = yaml.safe_load(conf.read_text(encoding="utf-8"))
    assert d["model"] == "sonnet" and d["read"] == ["CONVENTIONS.md", str(a.guide())]
    guide = a.guide().read_text(encoding="utf-8")
    assert "makeai_start_training" in guide and "--agent aider call" in guide
    a.disconnect()
    assert yaml.safe_load(conf.read_text(encoding="utf-8"))["read"] == ["CONVENTIONS.md"]
    # block list and no read key
    assert yaml.safe_load(A.yaml_add_read("read:\n  - a.md\n", "b.md"))["read"] == ["a.md", "b.md"]
    assert yaml.safe_load(A.yaml_add_read("read: [a.md]\n", "b.md"))["read"] == ["a.md", "b.md"]
    assert yaml.safe_load(A.yaml_add_read("", "C:\\x\\b.md"))["read"] == ["C:\\x\\b.md"]


def test_mcp_identifies_agent(tmp_path):
    """The MCP server tells MakeAI which agent runs it (env from the config, else clientInfo)."""
    from makeai.claudemode import mcp_server as M
    assert M.identify("cursor", None)["name"] == "Cursor"
    assert M.identify(None, {"name": "claude-code"})["id"] == "claude"
    assert M.identify(None, {"name": "codex-mcp-client"})["id"] == "codex"
    assert M.identify(None, {"name": "Zed"})["id"] == "zed"
    assert M.identify(None, {"name": "windsurf-client"})["id"] == "devin"
    assert M.identify(None, {"name": "some-new-agent"})["name"] == "some-new-agent"


def test_cli_mode_without_makeai_reports_error():
    """`mcp_server.py call` (used by Aider) prints JSON and a non-zero exit when a tool fails."""
    p = subprocess.run([sys.executable, str(A.MCP_SCRIPT), "--agent", "aider", "--url", "http://127.0.0.1:9",
                        "call", "makeai_nope"], capture_output=True, text=True, timeout=60)
    assert p.returncode == 1 and "unknown tool" in json.loads(p.stdout)["error"]
    p = subprocess.run([sys.executable, str(A.MCP_SCRIPT), "tools"], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and "makeai_start_training(" in p.stdout
