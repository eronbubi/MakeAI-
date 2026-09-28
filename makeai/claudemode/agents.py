"""Connect MakeAI to coding agents.

Every agent that speaks MCP gets the same stdio server (``mcp_server.py``); only
the place and format of its config differ. MakeAI writes that config for the
user (``connect``), shows whether it is there (``status``) and can take it out
again (``disconnect``). Before the first change to a file a copy is kept next to
it as ``<file>.makeai-backup``. JSON/JSONC files are edited in place so the
user's comments and formatting survive; the result is parsed again before it is
written, and nothing is written if that check fails.

Aider has no MCP support, so it gets a conventions file that tells it how to
call the same tools through ``mcp_server.py call <tool>``.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .. import store

MCP_SCRIPT = Path(__file__).resolve().parent / "mcp_server.py"
NAME = "makeai"
HOME = Path.home()
APPDATA = Path(os.environ.get("APPDATA") or HOME / "AppData" / "Roaming")
LOCALAPPDATA = Path(os.environ.get("LOCALAPPDATA") or HOME / "AppData" / "Local")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def console_python() -> str:
    """The MCP server talks over stdin/stdout, so it must run with python.exe, not pythonw.exe."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def server_env(base_url: str, agent_id: str) -> dict[str, str]:
    return {"MAKEAI_URL": base_url, "MAKEAI_AGENT": agent_id, "PYTHONUTF8": "1"}


def _norm(p: str) -> str:
    return p.replace("\\\\", "\\").replace("\\", "/").lower()


def _mentions_us(text: str) -> bool:
    return _norm(str(MCP_SCRIPT)) in _norm(text)


# ============================================================================ JSONC
def jsonc_loads(text: str) -> Any:
    """json.loads for JSON with // and /* */ comments and trailing commas (Zed, VS Code, OpenCode)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = _skip_string(text, i)
            out.append(text[i:j])
            i = j
        elif text.startswith("//", i):
            i = text.find("\n", i)
            i = n if i < 0 else i
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    clean = re.sub(r",(\s*[}\]])", r"\1", "".join(out))
    return json.loads(clean) if clean.strip() else {}


def _skip_string(t: str, i: int) -> int:
    i += 1
    while i < len(t):
        if t[i] == "\\":
            i += 2
            continue
        if t[i] == '"':
            return i + 1
        i += 1
    raise ValueError("unterminated string")


def _skip_ws(t: str, i: int) -> int:
    while i < len(t):
        if t[i].isspace():
            i += 1
        elif t.startswith("//", i):
            j = t.find("\n", i)
            i = len(t) if j < 0 else j + 1
        elif t.startswith("/*", i):
            j = t.find("*/", i + 2)
            i = len(t) if j < 0 else j + 2
        else:
            break
    return i


def _skip_value(t: str, i: int) -> int:
    if t[i] == '"':
        return _skip_string(t, i)
    if t[i] in "{[":
        depth = 0
        while i < len(t):
            i = _skip_ws(t, i)
            c = t[i]
            if c == '"':
                i = _skip_string(t, i)
                continue
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        raise ValueError("unbalanced brackets")
    m = re.compile(r"[^,}\]\s/]+").match(t, i)
    return m.end() if m else i


def _members(t: str, start: int):
    """(key, key_pos, value_pos, value_end) for each member of the object whose '{' is at start."""
    i = _skip_ws(t, start + 1)
    while i < len(t) and t[i] != "}":
        if t[i] == ",":
            i = _skip_ws(t, i + 1)
            continue
        k_end = _skip_string(t, i)
        key = json.loads(t[i:k_end])
        c = _skip_ws(t, k_end)
        if t[c] != ":":
            raise ValueError("expected ':'")
        v = _skip_ws(t, c + 1)
        v_end = _skip_value(t, v)
        yield key, i, v, v_end
        i = _skip_ws(t, v_end)


def _indent_at(t: str, pos: int) -> str:
    line = t[t.rfind("\n", 0, pos) + 1:pos]
    return line[:len(line) - len(line.lstrip())]


def _dump(value: Any, indent: str) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False).replace("\n", "\n" + indent)


def jsonc_set(text: str, keys: list[str], value: Any) -> str:
    """Set text[keys...] = value in a JSON/JSONC document, keeping everything else as it is."""
    if not text.strip():
        return json.dumps(_nest(keys, value), indent=2, ensure_ascii=False) + "\n"
    obj = _skip_ws(text, 0)
    if text[obj] != "{":
        raise ValueError("config file is not a JSON object")
    for depth, key in enumerate(keys):
        found = next((m for m in _members(text, obj) if m[0] == key), None)
        if found is None:
            rest = _nest(keys[depth + 1:], value)
            close = _skip_value(text, obj) - 1
            inner = text[obj + 1:close]
            ind = _indent_at(text, obj)
            child = ind + "  "
            body = f"\n{child}{json.dumps(key)}: {_dump(rest, child)}"
            has_members = next(_members(text, obj), None) is not None
            if has_members:
                return text[:obj + 1] + body + "," + text[obj + 1:]
            return text[:obj + 1] + body + inner.rstrip(" \t") + ("" if inner.endswith("\n") else "\n" + ind) + text[close:]
        _, _, v, v_end = found
        if depth == len(keys) - 1:
            return text[:v] + _dump(value, _indent_at(text, found[1])) + text[v_end:]
        if text[v] != "{":
            return text[:v] + _dump(_nest(keys[depth + 1:], value), _indent_at(text, found[1])) + text[v_end:]
        obj = v
    return text


def jsonc_delete(text: str, keys: list[str]) -> str:
    """Remove text[keys...] (and its comma) from a JSON/JSONC document, keeping everything else."""
    obj = _skip_ws(text, 0) if text.strip() else 0
    if not text.strip() or text[obj] != "{":
        return text
    for depth, key in enumerate(keys):
        found = next((m for m in _members(text, obj) if m[0] == key), None)
        if found is None:
            return text
        _, k, v, v_end = found
        if depth < len(keys) - 1:
            obj = v
            continue
        after = _skip_ws(text, v_end)
        if after < len(text) and text[after] == ",":
            cut = (k, after + 1)
        else:                                         # last member: take the comma in front of it
            j = k - 1
            while j >= 0 and text[j] in " \t\r\n":
                j -= 1
            cut = (j, v_end) if j >= 0 and text[j] == "," else (k, v_end)
        new = text[:cut[0]] + text[cut[1]:]
        ls = new.rfind("\n", 0, cut[0]) + 1
        le = new.find("\n", cut[0])
        if le >= 0 and not new[ls:le].strip():        # don't leave an empty line behind
            new = new[:ls] + new[le + 1:]
        return new
    return text


def _nest(keys: list[str], value: Any) -> Any:
    for k in reversed(keys):
        value = {k: value}
    return value


def _get(data: Any, keys: list[str]) -> Any:
    for k in keys:
        if not isinstance(data, dict) or k not in data:
            return None
        data = data[k]
    return data


# ============================================================================ agents
class Agent:
    id = ""
    name = ""
    kind = "json"               # json | toml | cli | aider
    docs = ""
    keys: list[str] = []        # JSON path of our entry
    restart = "Restart the app so it loads the MakeAI tools."

    def paths(self) -> list[Path]:
        return []

    def installed(self) -> bool:
        return False

    def entry(self, base_url: str) -> dict:
        return {"command": console_python(), "args": [str(MCP_SCRIPT)], "env": server_env(base_url, self.id)}

    # -------------------------------------------------------------- JSON default
    def target(self) -> Path:
        ps = self.paths()
        return next((p for p in ps if p.exists()), ps[0])

    def configured(self) -> bool:
        for p in self.paths():
            try:
                val = _get(jsonc_loads(p.read_text(encoding="utf-8-sig")), self.keys)
            except (OSError, ValueError):
                continue
            if val is not None and _mentions_us(json.dumps(val)):
                return True
        return False

    def connect(self, base_url: str) -> str:
        p = self.target()
        _write_checked(p, lambda t: jsonc_set(t, self.keys, self.entry(base_url)),
                       lambda d: _get(d, self.keys) == self.entry(base_url))
        return str(p)

    def disconnect(self) -> list[str]:
        done = []
        for p in self.paths():
            if p.exists() and _get(_load(p), self.keys) is not None:
                _write_checked(p, lambda t: jsonc_delete(t, self.keys), lambda d: _get(d, self.keys) is None)
                done.append(str(p))
        return done

    def snippet(self, base_url: str) -> str:
        return json.dumps(_nest(self.keys, self.entry(base_url)), indent=2)

    def info(self, base_url: str) -> dict[str, Any]:
        try:
            configured = self.configured()
        except Exception:
            configured = False
        return {"id": self.id, "name": self.name, "installed": self.installed(), "configured": configured,
                "config_path": str(self.target()) if self.paths() else None, "restart": self.restart,
                "snippet": self.snippet(base_url), "docs": self.docs, "kind": self.kind}


def _load(p: Path) -> Any:
    return jsonc_loads(p.read_text(encoding="utf-8-sig")) if p.exists() else {}


def _write_checked(p: Path, edit: Callable[[str], str], check: Callable[[Any], bool], parse=jsonc_loads) -> None:
    old = p.read_text(encoding="utf-8-sig") if p.exists() else ""
    new = edit(old)
    data = parse(new)
    if not check(data):
        raise RuntimeError(f"could not update {p} safely - nothing was changed")
    p.parent.mkdir(parents=True, exist_ok=True)
    backup = p.with_name(p.name + ".makeai-backup")
    if p.exists() and not backup.exists():
        shutil.copy2(p, backup)
    tmp = p.with_name(p.name + ".makeai-tmp")
    tmp.write_text(new, encoding="utf-8")
    store._replace_retry(tmp, p)


def _exists(*paths: Path | str | None) -> bool:
    return any(p and Path(p).exists() for p in paths)


def _which(*names: str) -> str | None:
    return next((w for w in (shutil.which(n) for n in names) if w), None)


def _vscode_ext(ext_dirs: list[Path], prefix: str) -> bool:
    return any(glob.glob(str(d / f"{prefix}*")) for d in ext_dirs)


class Claude(Agent):
    id, name, kind = "claude", "Claude Code", "cli"
    docs = "https://code.claude.com/docs/en/mcp"
    restart = "Open a new Claude Code chat."

    def paths(self):
        return [HOME / ".claude.json"]

    def installed(self):
        return _which("claude") is not None

    def configured(self):
        cli = _which("claude")
        if not cli:
            return False
        p = subprocess.run([cli, "mcp", "get", NAME], capture_output=True, text=True, timeout=30, encoding="utf-8",
                           errors="replace", creationflags=NO_WINDOW)
        return p.returncode == 0 and _mentions_us(p.stdout)

    def _cmd(self, base_url):
        env = [x for k, v in server_env(base_url, self.id).items() for x in ("-e", f"{k}={v}")]
        return ["mcp", "add", "--scope", "user", NAME, *env, "--", console_python(), str(MCP_SCRIPT)]

    def connect(self, base_url):
        cli = _which("claude")
        if not cli:
            raise RuntimeError("Claude Code is not installed on this computer")
        subprocess.run([cli, "mcp", "remove", "--scope", "user", NAME], capture_output=True, timeout=60, creationflags=NO_WINDOW)
        p = subprocess.run([cli, *self._cmd(base_url)], capture_output=True, text=True, timeout=60, encoding="utf-8",
                           errors="replace", creationflags=NO_WINDOW)
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout).strip()[-500:])
        return "claude mcp add --scope user"

    def disconnect(self):
        cli = _which("claude")
        if cli:
            subprocess.run([cli, "mcp", "remove", "--scope", "user", NAME], capture_output=True, timeout=60,
                           creationflags=NO_WINDOW)
        return ["claude mcp remove"]

    def snippet(self, base_url):
        return " ".join(f'"{c}"' if " " in c else c for c in ["claude", *self._cmd(base_url)])


def codex_cli() -> str | None:
    """codex on PATH, else the newest codex.exe the Codex app ships."""
    w = _which("codex")
    if w:
        return w
    found = glob.glob(str(LOCALAPPDATA / "OpenAI" / "Codex" / "bin" / "**" / "codex.exe"), recursive=True)
    return max(found, key=os.path.getmtime) if found else None


class Codex(Agent):
    id, name, kind = "codex", "Codex", "toml"
    docs = "https://learn.chatgpt.com/docs/extend/mcp?surface=cli"
    restart = "Start a new Codex chat (the Codex app or `codex` picks the tools up on start)."

    def paths(self):
        return [Path(os.environ.get("CODEX_HOME") or HOME / ".codex") / "config.toml"]

    def installed(self):
        return codex_cli() is not None or self.paths()[0].parent.exists()

    def _table(self) -> dict | None:
        p = self.paths()[0]
        if not p.exists():
            return None
        import tomli
        return (tomli.loads(p.read_text(encoding="utf-8-sig")).get("mcp_servers") or {}).get(NAME)

    def configured(self):
        t = self._table()
        return bool(t) and _mentions_us(json.dumps(t))

    def connect(self, base_url):
        cli = codex_cli()
        env = [x for k, v in server_env(base_url, self.id).items() for x in ("--env", f"{k}={v}")]
        if cli:
            subprocess.run([cli, "mcp", "remove", NAME], capture_output=True, timeout=60, creationflags=NO_WINDOW)
            p = subprocess.run([cli, "mcp", "add", NAME, *env, "--", console_python(), str(MCP_SCRIPT)],
                               capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace",
                               creationflags=NO_WINDOW)
            if p.returncode == 0 and self.configured():
                return str(self.paths()[0])
        # no usable CLI (or it cannot read this config version): edit config.toml as text
        p = self.paths()[0]
        _write_checked(p, lambda t: toml_set_server(t, base_url, self.id),
                       lambda d: _mentions_us(json.dumps((d.get("mcp_servers") or {}).get(NAME))), parse=_toml_loads)
        return str(p)

    def disconnect(self):
        p = self.paths()[0]
        if p.exists() and self._table() is not None:
            _write_checked(p, toml_remove_server, lambda d: NAME not in (d.get("mcp_servers") or {}), parse=_toml_loads)
            return [str(p)]
        return []

    def snippet(self, base_url):
        return toml_block(base_url, self.id).strip()


def _toml_loads(t: str):
    import tomli
    return tomli.loads(t)


def _tq(s: str) -> str:
    return json.dumps(s)          # a JSON string is a valid TOML basic string


def toml_block(base_url: str, agent_id: str) -> str:
    env = "\n".join(f"{k} = {_tq(v)}" for k, v in server_env(base_url, agent_id).items())
    return (f"\n[mcp_servers.{NAME}]\ncommand = {_tq(console_python())}\nargs = [{_tq(str(MCP_SCRIPT))}]\n"
            f"startup_timeout_sec = 30\n\n[mcp_servers.{NAME}.env]\n{env}\n")


def toml_remove_server(text: str) -> str:
    out, skip = [], False
    for line in text.splitlines(keepends=True):
        m = re.match(r"\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(#.*)?$", line)
        if m:
            name = m.group(1).replace('"', "").replace("'", "")
            skip = name == f"mcp_servers.{NAME}" or name.startswith(f"mcp_servers.{NAME}.")
        if not skip:
            out.append(line)
    return "".join(out).rstrip("\n") + "\n"


def toml_set_server(text: str, base_url: str, agent_id: str) -> str:
    rest = toml_remove_server(text) if text.strip() else ""
    return rest + toml_block(base_url, agent_id)


class Cursor(Agent):
    id, name = "cursor", "Cursor"
    keys = ["mcpServers", NAME]
    docs = "https://cursor.com/docs/context/mcp"
    restart = "Cursor loads it on its own; if the tools do not show up, open Settings → MCP or restart Cursor."

    def paths(self):
        return [HOME / ".cursor" / "mcp.json"]

    def entry(self, base_url):
        return {"type": "stdio", **super().entry(base_url)}

    def installed(self):
        return _exists(LOCALAPPDATA / "Programs" / "cursor", HOME / ".cursor") or _which("cursor") is not None


def opencode_cli() -> str | None:
    w = _which("opencode", "opencode-cli")
    if w:
        return w
    exe = LOCALAPPDATA / "Programs" / "@opencodedesktop" / "resources" / "opencode-cli.exe"
    return str(exe) if exe.exists() else None


class OpenCode(Agent):
    """OpenCode 1.x reads mcp.<name>, OpenCode 2 reads mcp.servers.<name>; its own CLI writes the right one."""
    id, name = "opencode", "OpenCode"
    keys = ["mcp", NAME]
    KEYS_V2 = ["mcp", "servers", NAME]
    docs = "https://opencode.ai/docs/mcp-servers/"
    restart = "Restart OpenCode."

    def paths(self):
        d = Path(os.environ.get("XDG_CONFIG_HOME") or HOME / ".config") / "opencode"
        return [d / "opencode.json", d / "opencode.jsonc"]

    def installed(self):
        return _exists(LOCALAPPDATA / "Programs" / "@opencodedesktop", self.paths()[0].parent) or opencode_cli() is not None

    def entry(self, base_url):
        return {"type": "local", "command": [console_python(), str(MCP_SCRIPT)],
                "environment": server_env(base_url, self.id), "enabled": True}

    def configured(self):
        for p in self.paths():
            try:
                d = jsonc_loads(p.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            for keys in (self.KEYS_V2, self.keys):
                v = _get(d, keys)
                if isinstance(v, dict) and v.get("command") and _mentions_us(json.dumps(v)):
                    return True
        return False

    def connect(self, base_url):
        cli = opencode_cli()
        if cli:
            self.disconnect()
            env = [x for k, v in server_env(base_url, self.id).items() for x in ("--env", f"{k}={v}")]
            p = subprocess.run([cli, "mcp", "add", "--global", *env, NAME, "--", console_python(), str(MCP_SCRIPT)],
                               capture_output=True, text=True, timeout=90, encoding="utf-8", errors="replace",
                               creationflags=NO_WINDOW, cwd=str(HOME))
            if p.returncode == 0 and self.configured():
                return str(self.target())
        p = self.target()
        if not p.exists():                                   # a new file gets the schema line as well
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {"$schema": "https://opencode.ai/config.json", "mcp": {NAME: self.entry(base_url)}}
            _write_checked(p, lambda _t: json.dumps(data, indent=2) + "\n", lambda d: _get(d, self.keys) == self.entry(base_url))
            return str(p)
        return super().connect(base_url)

    def disconnect(self):
        done = []
        for p in self.paths():
            for keys in (self.KEYS_V2, self.keys):
                if p.exists() and isinstance(_get(_load(p), keys), dict) and _get(_load(p), keys).get("command"):
                    _write_checked(p, lambda t, k=keys: jsonc_delete(t, k), lambda d, k=keys: _get(d, k) is None)
                    done.append(str(p))
        return done


class Antigravity(Agent):
    id, name = "antigravity", "Antigravity"
    keys = ["mcpServers", NAME]
    docs = "https://antigravity.google/docs/mcp"
    restart = "Restart Antigravity (or refresh in Agent → MCP Servers → Manage)."

    def paths(self):
        g = HOME / ".gemini"
        # Antigravity 2 reads ~/.gemini/config, the Antigravity IDE ~/.gemini/antigravity(-ide)
        return [g / "config" / "mcp_config.json", g / "antigravity-ide" / "mcp_config.json",
                g / "antigravity" / "mcp_config.json"]

    def installed(self):
        return _exists(LOCALAPPDATA / "Programs" / "Antigravity", HOME / ".gemini" / "antigravity")

    def targets(self) -> list[Path]:
        ps = [p for p in self.paths() if p.parent.exists()]
        return ps or self.paths()[:1]

    def connect(self, base_url):
        for p in self.targets():
            _write_checked(p, lambda t: jsonc_set(t, self.keys, self.entry(base_url)),
                           lambda d: _get(d, self.keys) == self.entry(base_url))
        return ", ".join(str(p) for p in self.targets())


class Devin(Agent):
    id, name = "devin", "Devin"
    keys = ["mcpServers", NAME]
    docs = "https://docs.devin.ai/cli/extensibility/mcp/configuration"
    restart = "Restart Devin (or refresh the MCP servers in its settings)."

    def paths(self):
        # Devin (desktop app and Devin CLI): %APPDATA%\devin on Windows, ~/.config/devin elsewhere.
        # Cloud Devin runs in a VM and cannot reach MakeAI on this computer. Windsurf: ~/.codeium/windsurf
        cfg = APPDATA / "devin" if os.name == "nt" else Path(os.environ.get("XDG_CONFIG_HOME") or HOME / ".config") / "devin"
        return [cfg / "mcp_config.json", HOME / ".codeium" / "windsurf" / "mcp_config.json"]

    def installed(self):
        return _exists(LOCALAPPDATA / "Programs" / "Devin", LOCALAPPDATA / "Programs" / "Windsurf") or \
            _which("devin", "devin-desktop", "windsurf") is not None

    def target(self):
        return self.paths()[0] if _exists(LOCALAPPDATA / "Programs" / "Devin", self.paths()[0].parent) else self.paths()[1]


class Cline(Agent):
    id, name = "cline", "Cline"
    keys = ["mcpServers", NAME]
    docs = "https://docs.cline.bot/mcp/configuring-mcp-servers"
    restart = "Cline reloads the file automatically; otherwise reload the editor window."
    EXT = "saoudrizwan.claude-dev"

    def hosts(self) -> list[tuple[Path, Path]]:
        """(extensions dir, globalStorage dir) of the VS Code family editors."""
        return [(HOME / ".vscode" / "extensions", APPDATA / "Code" / "User" / "globalStorage"),
                (HOME / ".vscode-insiders" / "extensions", APPDATA / "Code - Insiders" / "User" / "globalStorage"),
                (HOME / ".cursor" / "extensions", APPDATA / "Cursor" / "User" / "globalStorage"),
                (HOME / ".devin" / "extensions", APPDATA / "devin" / "User" / "globalStorage"),
                (HOME / ".windsurf" / "extensions", APPDATA / "Windsurf" / "User" / "globalStorage")]

    def paths(self):
        # current Cline (extension and CLI) share ~/.cline/data; older extension builds read their globalStorage
        shared = Path(os.environ.get("CLINE_MCP_SETTINGS_PATH") or
                      Path(os.environ.get("CLINE_DATA_DIR") or HOME / ".cline" / "data") / "settings" / "cline_mcp_settings.json")
        old = [gs / self.EXT / "settings" / "cline_mcp_settings.json" for ext, gs in self.hosts()
               if (gs / self.EXT).exists()]
        return [shared, *old]

    def installed(self):
        return any(_vscode_ext([ext], self.EXT) for ext, _ in self.hosts()) or _which("cline") is not None

    def entry(self, base_url):
        return {"type": "stdio", **super().entry(base_url), "disabled": False, "autoApprove": [], "timeout": 60}

    def connect(self, base_url):
        ps = self.paths()
        for p in ps:
            _write_checked(p, lambda t: jsonc_set(t, self.keys, self.entry(base_url)),
                           lambda d: _get(d, self.keys) == self.entry(base_url))
        return ", ".join(map(str, ps))


class Zed(Agent):
    id, name = "zed", "Zed"
    keys = ["context_servers", NAME]
    docs = "https://zed.dev/docs/ai/mcp"
    restart = "Zed starts the tools as soon as a project is open (Agent Panel → Settings shows the server)."

    def paths(self):
        if os.name == "nt":
            return [APPDATA / "Zed" / "settings.json"]
        return [Path(os.environ.get("XDG_CONFIG_HOME") or HOME / ".config") / "zed" / "settings.json"]

    def installed(self):
        return _exists(LOCALAPPDATA / "Programs" / "Zed", self.paths()[0].parent) or _which("zed") is not None

    def entry(self, base_url):
        return {**super().entry(base_url), "enabled": True}


class Kiro(Agent):
    id, name = "kiro", "Kiro"
    keys = ["mcpServers", NAME]
    docs = "https://kiro.dev/docs/mcp/configuration/"
    restart = "Kiro reloads MCP config on save; otherwise reconnect in the MCP Servers panel."

    def paths(self):
        return [HOME / ".kiro" / "settings" / "mcp.json"]

    def installed(self):
        return _exists(LOCALAPPDATA / "Programs" / "Kiro") or _which("kiro", "kiro-cli") is not None

    def entry(self, base_url):
        return {**super().entry(base_url), "disabled": False, "autoApprove": []}


class Junie(Agent):
    id, name = "junie", "Junie"
    keys = ["mcpServers", NAME]
    docs = "https://junie.jetbrains.com/docs/junie-cli-mcp-configuration.html"
    restart = "Restart the JetBrains IDE (or Junie CLI) so Junie loads the tools."

    def paths(self):
        return [HOME / ".junie" / "mcp" / "mcp.json"]

    def installed(self):
        if _which("junie") or (HOME / ".junie").exists():
            return True
        roots = [APPDATA / "JetBrains", LOCALAPPDATA / "JetBrains"]
        return any(glob.glob(str(r / "*" / "plugins" / "*junie*"), recursive=False) for r in roots)


class Aider(Agent):
    """Aider has no MCP client. It reads a conventions file and runs the tools through the command line."""
    id, name, kind = "aider", "Aider", "aider"
    docs = "https://aider.chat/docs/usage/conventions.html"
    restart = "Start Aider again - it reads the MakeAI instructions on start."

    def conf(self) -> Path:
        return HOME / ".aider.conf.yml"

    def guide(self) -> Path:
        return store.sub("agents") / "aider" / "MAKEAI.md"

    def paths(self):
        return [self.conf()]

    def installed(self):
        return _which("aider") is not None or self.conf().exists()

    def configured(self):
        c = self.conf()
        return c.exists() and _norm(str(self.guide())) in _norm(c.read_text(encoding="utf-8-sig")) and self.guide().exists()

    def command(self, base_url: str) -> str:
        url = "" if base_url == "http://127.0.0.1:7860" else f" --url {base_url}"
        return f'"{console_python()}" "{MCP_SCRIPT}" --agent aider{url} call'

    def write_guide(self, base_url: str) -> Path:
        g = self.guide()
        g.parent.mkdir(parents=True, exist_ok=True)
        from .mcp_server import TOOLS
        tools = "\n".join(f"- `{n}` - {t['schema']['description']}" + _args_hint(t["schema"]) for n, t in TOOLS.items())
        cmd = self.command(base_url)
        g.write_text(AIDER_GUIDE.format(cmd=cmd, url=base_url, tools=tools), encoding="utf-8")
        return g

    def connect(self, base_url):
        g = self.write_guide(base_url)
        c = self.conf()
        text = c.read_text(encoding="utf-8-sig") if c.exists() else ""
        if _norm(str(g)) not in _norm(text):
            new = yaml_add_read(text, str(g))
            if c.exists() and not c.with_name(c.name + ".makeai-backup").exists():
                shutil.copy2(c, c.with_name(c.name + ".makeai-backup"))
            c.write_text(new, encoding="utf-8")
        return str(c)

    def disconnect(self):
        c = self.conf()
        if not c.exists():
            return []
        g = _norm(str(self.guide()))
        lines = c.read_text(encoding="utf-8-sig").splitlines(keepends=True)
        keep = [ln for ln in lines if g not in _norm(ln) and ln.strip() != "# MakeAI tools (added by MakeAI)"]
        c.write_text("".join(keep), encoding="utf-8")
        return [str(c)]

    def snippet(self, base_url):
        return f"# ~/.aider.conf.yml\nread:\n  - {self.guide()}\n\n# Aider then runs MakeAI tools like:\n{self.command(base_url)} makeai_overview"


def _args_hint(schema: dict) -> str:
    props = schema["inputSchema"].get("properties") or {}
    if not props:
        return ""
    req = set(schema["inputSchema"].get("required") or [])
    return " Arguments: " + ", ".join(f"{k}{'' if k in req else '?'}" for k in props) + "."


def yaml_add_read(text: str, path: str) -> str:
    """Add ``path`` to the ``read:`` list of an aider YAML config (keeps the rest untouched)."""
    item = json.dumps(path)
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"^read\s*:\s*(.*?)\s*$", ln)
        if not m:
            continue
        val = m.group(1)
        if val.startswith("["):                                       # flow list
            inner = val[1:val.rindex("]")].strip()
            lines[i] = f"read: [{inner + ', ' if inner else ''}{item}]"
        elif val and not val.startswith("#"):                         # single value -> list
            lines[i:i + 1] = ["read:", f"  - {val}", f"  - {item}"]
        else:                                                         # block list: append after its items
            j = i + 1
            while j < len(lines) and re.match(r"^\s+-|^-", lines[j]):
                j += 1
            lines.insert(j, f"  - {item}")
        return "\n".join(lines) + "\n"
    head = text.rstrip("\n") + ("\n\n" if text.strip() else "")
    return head + f"# MakeAI tools (added by MakeAI)\nread:\n  - {item}\n"


AIDER_GUIDE = """# MakeAI tools

MakeAI (local AI training app, {url}) is on this computer. When the user asks you to create, train,
watch, test or export an AI in MakeAI, use these tools by running shell commands. The user watches
what you do live in the MakeAI window (view only) and talks to you here.

Run a tool:

    {cmd} <tool> key=value key2=value2

or pass JSON: `{cmd} <tool> "{{\\"goal\\": \\"...\\"}}"`. Values are read as JSON when possible
(numbers, true/false, lists), otherwise as text. Every command prints its result as JSON.

Start with `makeai_session_start goal="..."`, use `makeai_wait` while a training run is going and
finish with `makeai_session_end summary="..."`. Don't delete anything and don't make an AI public
unless the user asks.

Tools:

{tools}
"""


AGENTS: list[Agent] = [Claude(), Codex(), Cursor(), OpenCode(), Antigravity(), Devin(), Cline(), Aider(), Zed(),
                       Kiro(), Junie()]
BY_ID = {a.id: a for a in AGENTS}


def get(agent_id: str) -> Agent:
    if agent_id not in BY_ID:
        raise KeyError(f"unknown agent {agent_id}")
    return BY_ID[agent_id]
