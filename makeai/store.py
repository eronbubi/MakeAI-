"""Filesystem layout and small persistence helpers.

Everything MakeAI owns lives under one home directory (``MAKEAI_HOME`` or
``~/MakeAI``). JSON files are written atomically (temp file + rename) so a
crash or a hard kill can never leave a half-written manifest or checkpoint
index behind.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any


def home() -> Path:
    root = Path(os.environ.get("MAKEAI_HOME") or (Path.home() / "MakeAI"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def sub(name: str) -> Path:
    p = home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path: return sub("models")
def datasets_dir() -> Path: return sub("datasets")
def tokenizers_dir() -> Path: return sub("tokenizers")
def runs_dir() -> Path: return sub("runs")
def projects_dir() -> Path: return sub("projects")
def exports_dir() -> Path: return sub("exports")
def backups_dir() -> Path: return sub("backups")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace_retry(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _replace_retry(src: str, dst: Path, attempts: int = 100) -> None:
    # On Windows the rename fails while another process has the target open for reading.
    # Readers hold it for microseconds, so a short retry loop is enough.
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.01 + 0.002 * i)


def write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(Path(path), json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"))


def read_json(path: Path, default: Any = None) -> Any:
    for i in range(50):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default
        except PermissionError:   # target is being replaced right now (Windows)
            time.sleep(0.01)
    return default


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower()).strip("-.")
    return s or "untitled"


def new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# --------------------------------------------------------------------- settings
DEFAULT_SETTINGS: dict[str, Any] = {
    "profile": {"name": "", "username": ""},
    "instant_kill": False,
    "auto_optimize": False,
    "share_host": "127.0.0.1",
    "share_port": 7860,
    "discover_hubs": [],          # URLs of other MakeAI instances to browse
    "simulation_mode": False,     # clearly labelled demo mode; never on by default
}


def settings_path() -> Path:
    return home() / "settings.json"


def load_settings() -> dict[str, Any]:
    data = read_json(settings_path(), {}) or {}
    merged = json.loads(json.dumps(DEFAULT_SETTINGS))
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return merged


def save_settings(patch: dict[str, Any]) -> dict[str, Any]:
    cur = load_settings()
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k].update(v)
        else:
            cur[k] = v
    write_json(settings_path(), cur)
    return cur
