"""Projects: a folder per project holding source, configs, tests, docs and links to AIs/datasets/tokenizers."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import store

FOLDERS = ("src", "models", "datasets", "tokenizers", "configs", "evaluation", "tests", "docs")
LINK_KINDS = ("models", "datasets", "tokenizers", "runs")
MAX_EDIT_BYTES = 2 * 2**20


def _root(name: str) -> Path:
    slug = store.slugify(name)
    d = store.projects_dir() / slug
    if not (d / "project.json").exists():
        raise FileNotFoundError(f"project {name} not found")
    return d


def safe_path(name: str, rel: str) -> Path:
    root = _root(name).resolve()
    p = (root / rel).resolve()
    if p != root and root not in p.parents:
        raise PermissionError("path escapes the project folder")
    return p


def create(name: str, description: str = "") -> dict[str, Any]:
    slug = store.slugify(name)
    d = store.projects_dir() / slug
    if (d / "project.json").exists():
        raise FileExistsError(f"project {slug} exists")
    for f in FOLDERS:
        (d / f).mkdir(parents=True, exist_ok=True)
    (d / "docs" / "README.md").write_text(f"# {name}\n\n{description}\n", encoding="utf-8")
    meta = {"name": name, "slug": slug, "description": description, "created": store.now_iso(),
            "links": {k: [] for k in LINK_KINDS}}
    store.write_json(d / "project.json", meta)
    return meta


def list_projects() -> list[dict[str, Any]]:
    out = []
    for d in sorted(store.projects_dir().iterdir()):
        m = store.read_json(d / "project.json")
        if m:
            out.append(m)
    return out


def get(name: str) -> dict[str, Any]:
    return store.read_json(_root(name) / "project.json")


def tree(name: str) -> list[dict[str, Any]]:
    root = _root(name)
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if rel == "project.json" or any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        out.append({"path": rel, "dir": p.is_dir(), "size": p.stat().st_size if p.is_file() else None})
    return out


def read_file(name: str, rel: str) -> dict[str, Any]:
    p = safe_path(name, rel)
    if p.stat().st_size > MAX_EDIT_BYTES:
        return {"path": rel, "binary": True, "size": p.stat().st_size, "content": None}
    data = p.read_bytes()
    try:
        return {"path": rel, "binary": False, "content": data.decode("utf-8")}
    except UnicodeDecodeError:
        return {"path": rel, "binary": True, "size": len(data), "content": None}


def write_file(name: str, rel: str, content: str) -> dict[str, Any]:
    p = safe_path(name, rel)
    if p.is_dir():
        raise IsADirectoryError(rel)
    store.atomic_write_bytes(p, content.encode("utf-8"))
    return {"path": rel, "size": p.stat().st_size}


def make_dir(name: str, rel: str) -> None:
    safe_path(name, rel).mkdir(parents=True, exist_ok=True)


def delete_path(name: str, rel: str) -> None:
    p = safe_path(name, rel)
    if p == _root(name).resolve():
        raise PermissionError("cannot delete the project root")
    shutil.rmtree(p) if p.is_dir() else p.unlink()


def link(name: str, kind: str, ref: str, on: bool = True) -> dict[str, Any]:
    if kind not in LINK_KINDS:
        raise ValueError(f"kind must be one of {LINK_KINDS}")
    meta = get(name)
    lst = meta.setdefault("links", {}).setdefault(kind, [])
    if on and ref not in lst:
        lst.append(ref)
    if not on and ref in lst:
        lst.remove(ref)
    store.write_json(_root(name) / "project.json", meta)
    return meta
