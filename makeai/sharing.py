"""Sharing (private / link / public) and Discover.

A MakeAI instance is its own hub: public AIs are listed at ``/api/public/index``
and downloadable as ``.makeai`` packages. Discover merges this instance's
public AIs with the public indexes of other MakeAI instances the user adds
(``discover_hubs`` in settings). Nothing is listed that does not exist.
"""
from __future__ import annotations

import json
import secrets
import time
import urllib.request
from typing import Any

from . import registry, store

VISIBILITY = ("private", "link", "public")
CATEGORIES = ("popular", "new", "coding", "reasoning", "general", "experimental", "small", "large")


def set_visibility(uid: str, visibility: str) -> dict[str, Any]:
    if visibility not in VISIBILITY:
        raise ValueError(f"visibility must be one of {VISIBILITY}")
    m = registry.load_manifest(uid)
    sh = m.setdefault("sharing", {"visibility": "private", "token": None})
    if visibility == "private":
        sh["token"] = None
    elif not sh.get("token"):
        sh["token"] = secrets.token_urlsafe(18)
    sh["visibility"] = visibility
    sh["changed"] = store.now_iso()
    m.setdefault("provenance", []).append({"event": "sharing", "visibility": visibility, "at": store.now_iso()})
    registry.save_manifest(m)
    return m


def by_token(token: str) -> dict[str, Any] | None:
    if not token:
        return None
    for m in registry.list_models():
        sh = m.get("sharing") or {}
        if sh.get("token") and secrets.compare_digest(sh["token"], token) and sh.get("visibility") in ("link", "public"):
            return m
    return None


def by_path(username: str, slug: str, version: str) -> dict[str, Any] | None:
    try:
        m = registry.load_manifest(registry.uid_for(username, slug, version))
    except (FileNotFoundError, ValueError):
        return None
    return m if (m.get("sharing") or {}).get("visibility") == "public" else None


def public_card(m: dict[str, Any], base_url: str = "") -> dict[str, Any]:
    oc = m.get("original_creator") or m["creator"]
    tok = (m.get("sharing") or {}).get("token")
    return {
        "id": m["id"], "name": m.get("display_name") or m["name"], "version": m["version"],
        "description": m.get("description", ""), "tags": m.get("tags", []), "icon": m.get("icon", ""),
        "creator": m["creator"], "original_creator": oc, "imported_by": m.get("imported_by", []),
        "created": m.get("created"), "updated": m.get("updated"), "param_count": m.get("param_count"),
        "context_length": m.get("context_length"), "format": m.get("source_format") or m.get("format"),
        "backend": m.get("backend") or "native", "architecture": m.get("architecture_summary"),
        "required_hardware": m.get("required_hardware"), "downloads": m.get("downloads", 0),
        "status": m.get("status"), "runnable": m.get("runnable"), "method": m.get("method"),
        "page": f"{base_url}/s/{tok}" if tok else None,
        "download": f"{base_url}/api/public/package/{tok}" if tok else None,
    }


def public_index(base_url: str = "") -> list[dict[str, Any]]:
    return [public_card(m, base_url) for m in registry.list_models()
            if (m.get("sharing") or {}).get("visibility") == "public"]


def count_download(uid: str) -> None:
    m = registry.load_manifest(uid)
    m["downloads"] = int(m.get("downloads", 0)) + 1
    store.write_json(registry.model_dir(uid) / "manifest.json", m)   # no 'updated' bump


def fetch_hub(url: str, timeout: float = 4.0) -> tuple[list[dict[str, Any]], str | None]:
    url = url.rstrip("/")
    try:
        with urllib.request.urlopen(url + "/api/public/index", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        items = data.get("items", data) if isinstance(data, dict) else data
        for it in items:
            it["hub"] = url
            for k in ("page", "download"):
                if it.get(k) and it[k].startswith("/"):
                    it[k] = url + it[k]
        return items, None
    except Exception as e:
        return [], f"{url}: {e}"


def _category_match(c: dict[str, Any], cat: str) -> bool:
    tags = {t.lower() for t in c.get("tags") or []}
    n = c.get("param_count") or 0
    if cat in ("coding", "reasoning", "general", "experimental"):
        return cat in tags or (cat == "coding" and "code" in tags)
    if cat == "small":
        return 0 < n < 1e9
    if cat == "large":
        return n >= 1e9
    return True


def discover(category: str | None = None, query: str = "", base_url: str = "") -> dict[str, Any]:
    items = [dict(c, hub="this computer") for c in public_index(base_url)]
    errors = []
    for hub in store.load_settings().get("discover_hubs") or []:
        got, err = fetch_hub(hub)
        items += got
        if err:
            errors.append(err)
    if query:
        q = query.lower()
        items = [c for c in items if q in (c.get("name") or "").lower() or q in (c.get("description") or "").lower()
                 or q in (c.get("creator", {}).get("username") or "") or any(q in t.lower() for t in c.get("tags") or [])]
    if category and category not in ("popular", "new"):
        items = [c for c in items if _category_match(c, category)]
    if category == "popular":
        items.sort(key=lambda c: c.get("downloads", 0), reverse=True)
    else:
        items.sort(key=lambda c: c.get("created") or "", reverse=True)
    return {"items": items, "errors": errors, "categories": CATEGORIES}


def download_to_temp(url: str) -> str:
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".makeai", dir=str(store.sub("downloads")))
    with urllib.request.urlopen(url, timeout=30) as r, open(fd, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return path
