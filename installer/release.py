"""Publish a GitHub release with the installer:  python installer/release.py v1.0.0

Uses the GitHub login that git already has (Windows Credential Manager via
`git credential fill`); the token is only held in memory for the API calls.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "eronbubi/MakeAI-"


def token() -> str:
    out = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                         capture_output=True, text=True, timeout=60, check=True).stdout
    cred = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    if not cred.get("password"):
        raise SystemExit("no GitHub login stored for git")
    return cred["password"]


def api(method: str, url: str, tok: str, body=None, data: bytes | None = None, ctype="application/json"):
    headers = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json", "User-Agent": "makeai-release",
               "X-GitHub-Api-Version": "2022-11-28"}
    if body is not None:
        data = json.dumps(body).encode()
    if data is not None:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raise SystemExit(f"GitHub API {method} {url.split('?')[0]} -> {e.code}: {e.read().decode()[:400]}")


def main(tag: str) -> None:
    tok = token()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    notes = (ROOT / "installer" / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    rel = api("POST", f"https://api.github.com/repos/{REPO}/releases", tok,
              {"tag_name": tag, "target_commitish": sha, "name": f"MakeAI {tag.lstrip('v')}", "body": notes,
               "draft": False, "prerelease": False, "make_latest": "true"})
    upload = rel["upload_url"].split("{")[0]
    for f in ("MakeAI-Setup.exe", "MakeAI-Setup.exe.sha256"):
        p = ROOT / "dist" / f
        ctype = "application/vnd.microsoft.portable-executable" if f.endswith(".exe") else "text/plain"
        a = api("POST", f"{upload}?name={f}", tok, data=p.read_bytes(), ctype=ctype)
        print(f"uploaded {a['name']} ({a['size']:,} bytes) -> {a['browser_download_url']}")
    print(f"release: {rel['html_url']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1.1.0")
