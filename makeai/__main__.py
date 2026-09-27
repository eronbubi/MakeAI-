"""Start MakeAI.

    python -m makeai [--host 127.0.0.1] [--port 7860] [--no-browser] [--home DIR]
    pythonw run.py --window        desktop mode: own app window, no console, stops when the window is closed
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def _log_to_file_if_windowless(home: Path) -> None:
    """pythonw has no console: send output to a log file instead of losing it (or crashing on None streams)."""
    if sys.stdout is None or sys.stderr is None:
        f = open(home / "makeai.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or f
        sys.stderr = sys.stderr or f


def _makeai_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
            return json.loads(r.read().decode()).get("product") == "MakeAI"
    except Exception:
        return False


def _port_busy(port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _app_browser() -> str | None:
    """A Chromium browser that supports --app windows (Edge, Chrome, Brave)."""
    cands = []
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            cands += [Path(base) / "Microsoft/Edge/Application/msedge.exe", Path(base) / "Google/Chrome/Application/chrome.exe",
                      Path(base) / "BraveSoftware/Brave-Browser/Application/brave.exe"]
    for c in cands:
        if c.exists():
            return str(c)
    for name in ("msedge", "chrome", "google-chrome", "chromium", "brave"):
        p = shutil.which(name)
        if p:
            return p
    return None


def open_window(url: str, home: Path) -> subprocess.Popen | None:
    """Open MakeAI in its own app window. Returns the window process (None if only a normal tab could be opened)."""
    exe = _app_browser()
    if not exe:
        webbrowser.open(url)
        return None
    profile = home / "window-profile"          # own profile -> own process that ends when the window is closed
    return subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={profile}", "--no-first-run",
                             "--no-default-browser-check", "--window-size=1440,920", "--disable-features=Translate"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _message(title: str, text: str) -> None:
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, title, 0x40)
            return
        except Exception:
            pass
    print(f"{title}: {text}", file=sys.stderr)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--home")
    known, _ = pre.parse_known_args(sys.argv[1:])
    if known.home:                      # data folder (default: ~/MakeAI or $MAKEAI_HOME)
        os.environ["MAKEAI_HOME"] = known.home
    from . import PRODUCT, VENDOR, __version__, store
    _log_to_file_if_windowless(store.home())
    ap = argparse.ArgumentParser(prog="makeai", description=f"{PRODUCT} by {VENDOR}")
    s = store.load_settings()
    ap.add_argument("--host", default=s.get("share_host") or "127.0.0.1",
                    help="127.0.0.1 (default, this computer only) or 0.0.0.0 to let other computers open shared AIs")
    ap.add_argument("--port", type=int, default=int(s.get("share_port") or 7860))
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--window", action="store_true", help="desktop mode: own app window; stops when it is closed")
    ap.add_argument("--home", help="data folder (default ~/MakeAI)")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/"

    if _makeai_running(a.port):         # already running -> just show it
        if a.window:
            open_window(url, store.home())
        elif not a.no_browser:
            webbrowser.open(url)
        return
    if _port_busy(a.port):
        _message(PRODUCT, f"Port {a.port} is used by another program, so MakeAI cannot start.")
        sys.exit(1)

    import uvicorn

    from .server.app import create_app
    app = create_app()
    server = uvicorn.Server(uvicorn.Config(app, host=a.host, port=a.port, log_level="warning"))
    print(f"{PRODUCT} {__version__} by {VENDOR} - data in {store.home()} - {url}", flush=True)

    def after_start():
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.1)
        if a.window:
            win = open_window(url, store.home())
            threading.Thread(target=_stop_when_closed, args=(server, app, win), daemon=True).start()
        elif not a.no_browser:
            webbrowser.open(url)

    threading.Thread(target=after_start, daemon=True).start()
    server.run()


def _stop_when_closed(server, app, win) -> None:
    """Desktop mode: after the app window closes, stop MakeAI - but never while a training run is active."""
    from .train.runs import ACTIVE, list_runs
    S = app.state.S
    if win is not None:
        code = win.wait()
        print(f"app window process ended (exit {code})", flush=True)
    else:
        time.sleep(30)
    idle_since = None
    while not server.should_exit:
        busy = any(r.get("state") in ACTIVE for r in list_runs()) or time.time() - getattr(S, "ui_seen", 0) < 15
        if busy:
            idle_since = None
        else:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > 20:
                print("window closed and nothing running - stopping MakeAI", flush=True)
                server.should_exit = True
                return
        time.sleep(2)


if __name__ == "__main__":
    main()
