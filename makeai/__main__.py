"""Start MakeAI:  python -m makeai [--host 127.0.0.1] [--port 7860] [--no-browser]"""
from __future__ import annotations

import argparse
import threading
import webbrowser


def main() -> None:
    import os
    import sys
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--home")
    known, _ = pre.parse_known_args(sys.argv[1:])
    if known.home:                      # data folder (default: ~/MakeAI or $MAKEAI_HOME)
        os.environ["MAKEAI_HOME"] = known.home
    from . import PRODUCT, VENDOR, __version__, store
    ap = argparse.ArgumentParser(prog="makeai", description=f"{PRODUCT} by {VENDOR}")
    s = store.load_settings()
    ap.add_argument("--host", default=s.get("share_host") or "127.0.0.1",
                    help="127.0.0.1 (default, this computer only) or 0.0.0.0 to let other computers open shared AIs")
    ap.add_argument("--port", type=int, default=int(s.get("share_port") or 7860))
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--home", help="data folder (default ~/MakeAI)")
    a = ap.parse_args()
    import uvicorn

    from .server.app import create_app
    url = f"http://127.0.0.1:{a.port}/"
    print(f"{PRODUCT} {__version__} by {VENDOR} - data in {store.home()}")
    print(f"open {url}")
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
