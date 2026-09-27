"""Build MakeAI-Setup.exe:  python installer/build.py

1. builds the UI (npm run build) so the bundled app has the current interface,
2. packs the tracked project files (git ls-files) into payload.zip,
3. runs PyInstaller (one file, no console) with the icon, license and payload.
Output: dist/MakeAI-Setup.exe and its SHA-256.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "installer"
DIST = ROOT / "dist"


def main() -> None:
    subprocess.run("npm run build", cwd=ROOT / "ui", shell=True, check=True)
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.split("\n")
    skip = ("ui/", "tests/", "installer/", ".gitignore", "pytest.ini")
    files = [f for f in files if f and not f.startswith(skip) and (ROOT / f).is_file()]
    BUILD.mkdir(parents=True, exist_ok=True)
    payload = BUILD / "payload.zip"
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, f)
    print(f"payload: {len(files)} files, {payload.stat().st_size / 2**20:.1f} MB")
    icon = ROOT / "makeai" / "assets" / "makeai.ico"
    sep = ";"
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
                    "--name", "MakeAI-Setup", "--icon", str(icon),
                    "--add-data", f"{payload}{sep}.", "--add-data", f"{ROOT / 'LICENSE'}{sep}.",
                    "--add-data", f"{icon}{sep}.",
                    "--distpath", str(DIST), "--workpath", str(BUILD / "work"), "--specpath", str(BUILD),
                    str(ROOT / "installer" / "setup_makeai.py")], check=True)
    exe = DIST / "MakeAI-Setup.exe"
    digest = hashlib.sha256(exe.read_bytes()).hexdigest()
    (DIST / "MakeAI-Setup.exe.sha256").write_text(f"{digest}  MakeAI-Setup.exe\n", encoding="utf-8")
    print(f"{exe} {exe.stat().st_size / 2**20:.1f} MB sha256 {digest}")


if __name__ == "__main__":
    main()
