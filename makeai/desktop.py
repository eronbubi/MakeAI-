"""Put MakeAI on the desktop:  python -m makeai.desktop

Creates the app icon (``makeai/assets/makeai.ico``) and a desktop shortcut that
starts MakeAI without a console window in its own app window.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICON = Path(__file__).resolve().parent / "assets" / "makeai.ico"


def make_icon(path: Path = ICON) -> Path:
    from PIL import Image, ImageDraw
    n = 256
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, n - 8, n - 8), radius=56, fill=(124, 58, 237, 255))
    s = n / 32                                           # same drawing as the web favicon (32x32 grid)
    pts = [(8 * s, 23 * s), (8 * s, 9 * s), (16 * s, 17 * s), (24 * s, 9 * s), (24 * s, 23 * s)]
    d.line(pts, fill=(255, 255, 255, 255), width=int(3.2 * s), joint="curve")
    for x, y in (pts[0], pts[-1], pts[1], pts[3]):
        r = 1.6 * s
        d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 255))
    r = 2.3 * s
    d.ellipse((16 * s - r, 17 * s - r + 3, 16 * s + r, 17 * s + r + 3), fill=(18, 16, 22, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path


def pythonw() -> str:
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    return str(w if w.exists() else exe)


def create_shortcut(name: str = "MakeAI") -> Path:
    if os.name != "nt":
        raise SystemExit("desktop shortcuts are created on Windows only")
    icon = make_icon()
    ps = r"""
$desk = [Environment]::GetFolderPath('Desktop')
$lnk = Join-Path $desk '{name}.lnk'
$sh = New-Object -ComObject WScript.Shell
$s = $sh.CreateShortcut($lnk)
$s.TargetPath = '{target}'
$s.Arguments = '"{run}" --window'
$s.WorkingDirectory = '{cwd}'
$s.IconLocation = '{icon},0'
$s.Description = 'MakeAI by Convergent - create, train and run your own AI'
$s.Save()
Write-Output $lnk
""".format(name=name, target=pythonw(), run=ROOT / "run.py", cwd=ROOT, icon=icon)
    out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps], capture_output=True,
                         text=True, encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if out.returncode != 0:
        raise SystemExit(out.stderr.strip() or "could not create the shortcut")
    return Path(out.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    print(create_shortcut())
