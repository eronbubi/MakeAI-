"""MakeAI Setup (Windows).

Built into MakeAI-Setup.exe by installer/build.py. It installs everything MakeAI
needs into one folder - nothing is added to the system Python:

    <dir>/python   official Python 3.10 embeddable build + pip + PyTorch (CUDA if an NVIDIA GPU is present)
    <dir>/app      MakeAI (bundled inside this exe)

then creates desktop / Start menu shortcuts and starts MakeAI. Running it again
updates MakeAI and keeps already-installed packages. User data stays in
%USERPROFILE%\\MakeAI and is never touched.

    MakeAI-Setup.exe                                   window
    MakeAI-Setup.exe --silent --accept-license [--dir D] [--no-shortcuts] [--no-launch]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

VERSION = "1.1.0"
PY_VER = "3.10.11"
PY_URL = f"https://www.python.org/ftp/python/{PY_VER}/python-{PY_VER}-embed-amd64.zip"
PY_SHA256 = "608619f8619075629c9c69f361352a0da6ed7e62f83a0e19c63e0ea32eb7629d"
GET_PIP = "https://bootstrap.pypa.io/get-pip.py"
TORCH = "torch==2.3.1"
TORCH_INDEX = {"cuda": "https://download.pytorch.org/whl/cu121", "cpu": "https://download.pytorch.org/whl/cpu"}
LLAMA_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ENV = {**os.environ, "PYTHONNOUSERSITE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
SITECUSTOMIZE = """# MakeAI: ignore the user's own site-packages (%APPDATA%/Python) so this Python stays self-contained
import site, sys
site.ENABLE_USER_SITE = False
_u = site.getusersitepackages()
sys.path[:] = [p for p in sys.path if not p.lower().startswith(_u.lower())]
"""


def resource(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


def default_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "Programs" / "MakeAI"


class Installer:
    def __init__(self, target: Path, shortcuts: bool, launch: bool, say, progress):
        self.dir = target
        self.py_dir = target / "python"
        self.app_dir = target / "app"
        self.shortcuts = shortcuts
        self.launch = launch
        self.say = say
        self.progress = progress
        self.log_path = target / "install.log"

    # --------------------------------------------------------------- helpers
    def log(self, msg: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    def download(self, url: str, dst: Path, label: str, lo: float, hi: float) -> Path:
        self.say(f"Downloading {label}…")
        self.log(f"download {url}")
        req = urllib.request.Request(url, headers={"User-Agent": f"MakeAI-Setup/{VERSION}"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while chunk := r.read(1 << 16):
                f.write(chunk)
                done += len(chunk)
                if total:
                    self.progress(lo + (hi - lo) * done / total)
        return dst

    def run(self, args: list[str], label: str, lo: float, hi: float, check: bool = True) -> int:
        """Run a command, log its output, move the progress bar slowly between lo and hi."""
        self.say(label)
        self.log("$ " + " ".join(args))
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", creationflags=NO_WINDOW, cwd=str(self.dir), env=ENV)
        n = 0
        with open(self.log_path, "a", encoding="utf-8") as logf:
            for line in p.stdout:
                logf.write(line)
                n += 1
                self.progress(min(hi, lo + (hi - lo) * (1 - 1 / (1 + n / 40))))
                if line.startswith(("Downloading", "Collecting", "Installing")):
                    self.say(f"{label} {line.strip()[:90]}")
        code = p.wait()
        if check and code != 0:
            raise RuntimeError(f"{label.rstrip('…')} failed (exit {code}). Details: {self.log_path}")
        self.progress(hi)
        return code

    @property
    def python(self) -> str:
        return str(self.py_dir / "python.exe")

    def state(self) -> dict:
        try:
            return json.loads((self.dir / "setup-state.json").read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_state(self, st: dict) -> None:
        (self.dir / "setup-state.json").write_text(json.dumps(st, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- steps
    def install(self) -> dict:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log(f"MakeAI Setup {VERSION} -> {self.dir}")
        st = self.state()
        gpu = detect_gpu()
        flavor = "cuda" if gpu else "cpu"
        self.log(f"GPU: {gpu or 'none'} -> PyTorch {flavor}")

        # 1. Python
        if not (self.py_dir / "python.exe").exists() or st.get("python") != PY_VER:
            tmp = self.dir / "python.zip"
            self.download(PY_URL, tmp, f"Python {PY_VER}", 0.0, 0.06)
            digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if digest != PY_SHA256:
                raise RuntimeError(f"Python download is corrupted (sha256 {digest})")
            shutil.rmtree(self.py_dir, ignore_errors=True)
            with zipfile.ZipFile(tmp) as z:
                z.extractall(self.py_dir)
            tmp.unlink()
            pth = next(self.py_dir.glob("python*._pth"))
            pth.write_text(pth.read_text().replace("#import site", "import site"), encoding="utf-8")
            gp = self.download(GET_PIP, self.dir / "get-pip.py", "pip", 0.06, 0.08)
            self.run([self.python, str(gp), "--no-warn-script-location"], "Installing pip…", 0.08, 0.12)
            gp.unlink()
            st = {"python": PY_VER}
            self.save_state(st)
        # keep MakeAI's Python isolated from packages the user installed for their own Python
        (self.py_dir / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
        # the embeddable Python ignores PYTHONPATH and the working directory (._pth mode), so the app folder
        # must be listed in the ._pth file - otherwise `python -m makeai.train.worker` cannot find MakeAI
        pth = next(self.py_dir.glob("python*._pth"))
        lines = pth.read_text(encoding="utf-8").splitlines()
        if r"..\app" not in lines:
            pth.write_text("\n".join(lines + [r"..\app"]) + "\n", encoding="utf-8")
        self.progress(0.12)

        # 2. PyTorch (CUDA build when an NVIDIA GPU is present)
        if st.get("torch") != f"{TORCH}+{flavor}":
            self.run([self.python, "-m", "pip", "install", "--no-warn-script-location", TORCH, "--index-url",
                      TORCH_INDEX[flavor]], f"Installing PyTorch ({'GPU' if gpu else 'CPU'}), about "
                     f"{'2.5 GB' if gpu else '200 MB'}…", 0.12, 0.70)
            st["torch"] = f"{TORCH}+{flavor}"
            self.save_state(st)

        # 3. MakeAI itself
        self.say("Installing MakeAI…")
        new_app = self.dir / "app.new"
        shutil.rmtree(new_app, ignore_errors=True)
        with zipfile.ZipFile(resource("payload.zip")) as z:
            z.extractall(new_app)
        shutil.rmtree(self.app_dir, ignore_errors=True)
        new_app.rename(self.app_dir)
        self.progress(0.74)

        # 4. Python packages
        req = (self.app_dir / "requirements.txt").read_text(encoding="utf-8")
        req_hash = hashlib.sha256(req.encode()).hexdigest()[:16]
        if st.get("requirements") != req_hash:
            self.run([self.python, "-m", "pip", "install", "--no-warn-script-location", "-r",
                      str(self.app_dir / "requirements.txt")], "Installing MakeAI's packages…", 0.74, 0.92)
            # optional: run GGUF models with llama.cpp (prebuilt CPU wheel); MakeAI works without it
            self.run([self.python, "-m", "pip", "install", "--no-warn-script-location", "--only-binary=:all:",
                      "llama-cpp-python", "--extra-index-url", LLAMA_INDEX], "Installing llama.cpp (optional)…",
                     0.92, 0.95, check=False)
            st["requirements"] = req_hash
            self.save_state(st)
        self.progress(0.95)

        # 5. check that it really works
        self.say("Checking the installation…")
        out = subprocess.run([self.python, "-c", "import site, sys, torch, makeai, makeai.train.worker, makeai.server.app, fastapi, tokenizers; "
                              "assert not any('Roaming' in p for p in sys.path), sys.path; "
                              "print(makeai.__version__, torch.__version__, torch.cuda.is_available())"],
                             cwd=str(self.app_dir), capture_output=True, text=True, creationflags=NO_WINDOW, env=ENV)
        self.log("check: " + (out.stdout + out.stderr).strip())
        if out.returncode != 0:
            raise RuntimeError(f"MakeAI does not start: {out.stderr.strip()[-400:]}")
        version, torch_v, cuda = out.stdout.split()
        st.update({"makeai": version, "installed": time.strftime("%Y-%m-%d %H:%M")})
        self.save_state(st)
        self.write_uninstaller()
        if self.shortcuts:
            self.say("Creating shortcuts…")
            make_shortcuts(self.dir)
        self.progress(1.0)
        result = {"dir": str(self.dir), "makeai": version, "torch": torch_v, "cuda": cuda == "True", "gpu": gpu}
        self.log(f"done: {result}")
        if self.launch:
            subprocess.Popen([str(self.py_dir / "pythonw.exe"), str(self.app_dir / "run.py"), "--window"],
                             cwd=str(self.app_dir), creationflags=NO_WINDOW)
        return result

    def write_uninstaller(self) -> None:
        (self.dir / "Uninstall MakeAI.cmd").write_text(
            "@echo off\r\n"
            "echo Removing MakeAI program files (your AIs and data in %USERPROFILE%\\MakeAI are kept).\r\n"
            "taskkill /F /FI \"WINDOWTITLE eq MakeAI\" >nul 2>&1\r\n"
            f"del \"%USERPROFILE%\\Desktop\\MakeAI.lnk\" >nul 2>&1\r\n"
            f"del \"%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\MakeAI.lnk\" >nul 2>&1\r\n"
            f"cd /d \"%TEMP%\"\r\n"
            f"rmdir /s /q \"{self.dir}\"\r\n"
            "echo Done.\r\npause\r\n", encoding="utf-8")


def detect_gpu() -> str | None:
    smi = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    try:
        out = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=15, creationflags=NO_WINDOW)
        name = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
        return name
    except Exception:
        return None


def make_shortcuts(target: Path) -> None:
    py = target / "python" / "pythonw.exe"
    run = target / "app" / "run.py"
    icon = target / "app" / "makeai" / "assets" / "makeai.ico"
    ps = f"""
$sh = New-Object -ComObject WScript.Shell
foreach ($d in @([Environment]::GetFolderPath('Desktop'), (Join-Path ([Environment]::GetFolderPath('StartMenu')) 'Programs'))) {{
  $s = $sh.CreateShortcut((Join-Path $d 'MakeAI.lnk'))
  $s.TargetPath = '{py}'
  $s.Arguments = '"{run}" --window'
  $s.WorkingDirectory = '{target / "app"}'
  $s.IconLocation = '{icon},0'
  $s.Description = 'MakeAI by Convergent'
  $s.Save()
}}"""
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps], check=True,
                   capture_output=True, creationflags=NO_WINDOW)


# ====================================================================== window
def gui(args) -> None:
    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("MakeAI Setup")
    root.geometry("620x520")
    root.resizable(False, False)
    try:
        root.iconbitmap(str(resource("makeai.ico")))
    except Exception:
        pass
    bg, acc = "#ffffff", "#7c3aed"
    root.configure(bg=bg)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TProgressbar", troughcolor="#eeeaf5", background=acc, bordercolor=bg, lightcolor=acc, darkcolor=acc)
    tk.Label(root, text="MakeAI", font=("Segoe UI", 20, "bold"), bg=bg, fg="#16131d").pack(anchor="w", padx=28, pady=(24, 0))
    tk.Label(root, text=f"by Convergent · version {VERSION}", font=("Segoe UI", 10), bg=bg, fg="#635d72").pack(anchor="w", padx=28)
    tk.Label(root, text="License", font=("Segoe UI", 10, "bold"), bg=bg).pack(anchor="w", padx=28, pady=(16, 4))
    lic = tk.Text(root, height=10, wrap="word", font=("Segoe UI", 9), relief="solid", bd=1, fg="#3d3552")
    lic.insert("1.0", resource("LICENSE").read_text(encoding="utf-8"))
    lic.configure(state="disabled")
    lic.pack(fill="x", padx=28)
    agree = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="I accept the license", variable=agree, bg=bg, activebackground=bg,
                   command=lambda: btn.configure(state="normal" if agree.get() else "disabled")).pack(anchor="w", padx=24, pady=(6, 0))
    row = tk.Frame(root, bg=bg)
    row.pack(fill="x", padx=28, pady=(10, 0))
    tk.Label(row, text="Install to", bg=bg, font=("Segoe UI", 9)).pack(side="left")
    dir_var = tk.StringVar(value=str(Path(args.dir) if args.dir else default_dir()))
    tk.Entry(row, textvariable=dir_var, font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True, padx=8)
    tk.Button(row, text="…", command=lambda: dir_var.set(filedialog.askdirectory() or dir_var.get())).pack(side="left")
    status = tk.StringVar(value="MakeAI needs about 5 GB. PyTorch is downloaded during setup.")
    tk.Label(root, textvariable=status, bg=bg, fg="#635d72", font=("Segoe UI", 9), anchor="w", justify="left",
             wraplength=560).pack(fill="x", padx=28, pady=(14, 4))
    bar = ttk.Progressbar(root, maximum=1000, style="TProgressbar")
    bar.pack(fill="x", padx=28)
    btn = tk.Button(root, text="Install", state="disabled", bg=acc, fg="white", activebackground="#6d28d9",
                    activeforeground="white", relief="flat", font=("Segoe UI", 11, "bold"), padx=24, pady=6)
    btn.pack(anchor="e", padx=28, pady=18)
    q: queue.Queue = queue.Queue()

    def pump():
        try:
            while True:
                kind, val = q.get_nowait()
                if kind == "say":
                    status.set(val)
                elif kind == "p":
                    bar["value"] = int(val * 1000)
                elif kind == "done":
                    status.set(f"MakeAI is installed{' and uses your GPU' if val['cuda'] else ''}. It is starting now - "
                               f"you find it on your desktop and in the Start menu.")
                    btn.configure(text="Close", state="normal", command=root.destroy)
                elif kind == "error":
                    status.set(f"Setup failed: {val}")
                    btn.configure(text="Close", state="normal", command=root.destroy)
        except queue.Empty:
            pass
        root.after(100, pump)

    def start():
        btn.configure(state="disabled")
        inst = Installer(Path(dir_var.get()), shortcuts=True, launch=True,
                         say=lambda m: q.put(("say", m)), progress=lambda v: q.put(("p", v)))

        def work():
            try:
                q.put(("done", inst.install()))
            except Exception as e:
                inst.log(traceback.format_exc())
                q.put(("error", str(e)))
        threading.Thread(target=work, daemon=True).start()

    btn.configure(command=start)
    pump()
    root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description="MakeAI Setup")
    ap.add_argument("--silent", action="store_true")
    ap.add_argument("--accept-license", action="store_true")
    ap.add_argument("--dir")
    ap.add_argument("--no-shortcuts", action="store_true")
    ap.add_argument("--no-launch", action="store_true")
    args = ap.parse_args()
    if not args.silent:
        gui(args)
        return
    if not args.accept_license:
        print("Use --accept-license to accept the MakeAI license (see LICENSE).", file=sys.stderr)
        sys.exit(2)
    last = [0.0]

    def prog(v):
        if v - last[0] >= 0.05 or v >= 1:
            last[0] = v
            print(f"  {int(v * 100)}%", flush=True)
    inst = Installer(Path(args.dir) if args.dir else default_dir(), shortcuts=not args.no_shortcuts,
                     launch=not args.no_launch, say=lambda m: print(m, flush=True), progress=prog)
    try:
        print(json.dumps(inst.install()))
    except Exception as e:
        inst.log(traceback.format_exc())
        print(f"Setup failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
