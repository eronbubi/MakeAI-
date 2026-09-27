"""Hardware scanner and live telemetry.

Only values the OS/driver actually reports are returned. Anything that cannot
be read is ``None`` and listed in ``unavailable`` with the reason - the UI hides
those metrics instead of inventing them.
"""
from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from collections import deque
from typing import Any

import psutil

try:
    import pynvml  # nvidia-ml-py
except Exception:  # pragma: no cover
    pynvml = None

_nvml_ok: bool | None = None
_nvml_lock = threading.Lock()


def _nvml() -> bool:
    global _nvml_ok
    with _nvml_lock:
        if _nvml_ok is None:
            try:
                pynvml.nvmlInit()
                _nvml_ok = True
            except Exception:
                _nvml_ok = False
    return _nvml_ok


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


def _s(v):
    return v.decode() if isinstance(v, bytes) else v


# ------------------------------------------------------------------- CPU name
def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if platform.system() == "Linux":
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        out = _safe(subprocess.check_output, ["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.decode().strip()
    return platform.processor() or "Unknown CPU"


class _CpuTemp:
    """CPU temperature where the OS exposes it (Linux hwmon, Windows ACPI/LibreHardwareMonitor WMI)."""

    def __init__(self):
        self.source: str | None = None
        self.reason = "not probed"
        self._probe()

    def _probe(self):
        if hasattr(psutil, "sensors_temperatures"):
            t = _safe(psutil.sensors_temperatures) or {}
            for key in ("k10temp", "coretemp", "zenpower", "cpu_thermal", "acpitz"):
                if t.get(key):
                    self.source, self.key = "psutil", key
                    return
        if platform.system() == "Windows":
            for ns, q, src in (
                ("root/LibreHardwareMonitor", "SELECT Value FROM Sensor WHERE SensorType='Temperature' AND Name LIKE '%CPU%'", "lhm"),
                ("root/OpenHardwareMonitor", "SELECT Value FROM Sensor WHERE SensorType='Temperature' AND Name LIKE '%CPU%'", "ohm"),
                ("root/wmi", "SELECT CurrentTemperature FROM MSAcpi_ThermalZoneTemperature", "acpi"),
            ):
                v = self._wmi(ns, q)
                if v is not None:
                    self.source, self.ns, self.q = src, ns, q
                    return
            self.reason = ("Windows exposes CPU temperature only through ACPI WMI (requires administrator) or "
                           "a running LibreHardwareMonitor/OpenHardwareMonitor")
            return
        self.reason = "no temperature sensor exposed by the OS"

    @staticmethod
    def _wmi(ns: str, q: str) -> float | None:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance -Namespace {ns} -Query \"{q}\" -ErrorAction Stop | Select-Object -First 1 | "
                 f"ForEach-Object {{ $_.PSObject.Properties | Where-Object {{ $_.Name -in 'Value','CurrentTemperature' }} "
                 f"| Select-Object -ExpandProperty Value }})"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            v = float(out.stdout.strip().splitlines()[0])
            if "MSAcpi" in q:
                v = v / 10.0 - 273.15
            return round(v, 1) if 0 < v < 130 else None
        except Exception:
            return None

    def read(self) -> float | None:
        if self.source == "psutil":
            t = _safe(psutil.sensors_temperatures) or {}
            vals = [s.current for s in t.get(self.key, []) if s.current]
            return round(max(vals), 1) if vals else None
        if self.source in ("lhm", "ohm", "acpi"):
            return self._wmi(self.ns, self.q)
        return None


# ---------------------------------------------------------------- GPU (NVML)
def _gpu_static(i: int) -> dict[str, Any]:
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    d: dict[str, Any] = {
        "index": i,
        "vendor": "NVIDIA",
        "name": _s(pynvml.nvmlDeviceGetName(h)),
        "uuid": _s(_safe(pynvml.nvmlDeviceGetUUID, h)),
        "vram_total_mb": round(mem.total / 2**20),
        "power_limit_w": _div(_safe(pynvml.nvmlDeviceGetEnforcedPowerLimit, h), 1000),
        "max_graphics_clock_mhz": _safe(pynvml.nvmlDeviceGetMaxClockInfo, h, pynvml.NVML_CLOCK_GRAPHICS),
        "max_memory_clock_mhz": _safe(pynvml.nvmlDeviceGetMaxClockInfo, h, pynvml.NVML_CLOCK_MEM),
        "slowdown_temp_c": _safe(pynvml.nvmlDeviceGetTemperatureThreshold, h, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN),
        "shutdown_temp_c": _safe(pynvml.nvmlDeviceGetTemperatureThreshold, h, pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN),
        "pcie_gen": _safe(pynvml.nvmlDeviceGetMaxPcieLinkGeneration, h),
        "pcie_width": _safe(pynvml.nvmlDeviceGetMaxPcieLinkWidth, h),
    }
    cc = _safe(pynvml.nvmlDeviceGetCudaComputeCapability, h)
    if cc:
        d["compute_capability"] = f"{cc[0]}.{cc[1]}"
    return d


def _div(v, n):
    return None if v is None else round(v / n, 1)


_THROTTLE_BITS = {
    "gpu_idle": 0x1, "app_clocks": 0x2, "sw_power_cap": 0x4, "hw_slowdown": 0x8, "sync_boost": 0x10,
    "sw_thermal": 0x20, "hw_thermal": 0x40, "hw_power_brake": 0x80, "display_clocks": 0x100,
}


def _gpu_live(i: int) -> dict[str, Any]:
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    util = _safe(pynvml.nvmlDeviceGetUtilizationRates, h)
    reasons = _safe(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons, h)
    return {
        "index": i,
        "util_pct": util.gpu if util else None,
        "mem_util_pct": util.memory if util else None,
        "vram_used_mb": round(mem.used / 2**20),
        "vram_total_mb": round(mem.total / 2**20),
        "temp_c": _safe(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU),
        "power_w": _div(_safe(pynvml.nvmlDeviceGetPowerUsage, h), 1000),
        "power_limit_w": _div(_safe(pynvml.nvmlDeviceGetEnforcedPowerLimit, h), 1000),
        "clock_mhz": _safe(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_GRAPHICS),
        "mem_clock_mhz": _safe(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_MEM),
        "fan_pct": _safe(pynvml.nvmlDeviceGetFanSpeed, h),
        "pstate": _safe(pynvml.nvmlDeviceGetPerformanceState, h),
        "throttle": [k for k, bit in _THROTTLE_BITS.items() if reasons is not None and reasons & bit and k != "gpu_idle"],
    }


def _other_display_adapters() -> list[dict[str, Any]]:
    """Non-NVIDIA adapters (e.g. AMD/Intel iGPUs) so the user sees every device."""
    if platform.system() != "Windows":
        return []
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.AdapterRAM }"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        res = []
        for line in out.stdout.splitlines():
            if "|" not in line:
                continue
            name, ram = line.rsplit("|", 1)
            if "nvidia" in name.lower() or not name.strip():
                continue
            res.append({"name": name.strip(), "adapter_ram_mb": round(int(ram) / 2**20) if ram.strip().isdigit() else None,
                        "usable_for_training": False,
                        "note": "no CUDA/ROCm runtime on this device in this build"})
        return res
    except Exception:
        return []


# ------------------------------------------------------------------ the scan
def scan() -> dict[str, Any]:
    """Full static + live scan of the machine."""
    unavailable: dict[str, str] = {}
    info: dict[str, Any] = {"scanned_at": time.time()}

    # OS
    info["os"] = {"system": platform.system(), "release": platform.release(), "version": platform.version(),
                  "machine": platform.machine(), "python": platform.python_version()}

    # Torch / accelerators
    torch_info: dict[str, Any] = {"available": False}
    try:
        import torch
        torch_info = {"available": True, "version": torch.__version__, "cuda": torch.version.cuda,
                      "rocm": getattr(torch.version, "hip", None), "cuda_available": torch.cuda.is_available(),
                      "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                      "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
                      "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0}
    except Exception as e:  # pragma: no cover
        unavailable["torch"] = f"PyTorch not importable: {e}"
    info["torch"] = torch_info

    gpus: list[dict[str, Any]] = []
    if pynvml and _nvml():
        info["driver_version"] = _s(_safe(pynvml.nvmlSystemGetDriverVersion))
        v = _safe(pynvml.nvmlSystemGetCudaDriverVersion)
        info["driver_cuda_version"] = f"{v // 1000}.{(v % 1000) // 10}" if v else None
        for i in range(pynvml.nvmlDeviceGetCount()):
            g = _gpu_static(i)
            g.update({k: v for k, v in _gpu_live(i).items() if k not in g})
            gpus.append(g)
    else:
        unavailable["nvml"] = "NVIDIA management library not available (no NVIDIA driver)"

    if torch_info.get("cuda_available"):
        import torch
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            cc = (p.major, p.minor)
            extra = {"compute_capability": f"{p.major}.{p.minor}",
                     "sm_count": p.multi_processor_count,
                     "fp16": cc >= (5, 3),
                     "bf16": bool(torch.cuda.is_bf16_supported()) if i == 0 else cc >= (8, 0),
                     "tf32": cc >= (8, 0),
                     "tensor_cores": cc >= (7, 0),
                     "fp8": cc >= (8, 9),
                     "int8_tensor_cores": cc >= (7, 5)}
            if i < len(gpus):
                gpus[i].update(extra)
            else:
                gpus.append({"index": i, "vendor": "AMD" if torch_info.get("rocm") else "NVIDIA", "name": p.name,
                             "vram_total_mb": round(p.total_memory / 2**20), **extra})
    info["gpus"] = gpus
    info["other_adapters"] = _other_display_adapters()
    info["multi_gpu"] = len([g for g in gpus if g.get("compute_capability")]) > 1

    # CPU
    freq = _safe(psutil.cpu_freq)
    info["cpu"] = {"name": cpu_name(), "cores": psutil.cpu_count(logical=False), "threads": psutil.cpu_count(),
                   "max_mhz": round(freq.max) if freq and freq.max else None,
                   "util_pct": psutil.cpu_percent(interval=0.2)}
    temp = cpu_temp().read()
    info["cpu"]["temp_c"] = temp
    if temp is None:
        unavailable["cpu_temp"] = cpu_temp().reason

    vm = psutil.virtual_memory()
    info["ram"] = {"total_mb": round(vm.total / 2**20), "used_mb": round((vm.total - vm.available) / 2**20),
                   "available_mb": round(vm.available / 2**20), "pct": vm.percent}

    disks = []
    for part in _safe(psutil.disk_partitions) or []:
        if "cdrom" in part.opts or not part.fstype:
            continue
        u = _safe(psutil.disk_usage, part.mountpoint)
        if u:
            disks.append({"mount": part.mountpoint, "fs": part.fstype, "total_gb": round(u.total / 2**30, 1),
                          "free_gb": round(u.free / 2**30, 1), "pct": u.percent})
    info["storage"] = disks
    from .. import store
    hu = _safe(psutil.disk_usage, str(store.home()))
    info["makeai_home"] = {"path": str(store.home()), "free_gb": round(hu.free / 2**30, 1) if hu else None}
    info["unavailable"] = unavailable
    info["best_device"] = best_device(info)
    return info


def best_device(info: dict[str, Any]) -> dict[str, Any]:
    g = [x for x in info.get("gpus", []) if x.get("compute_capability")]
    if g and info["torch"].get("cuda_available"):
        top = max(g, key=lambda x: x.get("vram_total_mb", 0))
        return {"type": "cuda", "index": top["index"], "name": top["name"], "vram_mb": top["vram_total_mb"],
                "bf16": top.get("bf16", False), "fp16": top.get("fp16", False), "tf32": top.get("tf32", False),
                "tensor_cores": top.get("tensor_cores", False), "compute_capability": top["compute_capability"]}
    if info["torch"].get("mps"):
        return {"type": "mps", "name": "Apple GPU", "vram_mb": info["ram"]["total_mb"], "bf16": False, "fp16": True}
    return {"type": "cpu", "name": info["cpu"]["name"], "vram_mb": 0, "bf16": False, "fp16": False}


_cpu_temp: _CpuTemp | None = None


def cpu_temp() -> _CpuTemp:
    global _cpu_temp
    if _cpu_temp is None:
        _cpu_temp = _CpuTemp()
    return _cpu_temp


# ---------------------------------------------------------------- telemetry
class Telemetry:
    """Background sampler: GPU/CPU/RAM every ``interval`` seconds, with history."""

    def __init__(self, interval: float = 1.0, history: int = 900):
        self.interval = interval
        self.history: deque[dict[str, Any]] = deque(maxlen=history)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._temp_every = 5
        self._last_cpu_temp: float | None = None
        self._n = 0
        self.proc_watch: dict[int, str] = {}   # pid -> run id, for per-process RAM/CPU

    def start(self):
        if self._thread is None:
            psutil.cpu_percent(None)
            self._thread = threading.Thread(target=self._loop, name="telemetry", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                s = self.sample()
                with self._lock:
                    self.history.append(s)
            except Exception:
                pass
            self._stop.wait(max(0.05, self.interval - (time.time() - t0)))

    def sample(self) -> dict[str, Any]:
        s: dict[str, Any] = {"t": time.time()}
        if pynvml and _nvml():
            s["gpus"] = [_gpu_live(i) for i in range(pynvml.nvmlDeviceGetCount())]
        else:
            s["gpus"] = []
        s["cpu_pct"] = psutil.cpu_percent(None)
        s["cpu_per_core"] = psutil.cpu_percent(None, percpu=True)
        if self._n % self._temp_every == 0 and cpu_temp().source:
            self._last_cpu_temp = cpu_temp().read()
        s["cpu_temp_c"] = self._last_cpu_temp if cpu_temp().source else None
        freq = _safe(psutil.cpu_freq)
        s["cpu_mhz"] = round(freq.current) if freq and freq.current else None
        vm = psutil.virtual_memory()
        s["ram_used_mb"] = round((vm.total - vm.available) / 2**20)
        s["ram_total_mb"] = round(vm.total / 2**20)
        s["ram_pct"] = vm.percent
        procs = {}
        for pid, run in list(self.proc_watch.items()):
            try:
                p = psutil.Process(pid)
                with p.oneshot():
                    procs[run] = {"pid": pid, "rss_mb": round(p.memory_info().rss / 2**20),
                                  "cpu_pct": p.cpu_percent(None), "threads": p.num_threads()}
            except psutil.NoSuchProcess:
                self.proc_watch.pop(pid, None)
        s["procs"] = procs
        self._n += 1
        return s

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return self.history[-1] if self.history else None

    def since(self, t: float) -> list[dict[str, Any]]:
        with self._lock:
            return [h for h in self.history if h["t"] > t]

    def window(self, seconds: float) -> list[dict[str, Any]]:
        return self.since(time.time() - seconds)
