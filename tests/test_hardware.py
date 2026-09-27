import shutil
import subprocess

import pytest

from makeai.hardware.scanner import Telemetry, scan

nvsmi = shutil.which("nvidia-smi")


@pytest.mark.skipif(not nvsmi, reason="no NVIDIA driver")
def test_scanner_reports_correct_gpu_and_vram(home):
    out = subprocess.check_output([nvsmi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True)
    name, total = [x.strip() for x in out.splitlines()[0].split(",")]
    s = scan()
    g = s["gpus"][0]
    assert g["name"] == name
    assert g["vram_total_mb"] == int(total)
    assert g["compute_capability"] and "bf16" in g and "tensor_cores" in g


def test_scanner_cpu_ram_storage_os(home):
    import psutil
    s = scan()
    assert s["cpu"]["threads"] == psutil.cpu_count()
    assert s["cpu"]["cores"] == psutil.cpu_count(logical=False)
    assert abs(s["ram"]["total_mb"] - psutil.virtual_memory().total / 2**20) < 2
    assert s["storage"] and s["os"]["system"]
    # values the OS cannot report are None and explained, never invented
    if s["cpu"]["temp_c"] is None:
        assert "cpu_temp" in s["unavailable"]


def test_telemetry_updates_live(home):
    t = Telemetry(0.2)
    t.start()
    import time
    time.sleep(1.3)
    t.stop()
    h = t.window(10)
    assert len(h) >= 4
    assert h[-1]["t"] > h[0]["t"]
    assert 0 <= h[-1]["cpu_pct"] <= 100 and h[-1]["ram_total_mb"] > 0
