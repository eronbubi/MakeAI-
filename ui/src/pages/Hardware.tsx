import React, { useState } from "react";
import { api, fmt } from "../api";
import { Bar, Icon, LineChart, Spinner } from "../components/ui";
import { useApp, useFetch } from "../store";

export default function Hardware() {
  const { telemetry, history, track, toast } = useApp();
  const [refresh, setRefresh] = useState(0);
  const hw = useFetch<any>(`/api/hardware${refresh ? "?refresh=true" : ""}`, [refresh]);
  const [benching, setBenching] = useState(false);
  const h = hw.data;
  const bench = async () => {
    setBenching(true);
    try { await track(await api.post("/api/hardware/benchmark"), "Hardware benchmark"); hw.reload(); toast("Benchmark finished"); }
    catch (e: any) { toast(e.message, true); } finally { setBenching(false); }
  };
  const t0 = history[0]?.t || 0;
  const xs = (f: (s: any) => number | null) => history.map((s) => [s.t - t0, f(s)] as [number, number]);
  return (
    <div className="page">
      <div className="head">
        <h1 className="title">Hardware</h1>
        <span className="sub">Live values refresh every second; only metrics your OS and driver report are shown.</span>
        <div className="actions">
          <button className="btn" onClick={() => setRefresh((r) => r + 1)} disabled={hw.loading}>{hw.loading ? <Spinner /> : <Icon n="search" s={14} />}Rescan</button>
          <button className="btn primary" onClick={bench} disabled={benching}>{benching ? <Spinner /> : <Icon n="bolt" s={14} />}Measure throughput</button>
        </div>
      </div>
      {!h && <Spinner />}
      {h && (
        <>
          <div className="grid g2">
            {h.gpus.map((g: any) => {
              const live = telemetry?.gpus?.[g.index];
              return (
                <div className="card" key={g.index}>
                  <h3><Icon n="cpu" s={15} />GPU {g.index} · {g.name}<span className="r">{g.vendor}</span></h3>
                  <div className="grid g3" style={{ marginBottom: 12 }}>
                    <Meter label="Usage" v={live?.util_pct} unit="%" max={100} />
                    <Meter label="VRAM" v={live?.vram_used_mb} max={live?.vram_total_mb} txt={live ? `${fmt.mb(live.vram_used_mb)} / ${fmt.mb(live.vram_total_mb)}` : "–"} />
                    <Meter label="Temperature" v={live?.temp_c} unit="°C" max={g.slowdown_temp_c || 95} />
                    <Meter label="Power" v={live?.power_w} unit=" W" max={live?.power_limit_w || g.power_limit_w} />
                    <Meter label="Clock" v={live?.clock_mhz} unit=" MHz" max={g.max_graphics_clock_mhz} />
                    <Meter label="Memory clock" v={live?.mem_clock_mhz} unit=" MHz" max={g.max_memory_clock_mhz} />
                  </div>
                  <dl className="kv">
                    <dt>Compute capability</dt><dd>{g.compute_capability || "–"} {g.sm_count ? `· ${g.sm_count} SMs` : ""}</dd>
                    <dt>Precision</dt><dd>{[g.fp16 && "FP16", g.bf16 && "BF16", g.tf32 && "TF32", g.fp8 && "FP8"].filter(Boolean).join(" · ") || "FP32"}</dd>
                    <dt>Tensor Cores</dt><dd>{g.tensor_cores ? "yes" : "no"}{g.int8_tensor_cores ? " (INT8 capable)" : ""}</dd>
                    <dt>Power limit</dt><dd>{g.power_limit_w ? `${g.power_limit_w} W` : "–"}</dd>
                    <dt>Slowdown / shutdown</dt><dd>{g.slowdown_temp_c ?? "–"} °C / {g.shutdown_temp_c ?? "–"} °C</dd>
                    <dt>PCIe</dt><dd>{g.pcie_gen ? `Gen ${g.pcie_gen} x${g.pcie_width}` : "–"}</dd>
                    <dt>Fan</dt><dd>{live?.fan_pct != null ? `${live.fan_pct}%` : "–"}</dd>
                    <dt>Throttle reasons</dt><dd>{live?.throttle?.length ? live.throttle.join(", ") : "none"}</dd>
                  </dl>
                </div>
              );
            })}
            {!h.gpus.length && <div className="card"><h3>GPU</h3><div className="note warn">No CUDA/ROCm GPU detected. MakeAI will train on the CPU, which is much slower.</div></div>}
            <div className="card">
              <h3><Icon n="cpu" s={15} />CPU · {h.cpu.name}</h3>
              <div className="grid g3" style={{ marginBottom: 12 }}>
                <Meter label="Usage" v={telemetry?.cpu_pct} unit="%" max={100} />
                <Meter label="Clock" v={telemetry?.cpu_mhz} unit=" MHz" max={h.cpu.max_mhz} />
                <Meter label="Temperature" v={telemetry?.cpu_temp_c} unit="°C" max={100} na={h.unavailable?.cpu_temp} />
              </div>
              <div className="lbl" style={{ marginBottom: 4 }}>Per thread</div>
              <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.min(16, telemetry?.cpu_per_core?.length || 8)}, 1fr)`, gap: 3, height: 44, alignItems: "end" }}>
                {telemetry?.cpu_per_core?.map((v, i) => <div key={i} title={`thread ${i}: ${v}%`} style={{ height: `${Math.max(3, v)}%`, background: "var(--blue)", borderRadius: 2, opacity: 0.85 }} />)}
              </div>
              <dl className="kv" style={{ marginTop: 12 }}>
                <dt>Cores / threads</dt><dd>{h.cpu.cores} / {h.cpu.threads}</dd>
                <dt>RAM</dt><dd>{telemetry ? `${fmt.mb(telemetry.ram_used_mb)} / ${fmt.mb(telemetry.ram_total_mb)} (${telemetry.ram_pct}%)` : "–"}</dd>
                <dt>OS</dt><dd>{h.os.system} {h.os.release} ({h.os.version}) {h.os.machine}</dd>
                <dt>Python</dt><dd>{h.os.python}</dd>
              </dl>
            </div>
          </div>

          <div className="grid g2" style={{ marginTop: 12 }}>
            <div className="card"><h3>GPU usage &amp; memory<span className="r">last {Math.round((history.at(-1)?.t || 0) - t0)} s</span></h3>
              <LineChart height={160} yMin={0} yMax={100} xFmt={(x) => `${Math.round(x)}s`} series={[
                { name: "GPU %", color: "var(--acc)", points: xs((s) => s.gpus?.[0]?.util_pct ?? null) },
                { name: "VRAM %", color: "var(--violet)", points: xs((s) => s.gpus?.[0] ? (100 * s.gpus[0].vram_used_mb) / s.gpus[0].vram_total_mb : null) },
                { name: "CPU %", color: "var(--blue)", points: xs((s) => s.cpu_pct) },
                { name: "RAM %", color: "var(--warn)", points: xs((s) => s.ram_pct), dashed: true },
              ]} />
            </div>
            <div className="card"><h3>Thermals &amp; power</h3>
              <LineChart height={160} xFmt={(x) => `${Math.round(x)}s`} series={[
                { name: "GPU °C", color: "var(--orange)", points: xs((s) => s.gpus?.[0]?.temp_c ?? null) },
                { name: "Power W", color: "var(--warn)", points: xs((s) => s.gpus?.[0]?.power_w ?? null) },
                ...(telemetry?.cpu_temp_c != null ? [{ name: "CPU °C", color: "var(--blue)", points: xs((s) => s.cpu_temp_c) }] : []),
              ]} />
            </div>
          </div>

          <div className="grid g3" style={{ marginTop: 12 }}>
            <div className="card"><h3>Software</h3>
              <dl className="kv">
                <dt>PyTorch</dt><dd>{h.torch.version || "not installed"}</dd>
                <dt>CUDA</dt><dd>{h.torch.cuda || "–"}</dd>
                <dt>ROCm</dt><dd>{h.torch.rocm || "–"}</dd>
                <dt>cuDNN</dt><dd>{h.torch.cudnn || "–"}</dd>
                <dt>Driver</dt><dd>{h.driver_version || "–"} {h.driver_cuda_version ? `(CUDA ${h.driver_cuda_version})` : ""}</dd>
                <dt>Multi-GPU</dt><dd>{h.multi_gpu ? `yes (${h.torch.device_count})` : "no"}</dd>
              </dl>
            </div>
            <div className="card"><h3>Storage</h3>
              {h.storage.map((d: any) => (
                <div key={d.mount} style={{ marginBottom: 8 }}>
                  <div className="row small"><b>{d.mount}</b><span className="dim">{d.fs}</span><span className="sp" /><span className="mono">{d.free_gb} GB free of {d.total_gb}</span></div>
                  <Bar v={d.pct} color={d.pct > 90 ? "var(--bad)" : undefined} />
                </div>
              ))}
              <div className="dim small">MakeAI data: <span className="mono">{h.makeai_home.path}</span></div>
            </div>
            <div className="card"><h3>Measured throughput</h3>
              {h.benchmark ? (
                <dl className="kv">
                  {Object.entries(h.benchmark.tflops).map(([k, v]) => <React.Fragment key={k}><dt>{k.toUpperCase()} matmul</dt><dd>{String(v)} TFLOPS</dd></React.Fragment>)}
                  <dt>VRAM bandwidth</dt><dd>{h.benchmark.mem_bandwidth_gbs ?? "–"} GB/s</dd>
                  <dt>Host → GPU</dt><dd>{h.benchmark.h2d_bandwidth_gbs ?? "–"} GB/s</dd>
                  <dt>Measured</dt><dd>{new Date(h.benchmark.at * 1000).toLocaleString()}</dd>
                </dl>
              ) : <div className="dim small">Not measured yet. Measuring takes a few seconds and makes training-time estimates use this GPU's real speed (including load from other apps).</div>}
            </div>
          </div>
          {!!Object.keys(h.unavailable || {}).length && (
            <div className="card" style={{ marginTop: 12 }}><h3>Not available on this system</h3>
              {Object.entries(h.unavailable).map(([k, v]) => <div key={k} className="small"><span className="mono">{k}</span> <span className="dim">- {String(v)}</span></div>)}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Meter({ label, v, unit = "", max, txt, na }: { label: string; v?: number | null; unit?: string; max?: number | null; txt?: string; na?: string }) {
  const pct = v != null && max ? (100 * v) / max : 0;
  return (
    <div className="stat" title={v == null && na ? na : undefined}>
      <span className="k">{label}</span>
      <span className="v" style={{ fontSize: 16 }}>{v == null ? <span className="faint">n/a</span> : txt || `${Math.round(v)}${unit}`}</span>
      {v != null && max ? <Bar v={pct} color={pct > 90 ? "var(--warn)" : undefined} /> : null}
    </div>
  );
}
