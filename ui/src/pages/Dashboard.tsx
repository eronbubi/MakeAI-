import React from "react";
import { fmt, stateLabel } from "../api";
import { AiIcon, Bar, Icon, Spark } from "../components/ui";
import { useApp, useFetch } from "../store";

export default function Dashboard() {
  const { telemetry, history, open, activeRuns, settings } = useApp();
  const hw = useFetch<any>("/api/hardware");
  const models = useFetch<any[]>("/api/models");
  const runs = useFetch<any[]>("/api/runs");
  const ds = useFetch<any[]>("/api/datasets");
  const g = telemetry?.gpus?.[0];
  const gs = hw.data?.gpus?.[0];
  const go = (view: string, id?: string, title?: string) => open({ view, id, title });
  return (
    <div className="page">
      <div className="head">
        <h1 className="title">Dashboard</h1>
        <span className="sub">Hello {settings.profile.name}</span>
        <div className="actions">
          <button className="btn" onClick={() => go("import", undefined, "Import")}><Icon n="import" s={14} />Import model</button>
          <button className="btn primary" onClick={() => go("create", undefined, "Create AI")}><Icon n="plus" s={14} />Create AI</button>
        </div>
      </div>

      {activeRuns.map((r) => (
        <div key={r.run_id} className="card" style={{ marginBottom: 12, borderColor: "var(--acc-dim)" }}>
          <div className="row">
            <span className="dot run" /><b>Training {r.model_name}</b><span className="pill acc">{stateLabel[r.state] || r.state}</span>
            <span className="dim mono small">{fmt.int(r.step)} / {fmt.int(r.total_steps)} steps · loss {fmt.num(r.last_loss, 3)}</span>
            <span className="sp" />
            <button className="btn primary sm" onClick={() => go("run", r.run_id, `Run · ${r.model_name}`)}>Open live monitor</button>
          </div>
          <div style={{ marginTop: 10 }}><Bar v={r.total_steps ? (100 * (r.step || 0)) / r.total_steps : 0} /></div>
        </div>
      ))}

      <div className="grid g4">
        <Tile title="GPU" value={g ? `${g.util_pct ?? "–"}%` : "–"} sub={gs?.name || "no GPU"} spark={history.map((h) => h.gpus?.[0]?.util_pct ?? null)} max={100} />
        <Tile title="VRAM" value={g ? fmt.mb(g.vram_used_mb) : "–"} sub={g ? `of ${fmt.mb(g.vram_total_mb)}` : ""} spark={history.map((h) => h.gpus?.[0]?.vram_used_mb ?? null)} max={g?.vram_total_mb} color="var(--violet)" />
        <Tile title="CPU" value={telemetry ? `${Math.round(telemetry.cpu_pct)}%` : "–"} sub={hw.data?.cpu?.name} spark={history.map((h) => h.cpu_pct)} max={100} color="var(--blue)" />
        <Tile title="RAM" value={telemetry ? fmt.mb(telemetry.ram_used_mb) : "–"} sub={telemetry ? `of ${fmt.mb(telemetry.ram_total_mb)} · ${Math.round(telemetry.ram_pct)}%` : ""} spark={history.map((h) => h.ram_pct)} max={100} color="var(--warn)" />
      </div>

      <div className="grid g2" style={{ marginTop: 12 }}>
        <div className="card">
          <h3><Icon n="bot" s={15} />My AIs<span className="r"><button className="btn ghost sm" onClick={() => go("models", undefined, "My AIs")}>All</button></span></h3>
          {models.data?.length ? (
            <table className="t"><tbody>
              {models.data.slice(0, 6).map((m) => (
                <tr key={m.uid} style={{ cursor: "pointer" }} onClick={() => go("ai", m.uid, m.name)}>
                  <td style={{ width: 40 }}><div className="ai-card"><AiIcon icon={m.icon} name={m.name} /></div></td>
                  <td><b>{m.display_name || m.name}</b> <span className="dim">{m.version}</span><div className="dim small">by {m.creator.name} @{m.creator.username}</div></td>
                  <td className="num">{fmt.params(m.param_count)}</td>
                  <td><span className={`pill ${m.runnable ? "good" : ""}`}>{m.status}</span></td>
                </tr>
              ))}
            </tbody></table>
          ) : <div className="empty">No AIs yet. <button className="btn sm primary" onClick={() => go("create", undefined, "Create AI")}>Create your first AI</button></div>}
        </div>
        <div className="card">
          <h3><Icon n="train" s={15} />Recent training<span className="r"><button className="btn ghost sm" onClick={() => go("training", undefined, "Training")}>All</button></span></h3>
          {runs.data?.length ? (
            <table className="t"><tbody>
              {runs.data.slice(0, 6).map((r) => (
                <tr key={r.run_id} style={{ cursor: "pointer" }} onClick={() => go("run", r.run_id, `Run · ${r.model_name}`)}>
                  <td><b>{r.model_name}</b><div className="dim small">{r.method} · {fmt.date(r.created)}</div></td>
                  <td className="num">{fmt.int(r.step)}/{fmt.int(r.total_steps)}</td>
                  <td className="num">{r.best_val_loss != null ? `val ${fmt.num(r.best_val_loss, 3)}` : r.last_loss != null ? `loss ${fmt.num(r.last_loss, 3)}` : ""}</td>
                  <td><StatePill s={r.state} /></td>
                </tr>
              ))}
            </tbody></table>
          ) : <div className="empty">No training runs yet.</div>}
        </div>
      </div>

      <div className="grid g3" style={{ marginTop: 12 }}>
        <div className="card"><h3><Icon n="data" s={15} />Datasets</h3>
          <div className="stat"><span className="v">{ds.data?.length ?? "–"}</span></div>
          <div className="dim small">{fmt.int(ds.data?.reduce((a, d) => a + (d.stats?.samples || 0), 0))} samples total</div>
          <button className="btn sm" style={{ marginTop: 10 }} onClick={() => go("datasets", undefined, "Datasets")}>Manage</button>
        </div>
        <div className="card"><h3><Icon n="cpu" s={15} />Compute</h3>
          <dl className="kv">
            <dt>Device</dt><dd>{hw.data?.best_device?.name || "–"}</dd>
            <dt>Precision</dt><dd>{hw.data?.best_device ? ["bf16", "fp16", "tf32"].filter((k) => hw.data.best_device[k]).join(", ").toUpperCase() || "FP32" : "–"}</dd>
            <dt>Benchmark</dt><dd>{hw.data?.benchmark ? `${hw.data.benchmark.tflops.bf16 ?? hw.data.benchmark.tflops.fp16} TFLOPS measured` : "not measured yet"}</dd>
          </dl>
        </div>
        <div className="card"><h3><Icon n="play" s={15} />Playground</h3>
          <div className="dim small">Chat with any trained or imported AI locally - no internet, no cloud API.</div>
          <button className="btn sm" style={{ marginTop: 10 }} onClick={() => go("playground", undefined, "Playground")}>Open Playground</button>
        </div>
      </div>
    </div>
  );
}

function Tile({ title, value, sub, spark, max, color }: { title: string; value: string; sub?: string; spark: (number | null)[]; max?: number; color?: string }) {
  return (
    <div className="card">
      <div className="stat"><span className="k">{title}</span><span className="v">{value}</span></div>
      <div className="dim small" style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{sub}</div>
      <div style={{ marginTop: 8 }}><Spark values={spark.slice(-120)} max={max} color={color} /></div>
    </div>
  );
}

export function StatePill({ s }: { s?: string }) {
  const cls = s === "running" || s === "starting" ? "acc" : s === "completed" ? "good" : s === "paused" ? "warn" : ["failed", "crashed", "interrupted", "terminated"].includes(s || "") ? "bad" : "";
  return <span className={`pill ${cls}`}>{stateLabel[s || ""] || s || "–"}</span>;
}
