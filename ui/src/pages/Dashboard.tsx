import React from "react";
import { fmt, stateLabel } from "../api";
import { AiIcon, Bar, Icon } from "../components/ui";
import { useApp, useFetch } from "../store";

export default function Dashboard() {
  const { telemetry, open, activeRuns, settings } = useApp();
  const hw = useFetch<any>("/api/hardware");
  const models = useFetch<any[]>("/api/models");
  const runs = useFetch<any[]>("/api/runs", [activeRuns.length]);
  const g = telemetry?.gpus?.[0];
  const gs = hw.data?.gpus?.[0];
  const go = (view: string, id?: string, title?: string) => open({ view, id, title });
  const lastRun = runs.data?.[0];
  return (
    <div className="page" style={{ maxWidth: 1080 }}>
      <div className="head">
        <div>
          <h1 className="title">Hello {settings.profile.name}</h1>
          <div className="dim" style={{ marginTop: 4 }}>Create, train and run your own AI on this computer.</div>
        </div>
        <div className="actions"><button className="btn primary lg" onClick={() => go("create", undefined, "Create AI")}><Icon n="plus" s={16} />Create AI</button></div>
      </div>

      {activeRuns.map((r) => {
        const pct = r.total_steps ? (100 * (r.step || 0)) / r.total_steps : 0;
        return (
          <button key={r.run_id} className="card" style={{ width: "100%", textAlign: "left", cursor: "pointer", marginBottom: 20 }}
            onClick={() => go("run", r.run_id, r.model_name)}>
            <div className="row" style={{ marginBottom: 12 }}>
              <span className="dot run" /><b style={{ fontSize: 16 }}>Training {r.model_name}</b>
              <span className="dim small">{stateLabel[r.state] || r.state}</span>
              <span className="sp" /><span className="mono" style={{ fontSize: 20, color: "var(--acc)" }}>{Math.round(pct)}%</span>
            </div>
            <Bar v={pct} />
            <div className="row small dim" style={{ marginTop: 10, gap: 18 }}>
              <span>Step {fmt.int(r.step)} / {fmt.int(r.total_steps)}</span>
              {r.last_loss != null && <span>Loss {fmt.num(r.last_loss, 3)}</span>}
              <span className="sp" /><span>Open live view →</span>
            </div>
          </button>
        );
      })}

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.6fr) minmax(0, 1fr)" }}>
        <div className="card">
          <h3>My AIs<span className="r"><button className="btn ghost sm" onClick={() => go("models", undefined, "My AIs")}>See all</button></span></h3>
          {models.data?.length ? models.data.slice(0, 5).map((m) => (
            <div key={m.uid} className="list-row ai-card" style={{ flexDirection: "row" }} onClick={() => go("ai", m.uid, m.name)}>
              <AiIcon icon={m.icon} name={m.name} />
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontWeight: 600 }}>{m.display_name || m.name}</div>
                <div className="dim small">{fmt.params(m.param_count)} · {m.status.replace("_", " ")}</div>
              </div>
              {m.runnable && <button className="btn sm" onClick={(e) => { e.stopPropagation(); go("playground", m.uid, `Chat · ${m.name}`); }}>Chat</button>}
            </div>
          )) : (
            <div className="empty">You have no AIs yet.<div style={{ marginTop: 12 }}><button className="btn primary" onClick={() => go("create", undefined, "Create AI")}>Create your first AI</button></div></div>
          )}
        </div>
        <div className="col" style={{ gap: 16 }}>
          <div className="card">
            <h3>Your computer</h3>
            <div className="dim small">{gs?.name || "No GPU found - training uses the CPU"}</div>
            {g && (
              <div className="grid g2" style={{ marginTop: 14, gap: 12 }}>
                <div className="stat"><span className="k">GPU</span><span className="v" style={{ fontSize: 20 }}>{g.util_pct ?? "–"}%</span></div>
                <div className="stat"><span className="k">Free VRAM</span><span className="v" style={{ fontSize: 20 }}>{fmt.mb(g.vram_total_mb - g.vram_used_mb)}</span></div>
              </div>
            )}
            <button className="btn ghost sm" style={{ marginTop: 10, paddingLeft: 0 }} onClick={() => go("hardware", undefined, "Hardware")}>Details →</button>
          </div>
          {lastRun && !activeRuns.length && (
            <div className="card" style={{ cursor: "pointer" }} onClick={() => go("run", lastRun.run_id, lastRun.model_name)}>
              <h3>Last training</h3>
              <div style={{ fontWeight: 600 }}>{lastRun.model_name}</div>
              <div className="dim small">{stateLabel[lastRun.state] || lastRun.state}{lastRun.best_val_loss != null ? ` · val loss ${fmt.num(lastRun.best_val_loss, 3)}` : ""}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function StatePill({ s }: { s?: string }) {
  const cls = s === "running" || s === "starting" ? "acc" : s === "completed" ? "good" : s === "paused" ? "warn" : ["failed", "crashed", "interrupted", "terminated"].includes(s || "") ? "bad" : "";
  return <span className={`pill ${cls}`}>{stateLabel[s || ""] || s || "–"}</span>;
}
