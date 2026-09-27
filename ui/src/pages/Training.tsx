import React from "react";
import { fmt } from "../api";
import { Icon } from "../components/ui";
import { useApp, useFetch } from "../store";
import { StatePill } from "./Dashboard";

export default function Training() {
  const { open, activeRuns } = useApp();
  const runs = useFetch<any[]>("/api/runs", [activeRuns.length]);
  return (
    <div className="page">
      <div className="head">
        <h1 className="title">Training</h1><span className="sub">One run at a time uses the GPU; every run has its own process.</span>
        <div className="actions"><button className="btn primary" onClick={() => open({ view: "create", title: "Create AI" })}><Icon n="plus" s={14} />New training</button></div>
      </div>
      {runs.data && !runs.data.length && <div className="empty">No training runs yet. Create an AI to start one.</div>}
      {!!runs.data?.length && (
        <div className="card" style={{ padding: 0 }}>
          <table className="t"><thead><tr><th>AI</th><th>Method</th><th>State</th><th className="num">Step</th><th className="num">Loss</th><th className="num">Best val</th><th>Started</th><th>Duration</th><th /></tr></thead><tbody>
            {runs.data.map((r) => (
              <tr key={r.run_id} style={{ cursor: "pointer" }} onClick={() => open({ view: "run", id: r.run_id, title: `Run · ${r.model_name}` })}>
                <td><b>{r.model_name}</b><div className="faint mono small">{r.run_id}</div></td>
                <td>{r.method}</td>
                <td><StatePill s={r.state} />{r.message && r.state !== "running" && <div className="faint small" style={{ maxWidth: 260 }}>{r.message}</div>}</td>
                <td className="num">{fmt.int(r.step)} / {fmt.int(r.total_steps)}</td>
                <td className="num">{fmt.num(r.last_loss, 3)}</td>
                <td className="num">{fmt.num(r.best_val_loss, 3)}</td>
                <td className="small">{r.started_at ? new Date(r.started_at * 1000).toLocaleString() : "–"}</td>
                <td className="small">{r.started_at ? fmt.dur((r.ended_at || Date.now() / 1000) - r.started_at) : "–"}</td>
                <td><button className="btn sm">Open</button></td>
              </tr>
            ))}
          </tbody></table>
        </div>
      )}
    </div>
  );
}
