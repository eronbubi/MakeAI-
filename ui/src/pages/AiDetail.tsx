import React, { useState } from "react";
import { fmt } from "../api";
import { AiIcon, Icon, Spinner } from "../components/ui";
import { Route, useApp, useFetch } from "../store";
import { EditDialog, ExportDialog, ShareDialog, TrainDialog } from "./AiDialogs";

export default function AiDetail({ route }: { route: Route }) {
  const { open } = useApp();
  const q = useFetch<any>(`/api/models/${route.id}`);
  const [dlg, setDlg] = useState<string | null>(null);
  const m = q.data;
  if (q.error) return <div className="page"><div className="note bad">{q.error}</div></div>;
  if (!m) return <div className="page"><Spinner /></div>;
  const oc = m.original_creator || m.creator;
  const cfg = m.config;
  return (
    <div className="page">
      <div className="head">
        <div className="ai-card"><AiIcon icon={m.icon} name={m.name} /></div>
        <div>
          <h1 className="title">{m.display_name || m.name} <span className="dim">{m.version}</span></h1>
          <div className="dim">Created by {m.creator.name} @{m.creator.username}</div>
        </div>
        <div className="actions">
          <button className="btn primary" disabled={!m.runnable} onClick={() => open({ view: "playground", id: m.uid, title: `Chat · ${m.name}` })}><Icon n="play" s={14} />RUN</button>
          <button className="btn" onClick={() => setDlg("train")}>TRAIN</button>
          <button className="btn" onClick={() => setDlg("share")}>SHARE</button>
          <button className="btn" onClick={() => setDlg("export")}>EXPORT</button>
          <button className="btn" onClick={() => setDlg("edit")}>EDIT</button>
        </div>
      </div>
      <p style={{ maxWidth: 800 }}>{m.description || <span className="faint">No description.</span>}</p>
      <div className="grid g3">
        <div className="card"><h3>Identity</h3>
          <dl className="kv">
            <dt>AI ID</dt><dd>{m.id}</dd>
            <dt>Creator</dt><dd>{m.creator.name} (@{m.creator.username})</dd>
            <dt>Originally created by</dt><dd>{oc.name} (@{oc.username})</dd>
            {m.imported_by?.map((i: any, n: number) => <React.Fragment key={n}><dt>{n === 0 ? "Imported by" : ""}</dt><dd>{i.name} (@{i.username}) · {fmt.date(i.at)}</dd></React.Fragment>)}
            <dt>Created</dt><dd>{fmt.date(m.created)}</dd>
            <dt>Last update</dt><dd>{fmt.date(m.updated)}</dd>
            <dt>Tags</dt><dd>{m.tags?.join(", ") || "–"}</dd>
            <dt>License</dt><dd>{m.license || "–"}</dd>
            <dt>Sharing</dt><dd>{m.sharing?.visibility}</dd>
            <dt>Downloads</dt><dd>{m.downloads || 0}</dd>
          </dl>
        </div>
        <div className="card"><h3>Model</h3>
          <dl className="kv">
            <dt>Status</dt><dd>{m.status}{m.not_runnable_reason ? ` - ${m.not_runnable_reason}` : ""}</dd>
            <dt>Method</dt><dd>{m.method}{m.base_model ? ` on ${m.base_model}` : ""}</dd>
            <dt>Format</dt><dd>{m.source_format || m.format} · runs with {m.backend || "native"}</dd>
            <dt>Parameters</dt><dd>{fmt.int(m.param_count)} ({fmt.params(m.param_count)})</dd>
            <dt>Context</dt><dd>{m.context_length ?? "–"}</dd>
            {cfg && <><dt>Layers / hidden</dt><dd>{cfg.n_layers} / {cfg.hidden_size}</dd>
              <dt>Heads</dt><dd>{cfg.n_heads} query / {cfg.n_kv_heads} kv ({cfg.attention_kind})</dd>
              <dt>Blocks</dt><dd>{cfg.activation} · {cfg.norm} · {cfg.pos_encoding}{cfg.rope_scaling !== "none" ? ` (${cfg.rope_scaling} ×${cfg.rope_scaling_factor})` : ""}</dd>
              <dt>Vocabulary</dt><dd>{cfg.vocab_size}</dd></>}
            {!cfg && m.architecture_summary && Object.entries(m.architecture_summary).map(([k, v]) => <React.Fragment key={k}><dt>{k}</dt><dd>{Array.isArray(v) ? v.join(", ") : String(v ?? "–")}</dd></React.Fragment>)}
            {m.adapters?.map((a: any) => <React.Fragment key={a.name}><dt>Adapter</dt><dd>{a.kind}{a.rank ? ` r=${a.rank} α=${a.alpha}` : ""}{a.base_quant ? ` · base ${a.base_quant}` : ""}</dd></React.Fragment>)}
          </dl>
        </div>
        <div className="card"><h3>Required hardware</h3>
          {m.required_hardware ? <dl className="kv">{Object.entries(m.required_hardware).map(([k, v]) => <React.Fragment key={k}><dt>{k.replace(/_/g, " ").replace(" gb", "")}</dt><dd>{String(v)} GB</dd></React.Fragment>)}</dl> : <span className="dim">–</span>}
          {m.last_eval && <><h3 style={{ marginTop: 14 }}>Last evaluation</h3><dl className="kv"><dt>Val loss</dt><dd>{fmt.num(m.last_eval.val_loss, 4)}</dd><dt>Perplexity</dt><dd>{fmt.num(m.last_eval.val_ppl, 2)}</dd><dt>Token accuracy</dt><dd>{(100 * m.last_eval.token_accuracy).toFixed(1)}%</dd></dl></>}
        </div>
      </div>
      <div className="grid g2" style={{ marginTop: 12 }}>
        <div className="card"><h3>Training runs</h3>
          {m.training_runs?.length ? m.training_runs.slice().reverse().map((r: string) => <div key={r}><button className="btn ghost sm mono" onClick={() => open({ view: "run", id: r, title: `Run · ${m.name}` })}>{r}</button></div>) : <span className="dim">none</span>}
          {m.evaluations?.length > 0 && <><h3 style={{ marginTop: 12 }}>Evaluations</h3>
            <table className="t"><tbody>{m.evaluations.slice().reverse().map((e: any) => <tr key={e.id}><td>{e.name}</td><td className="num">{e.kind === "dataset" ? `loss ${fmt.num(e.loss, 3)} · ppl ${fmt.num(e.perplexity, 1)} · acc ${(100 * e.token_accuracy).toFixed(1)}%` : `accuracy ${e.accuracy != null ? (100 * e.accuracy).toFixed(1) + "%" : "–"} (${e.n})`}</td><td className="small dim">{fmt.date(e.at)}</td></tr>)}</tbody></table></>}
        </div>
        <div className="card"><h3>Provenance</h3>
          <table className="t"><tbody>{m.provenance?.slice().reverse().map((p: any, i: number) => (
            <tr key={i}><td>{p.event}</td><td className="small dim">{Object.entries(p).filter(([k]) => !["event", "at"].includes(k)).map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(",") : v}`).join(" · ")}</td><td className="small dim" style={{ whiteSpace: "nowrap" }}>{fmt.date(p.at)}</td></tr>
          ))}</tbody></table>
        </div>
      </div>
      {dlg === "share" && <ShareDialog m={m} onClose={() => { setDlg(null); q.reload(); }} />}
      {dlg === "export" && <ExportDialog m={m} onClose={() => { setDlg(null); q.reload(); }} />}
      {dlg === "edit" && <EditDialog m={m} onClose={() => { setDlg(null); q.reload(); }} />}
      {dlg === "train" && <TrainDialog m={m} onClose={() => setDlg(null)} />}
    </div>
  );
}
