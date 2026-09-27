import React, { useState } from "react";
import { fmt } from "../api";
import { AiIcon, Icon, Seg } from "../components/ui";
import { useApp, useFetch } from "../store";
import { EditDialog, ExportDialog, ShareDialog, TrainDialog } from "./AiDialogs";

export default function MyAIs() {
  const { open } = useApp();
  const models = useFetch<any[]>("/api/models");
  const [filter, setFilter] = useState<"all" | "mine" | "imported">("all");
  const [q, setQ] = useState("");
  const [dlg, setDlg] = useState<{ kind: string; m: any } | null>(null);
  const { settings } = useApp();
  const list = (models.data || []).filter((m) =>
    (filter === "all" || (filter === "mine" ? m.creator.username === settings.profile.username : m.imported_by?.length)) &&
    (!q || `${m.name} ${m.description} ${m.creator.name} ${m.tags?.join(" ")}`.toLowerCase().includes(q.toLowerCase())));
  return (
    <div className="page">
      <div className="head">
        <h1 className="title">My AIs</h1><span className="sub">{models.data?.length ?? 0} AIs</span>
        <div className="actions">
          <input className="in" style={{ width: 200 }} placeholder="Filter…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter AIs" />
          <Seg value={filter} onChange={setFilter} options={[{ value: "all", label: "All" }, { value: "mine", label: "Created by me" }, { value: "imported", label: "Imported" }]} />
          <button className="btn" onClick={() => open({ view: "import", title: "Import" })}><Icon n="import" s={14} />Import</button>
          <button className="btn primary" onClick={() => open({ view: "create", title: "Create AI" })}><Icon n="plus" s={14} />Create AI</button>
        </div>
      </div>
      {models.data && !list.length && <div className="empty">No AIs here yet.</div>}
      <div className="grid gauto">
        {list.map((m) => <AiCard key={m.uid} m={m} onAction={(kind) => {
          if (kind === "run") open({ view: "playground", id: m.uid, title: `Chat · ${m.name}` });
          else if (kind === "open") open({ view: "ai", id: m.uid, title: m.name });
          else setDlg({ kind, m });
        }} />)}
      </div>
      {dlg?.kind === "share" && <ShareDialog m={dlg.m} onClose={() => { setDlg(null); models.reload(); }} />}
      {dlg?.kind === "export" && <ExportDialog m={dlg.m} onClose={() => setDlg(null)} />}
      {dlg?.kind === "edit" && <EditDialog m={dlg.m} onClose={() => { setDlg(null); models.reload(); }} />}
      {dlg?.kind === "train" && <TrainDialog m={dlg.m} onClose={() => setDlg(null)} />}
    </div>
  );
}

export function AiCard({ m, onAction }: { m: any; onAction: (k: string) => void }) {
  const oc = m.original_creator || m.creator;
  const imported = m.imported_by?.length;
  return (
    <div className="card ai-card">
      <div className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap", cursor: "pointer" }} onClick={() => onAction("open")}>
        <AiIcon icon={m.icon} name={m.name} />
        <div style={{ minWidth: 0 }}>
          <div className="name">{m.display_name || m.name}</div>
          <div className="by">Created by {m.creator.name} <span className="faint">@{m.creator.username}</span></div>
          {imported ? <div className="by faint small">Imported by {m.imported_by.at(-1).name}{oc.username !== m.creator.username ? ` · originally by ${oc.name}` : ""}</div> : null}
        </div>
      </div>
      <div className="desc">{m.description || <span className="faint">No description</span>}</div>
      <div className="row small" style={{ gap: 6 }}>
        <span className="pill">Version {m.version}</span>
        <span className="pill">{fmt.params(m.param_count)} Parameters</span>
        <span className={`pill ${m.runnable ? "good" : m.status === "training" ? "acc" : ""}`}>{m.status}</span>
        {m.sharing?.visibility !== "private" && <span className="pill acc">{m.sharing?.visibility}</span>}
      </div>
      <div className="acts">
        <button className="btn primary" disabled={!m.runnable} title={m.runnable ? "Chat with this AI" : m.not_runnable_reason || "not trained yet"} onClick={() => onAction("run")}>RUN</button>
        <button className="btn" disabled={m.backend && m.backend !== "native"} title={m.backend && m.backend !== "native" ? `${m.backend} models can be run, not trained` : "Train"} onClick={() => onAction("train")}>TRAIN</button>
        <button className="btn" onClick={() => onAction("share")}>SHARE</button>
        <button className="btn" onClick={() => onAction("export")}>EXPORT</button>
        <button className="btn" onClick={() => onAction("edit")}>EDIT</button>
      </div>
    </div>
  );
}
