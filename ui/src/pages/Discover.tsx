import React, { useState } from "react";
import { api, fmt } from "../api";
import { AiIcon, Spinner } from "../components/ui";
import { useApp, useFetch } from "../store";

const CATS = ["popular", "new", "coding", "reasoning", "general", "experimental", "small", "large"];

export default function Discover() {
  const { track, toast, open, settings } = useApp();
  const [cat, setCat] = useState("new");
  const [q, setQ] = useState("");
  const d = useFetch<any>(`/api/discover?category=${cat}&q=${encodeURIComponent(q)}`);
  const [busy, setBusy] = useState<string | null>(null);
  const install = async (it: any) => {
    setBusy(it.id);
    try { const m = await track(await api.post("/api/discover/install", { download: it.download, name: it.name }), `Install ${it.name}`); toast(`Installed ${m.id}`); open({ view: "ai", id: m.uid, title: m.name }); }
    catch (e: any) { toast(e.message, true); } finally { setBusy(null); }
  };
  return (
    <div className="page">
      <div className="head"><h1 className="title">Discover</h1>
        <span className="sub">Public AIs on this computer{settings.discover_hubs?.length ? ` and ${settings.discover_hubs.length} connected MakeAI hub(s)` : ""}.</span>
        <div className="actions"><input className="in" style={{ width: 240 }} placeholder="Search name, creator, tag…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search" /></div>
      </div>
      <div className="row" style={{ marginBottom: 12 }} role="tablist">
        {CATS.map((c) => <button key={c} role="tab" aria-selected={c === cat} className={`btn sm ${c === cat ? "primary" : ""}`} onClick={() => setCat(c)}>{c[0].toUpperCase() + c.slice(1)}</button>)}
      </div>
      {d.data?.errors?.map((e: string) => <div key={e} className="note warn" style={{ marginBottom: 6 }}>Hub unreachable: {e}</div>)}
      {!d.data && <Spinner />}
      {d.data && !d.data.items.length && <div className="empty">Nothing here yet. AIs appear when their owner sets them to Public (SHARE → Public). Add other MakeAI instances as hubs in Settings to browse theirs.</div>}
      <div className="grid gauto">
        {d.data?.items?.map((it: any) => (
          <div key={`${it.hub}-${it.id}`} className="card ai-card">
            <div className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
              <AiIcon icon={it.icon} name={it.name} />
              <div><div className="name">{it.name}</div>
                <div className="by">Created by {it.creator.name} <span className="faint">@{it.creator.username}</span></div>
                {it.original_creator?.username !== it.creator.username && <div className="by faint small">Originally created by {it.original_creator.name}</div>}
              </div>
            </div>
            <div className="desc">{it.description || <span className="faint">No description</span>}</div>
            <div className="row small" style={{ gap: 6 }}>
              <span className="pill">v{it.version}</span><span className="pill">{fmt.params(it.param_count)}</span><span className="pill">ctx {it.context_length ?? "?"}</span>
              <span className="pill">{it.downloads} downloads</span>{it.tags?.map((t: string) => <span key={t} className="pill acc">{t}</span>)}
            </div>
            <div className="faint small">{it.hub} · updated {fmt.date(it.updated)}</div>
            <div className="row">
              {it.page && <a className="btn sm" href={it.page} target="_blank" rel="noreferrer">Page</a>}
              {it.hub !== "this computer" && it.download && <button className="btn sm primary" disabled={!!busy} onClick={() => install(it)}>{busy === it.id ? <Spinner /> : "Install"}</button>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
