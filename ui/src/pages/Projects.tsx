import React, { useEffect, useState } from "react";
import { api } from "../api";
import { Dialog, Field, Icon, Sel, Spinner } from "../components/ui";
import { Route, useApp, useFetch } from "../store";

export default function Projects({ route }: { route: Route }) {
  const { toast, open } = useApp();
  const list = useFetch<any[]>("/api/projects");
  const [cur, setCur] = useState<string | null>(route.id || null);
  const [dlg, setDlg] = useState(false);
  const [form, setForm] = useState({ name: "", description: "" });
  useEffect(() => { if (!cur && list.data?.length) setCur(list.data[0].slug); }, [list.data, cur]);
  const create = async () => {
    try { const p = await api.post("/api/projects", form); setDlg(false); list.reload(); setCur(p.slug); }
    catch (e: any) { toast(e.message, true); }
  };
  return (
    <div className="page wide">
      <div className="head"><h1 className="title">Projects</h1>
        <div style={{ width: 240 }}><Sel value={cur || ""} onChange={(v) => setCur(v)} options={[{ value: "", label: list.data?.length ? "Choose project…" : "No projects" }, ...(list.data || []).map((p) => ({ value: p.slug, label: p.name }))]} /></div>
        <div className="actions"><button className="btn primary" onClick={() => setDlg(true)}><Icon n="plus" s={14} />New project</button></div>
      </div>
      {cur ? <ProjectView slug={cur} /> : <div className="empty">A project groups source code, models, datasets, tokenizers, training configurations, evaluation, tests and documentation.</div>}
      <Dialog open={dlg} onClose={() => setDlg(false)} title="New project" footer={<><button className="btn" onClick={() => setDlg(false)}>Cancel</button><button className="btn primary" disabled={!form.name} onClick={create}>Create</button></>}>
        <Field label="Name"><input className="in" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} autoFocus /></Field>
        <Field label="Description" style={{ marginTop: 8 }}><textarea className="in" rows={3} value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} /></Field>
      </Dialog>
    </div>
  );
}

function ProjectView({ slug }: { slug: string }) {
  const { toast, open } = useApp();
  const p = useFetch<any>(`/api/projects/${slug}`);
  const models = useFetch<any[]>("/api/models");
  const datasets = useFetch<any[]>("/api/datasets");
  const toks = useFetch<any[]>("/api/tokenizers");
  const runs = useFetch<any[]>("/api/runs");
  const [file, setFile] = useState<string | null>(null);
  const [content, setContent] = useState<string>("");
  const [orig, setOrig] = useState<string>("");
  const [binary, setBinary] = useState(false);
  const [newName, setNewName] = useState("");
  const loadFile = async (path: string) => {
    const f = await api.get(`/api/projects/${slug}/file?path=${encodeURIComponent(path)}`);
    setFile(path); setBinary(f.binary); setContent(f.content || ""); setOrig(f.content || "");
  };
  const save = async () => { try { await api.put(`/api/projects/${slug}/file`, { path: file, content }); setOrig(content); toast("Saved"); } catch (e: any) { toast(e.message, true); } };
  const link = async (kind: string, ref: string, on = true) => { await api.post(`/api/projects/${slug}/link`, { kind, ref, on }); p.reload(); };
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s" && file) { e.preventDefault(); save(); } };
    window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k);
  });
  if (!p.data) return <Spinner />;
  const links = p.data.project.links || {};
  const lookup: Record<string, any[]> = { models: (models.data || []).map((m) => ({ id: m.uid, name: m.name })), datasets: (datasets.data || []).map((d) => ({ id: d.id, name: d.name })), tokenizers: (toks.data || []).map((t) => ({ id: t.id, name: t.name })), runs: (runs.data || []).map((r) => ({ id: r.run_id, name: `${r.model_name} ${r.run_id.slice(-6)}` })) };
  return (
    <div className="grid" style={{ gridTemplateColumns: "240px minmax(0,1fr) 280px", alignItems: "start" }}>
      <div className="card" style={{ padding: 8 }}>
        <div className="tree">
          {p.data.tree.map((t: any) => (
            <div key={t.path} className={file === t.path ? "on" : ""} style={{ paddingLeft: 6 + 12 * (t.path.split("/").length - 1) }} onClick={() => !t.dir && loadFile(t.path)}>
              <Icon n={t.dir ? "dir" : "file"} s={12} /> {t.path.split("/").at(-1)}
            </div>
          ))}
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <input className="in mono" placeholder="src/train.py" value={newName} onChange={(e) => setNewName(e.target.value)} aria-label="New file path" />
          <button className="btn sm" disabled={!newName} onClick={async () => { await api.put(`/api/projects/${slug}/file`, { path: newName, content: "" }); setNewName(""); p.reload(); loadFile(newName); }}>+</button>
        </div>
      </div>
      <div className="card">
        {file ? (
          <>
            <div className="row" style={{ marginBottom: 8 }}><span className="mono">{file}</span>{content !== orig && <span className="pill warn">modified</span>}<span className="sp" />
              <button className="btn sm danger" onClick={async () => { await api.del(`/api/projects/${slug}/file?path=${encodeURIComponent(file)}`); setFile(null); p.reload(); }}>Delete</button>
              <button className="btn sm primary" disabled={binary || content === orig} onClick={save}>Save <kbd>Ctrl S</kbd></button></div>
            {binary ? <div className="empty">Binary or large file - not editable here.</div>
              : <textarea className="editor" spellCheck={false} value={content} onChange={(e) => setContent(e.target.value)} onKeyDown={(e) => {
                if (e.key === "Tab") { e.preventDefault(); const t = e.currentTarget; const s = t.selectionStart; setContent(content.slice(0, s) + "    " + content.slice(t.selectionEnd)); requestAnimationFrame(() => { t.selectionStart = t.selectionEnd = s + 4; }); }
              }} aria-label={`Editing ${file}`} />}
          </>
        ) : <div className="empty">{p.data.project.description || "Select a file."}<div className="faint small" style={{ marginTop: 6 }}>Folder: src · models · datasets · tokenizers · configs · evaluation · tests · docs</div></div>}
      </div>
      <div className="col">
        {(["models", "datasets", "tokenizers", "runs"] as const).map((kind) => (
          <div className="card" key={kind}><h3 style={{ textTransform: "capitalize" }}>{kind}</h3>
            {(links[kind] || []).map((ref: string) => (
              <div key={ref} className="row small"><button className="btn ghost sm" onClick={() => open(kind === "models" ? { view: "ai", id: ref, title: ref } : kind === "datasets" ? { view: "dataset", id: ref, title: ref } : kind === "runs" ? { view: "run", id: ref, title: ref } : { view: "tokenizers", title: "Tokenizers" })}>{lookup[kind].find((x) => x.id === ref)?.name || ref}</button><span className="sp" /><button className="btn ghost sm" onClick={() => link(kind, ref, false)} aria-label="unlink">×</button></div>
            ))}
            <Sel value="" onChange={(v: string) => v && link(kind, v)} options={[{ value: "", label: `Link ${kind.slice(0, -1)}…` }, ...lookup[kind].filter((x) => !(links[kind] || []).includes(x.id)).map((x) => ({ value: x.id, label: x.name }))]} />
          </div>
        ))}
      </div>
    </div>
  );
}
