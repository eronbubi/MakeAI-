import React, { useRef, useState } from "react";
import { api, fmt } from "../api";
import { Dialog, Field, Icon, PathPicker, Sel, Spinner } from "../components/ui";
import { useApp, useFetch } from "../store";

export default function Datasets() {
  const { open, track, toast } = useApp();
  const ds = useFetch<any[]>("/api/datasets");
  const [dlg, setDlg] = useState(false);
  return (
    <div className="page">
      <div className="head">
        <h1 className="title">Datasets</h1><span className="sub">JSON · JSONL · TXT · CSV · Parquet · folders</span>
        <div className="actions"><button className="btn primary" onClick={() => setDlg(true)}><Icon n="plus" s={14} />Import dataset</button></div>
      </div>
      {ds.data && !ds.data.length && <div className="empty">No datasets. Import files or whole folders - they stay on your disk and are read locally.</div>}
      {!!ds.data?.length && (
        <div className="card" style={{ padding: 0 }}>
          <table className="t"><thead><tr><th>Name</th><th className="num">Samples</th><th className="num">Tokens</th><th className="num">Size</th><th className="num">Avg length</th><th>Type</th><th>Updated</th></tr></thead><tbody>
            {ds.data.map((d) => (
              <tr key={d.id} style={{ cursor: "pointer" }} onClick={() => open({ view: "dataset", id: d.id, title: d.name })}>
                <td><b>{d.name}</b><div className="faint small mono">{d.id}</div></td>
                <td className="num">{fmt.int(d.stats?.samples)}</td>
                <td className="num">{d.stats?.tokens ? fmt.int(d.stats.tokens) : <span className="faint">not counted</span>}</td>
                <td className="num">{fmt.bytes(d.stats?.bytes)}</td>
                <td className="num">{d.stats?.avg_tokens ? `${fmt.int(d.stats.avg_tokens)} tok` : `${fmt.int(d.stats?.avg_chars)} chars`}</td>
                <td>{d.stats?.kinds?.chat ? <span className="pill acc">chat {fmt.int(d.stats.kinds.chat)}</span> : null} {d.stats?.kinds?.text ? <span className="pill">text {fmt.int(d.stats.kinds.text)}</span> : null}</td>
                <td className="small">{fmt.date(d.updated)}</td>
              </tr>
            ))}
          </tbody></table>
        </div>
      )}
      {dlg && <ImportDataset onClose={() => setDlg(false)} onDone={(m) => { setDlg(false); ds.reload(); open({ view: "dataset", id: m.id, title: m.name }); }} track={track} toast={toast} />}
    </div>
  );
}

function ImportDataset({ onClose, onDone, track, toast }: { onClose: () => void; onDone: (m: any) => void; track: any; toast: any }) {
  const [name, setName] = useState("");
  const [paths, setPaths] = useState<string[]>([]);
  const [picker, setPicker] = useState<null | "file" | "dir">(null);
  const [txtMode, setTxtMode] = useState("file");
  const [exts, setExts] = useState("");
  const [field, setField] = useState("");
  const [busy, setBusy] = useState(false);
  const up = useRef<HTMLInputElement>(null);
  const go = async () => {
    setBusy(true);
    try {
      const job = await api.post("/api/datasets", { name: name || "dataset", paths, options: { txt_mode: txtMode, extensions: exts.split(",").map((s) => s.trim()).filter(Boolean), text_field: field || undefined } });
      onDone(await track(job, `Import dataset ${name}`));
    } catch (e: any) { toast(e.message, true); } finally { setBusy(false); }
  };
  return (
    <Dialog open onClose={onClose} title="Import dataset" wide footer={<><button className="btn" onClick={onClose}>Cancel</button><button className="btn primary" disabled={!paths.length || busy} onClick={go}>{busy ? <Spinner /> : "Import"}</button></>}>
      <div className="col">
        <Field label="Name"><input className="in" value={name} onChange={(e) => setName(e.target.value)} placeholder="my-dataset" autoFocus /></Field>
        <div className="row">
          <button className="btn" onClick={() => setPicker("file")}>Add files…</button>
          <button className="btn" onClick={() => setPicker("dir")}>Add folder…</button>
          <button className="btn" onClick={() => up.current?.click()}>Upload from browser…</button>
          <input ref={up} type="file" multiple hidden onChange={async (e) => { if (e.target.files?.length) { const r = await api.upload(e.target.files); setPaths((p) => [...p, ...r.paths]); } }} />
        </div>
        {paths.length > 0 && <div className="note mono small">{paths.map((p) => <div key={p} className="row">{p}<span className="sp" /><button className="btn ghost sm" onClick={() => setPaths(paths.filter((x) => x !== p))} aria-label="remove">×</button></div>)}</div>}
        <div className="grid g3">
          <Field label="Plain text files become" hint="records">
            <Sel value={txtMode} onChange={setTxtMode} options={[{ value: "file", label: "one record per file" }, { value: "paragraph", label: "one per paragraph" }, { value: "line", label: "one per line" }]} />
          </Field>
          <Field label="Folder file types" hint="e.g. .py,.md - empty = all supported"><input className="in mono" value={exts} onChange={(e) => setExts(e.target.value)} /></Field>
          <Field label="Text field" hint="auto-detected: text, content, messages, prompt/response, instruction/output…"><input className="in mono" value={field} onChange={(e) => setField(e.target.value)} /></Field>
        </div>
      </div>
      <PathPicker open={!!picker} mode={picker || "any"} multiple onClose={() => setPicker(null)} onPick={(p) => setPaths((x) => [...new Set([...x, ...p])])} />
    </Dialog>
  );
}
