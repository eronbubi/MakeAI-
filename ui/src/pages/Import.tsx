import React, { useRef, useState } from "react";
import { api, fmt } from "../api";
import { Field, PathPicker, Spinner } from "../components/ui";
import { useApp } from "../store";

export default function Import() {
  const { track, toast, open, settings } = useApp();
  const [path, setPath] = useState("");
  const [pick, setPick] = useState<null | "file" | "dir">(null);
  const [info, setInfo] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [o, setO] = useState({ name: "", version: "1.0", creator_name: "", creator_username: "", description: "" });
  const up = useRef<HTMLInputElement>(null);
  const inspect = async (p: string) => {
    setPath(p); setInfo(null); setBusy(true);
    try {
      const i = await api.post("/api/import/inspect", { path: p });
      setInfo(i);
      setO({ name: i.name || "", version: i.version || "1.0", creator_name: i.hf_org || i.author || "", creator_username: (i.hf_org || "").toLowerCase(), description: i.description || "" });
    } catch (e: any) { toast(e.message, true); } finally { setBusy(false); }
  };
  const doImport = async () => {
    setBusy(true);
    try {
      const m = await track(await api.post("/api/import", { path, overrides: info.attribution ? { version: o.version } : o }), `Import ${info.name}`);
      toast(`Imported ${m.id}`);
      open({ view: "ai", id: m.uid, title: m.name });
    } catch (e: any) { toast(e.message, true); } finally { setBusy(false); }
  };
  const a = info?.attribution;
  return (
    <div className="page">
      <div className="head"><h1 className="title">Import</h1><span className="sub">.makeai · .safetensors (+config.json) · .gguf · .pt / .pth · .onnx</span></div>
      <div className="card">
        <div className="row">
          <input className="in mono" style={{ flex: 1 }} value={path} onChange={(e) => setPath(e.target.value)} placeholder="Path to a model file or Hugging Face folder" onKeyDown={(e) => e.key === "Enter" && inspect(path)} />
          <button className="btn" onClick={() => setPick("file")}>Choose file…</button>
          <button className="btn" onClick={() => setPick("dir")}>Choose folder…</button>
          <button className="btn" onClick={() => up.current?.click()}>Upload…</button>
          <input ref={up} type="file" hidden accept=".makeai,.safetensors,.gguf,.pt,.pth,.onnx" onChange={async (e) => { if (e.target.files?.[0]) { const r = await api.upload(e.target.files); inspect(r.paths[0]); } }} />
          <button className="btn primary" disabled={!path || busy} onClick={() => inspect(path)}>Inspect</button>
        </div>
      </div>
      {busy && !info && <div style={{ marginTop: 12 }}><Spinner /></div>}
      {info && (
        <div className="grid g2" style={{ marginTop: 12 }}>
          <div className="card"><h3>Inspection · {info.format}</h3>
            <dl className="kv">
              <dt>Name</dt><dd>{info.name}</dd>
              <dt>Parameters</dt><dd>{fmt.int(info.param_count)} ({fmt.params(info.param_count)})</dd>
              {info.tensor_count != null && <><dt>Tensors</dt><dd>{fmt.int(info.tensor_count)}</dd></>}
              {info.architecture_name && <><dt>Architecture</dt><dd>{info.architecture_name}</dd></>}
              {info.hf_architecture && <><dt>HF architecture</dt><dd>{info.hf_architecture.join(", ")}</dd></>}
              {info.architecture && <><dt>Shape</dt><dd>{info.architecture.n_layers} layers · hidden {info.architecture.hidden_size} · {info.architecture.n_heads}/{info.architecture.n_kv_heads} heads · vocab {info.architecture.vocab_size}</dd></>}
              {info.context_length && <><dt>Context</dt><dd>{info.context_length}</dd></>}
              {info.quantization && <><dt>Quantization</dt><dd>{info.quantization.join(", ")}</dd></>}
              {info.dtypes && <><dt>Dtypes</dt><dd>{info.dtypes.join(", ")}</dd></>}
              {info.opset && <><dt>Opset</dt><dd>{info.opset.join(", ")}</dd></>}
              {info.inputs && <><dt>Inputs</dt><dd>{info.inputs.map((i: any) => `${i.name}[${i.shape.join(",")}]`).join(" ")}</dd><dt>Outputs</dt><dd>{info.outputs.map((i: any) => `${i.name}[${i.shape.join(",")}]`).join(" ")}</dd></>}
              <dt>Runs in MakeAI</dt><dd className={info.runnable ? "ok-t" : "warn-t"}>{info.runnable ? `yes · ${info.backend}` : `no - ${info.reason}`}</dd>
            </dl>
            {info.warnings?.map((w: string) => <div key={w} className="note warn" style={{ marginTop: 6 }}>{w}</div>)}
            {info.sample_tensors && <details style={{ marginTop: 8 }}><summary className="dim small">Tensors</summary><table className="t"><tbody>{info.sample_tensors.map((t: any) => <tr key={t.name}><td className="mono small">{t.name}</td><td className="mono small">[{t.shape.join(", ")}]</td><td className="small">{t.dtype}</td></tr>)}</tbody></table></details>}
          </div>
          <div className="card"><h3>Attribution</h3>
            {a ? (
              <>
                <div className="note ok">This file carries MakeAI creator metadata. The original creator is preserved; you are recorded as importer.</div>
                <dl className="kv" style={{ marginTop: 10 }}>
                  <dt>ID</dt><dd>{a.id || info.id}</dd>
                  <dt>Created by</dt><dd>{a.creator.name} (@{a.creator.username})</dd>
                  <dt>Originally created by</dt><dd>{a.original_creator.name} (@{a.original_creator.username})</dd>
                  {a.imported_by?.map((i: any, n: number) => <React.Fragment key={n}><dt>Imported by</dt><dd>{i.name} (@{i.username})</dd></React.Fragment>)}
                  <dt>Will be imported by</dt><dd>{settings.profile.name} (@{settings.profile.username})</dd>
                </dl>
              </>
            ) : (
              <>
                <div className="note warn">No creator metadata in this file. Enter who originally created it (for Hugging Face downloads this is prefilled with the publishing organisation). This becomes the permanent original creator.</div>
                <div className="grid g2" style={{ marginTop: 10 }}>
                  <Field label="AI name"><input className="in" value={o.name} onChange={(e) => setO({ ...o, name: e.target.value })} /></Field>
                  <Field label="Version"><input className="in mono" value={o.version} onChange={(e) => setO({ ...o, version: e.target.value })} /></Field>
                  <Field label="Original creator"><input className="in" value={o.creator_name} onChange={(e) => setO({ ...o, creator_name: e.target.value })} /></Field>
                  <Field label="Creator username"><input className="in mono" value={o.creator_username} onChange={(e) => setO({ ...o, creator_username: e.target.value.toLowerCase() })} /></Field>
                </div>
                <Field label="Description" style={{ marginTop: 8 }}><textarea className="in" rows={2} value={o.description} onChange={(e) => setO({ ...o, description: e.target.value })} /></Field>
              </>
            )}
            <button className="btn primary" style={{ marginTop: 12 }} disabled={busy || (!a && (!o.creator_name || !o.creator_username || !o.name))} onClick={doImport}>{busy ? <Spinner /> : "Import into My AIs"}</button>
          </div>
        </div>
      )}
      <PathPicker open={!!pick} mode={pick || "any"} onClose={() => setPick(null)} onPick={(p) => inspect(p[0])} />
    </div>
  );
}
