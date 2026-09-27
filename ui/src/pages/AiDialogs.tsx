import React, { useEffect, useState } from "react";
import { api, fmt } from "../api";
import { Check, Dialog, Field, Num, Seg, Sel, Spinner } from "../components/ui";
import { useApp, useFetch } from "../store";

export function ShareDialog({ m, onClose }: { m: any; onClose: () => void }) {
  const { toast } = useApp();
  const [vis, setVis] = useState<string>(m.sharing?.visibility || "private");
  const [card, setCard] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const apply = async (v: string) => {
    setBusy(true);
    try { const c = await api.post(`/api/models/${m.uid}/share`, { visibility: v }); setVis(v); setCard(c); toast(`${m.name} is now ${v}`); }
    catch (e: any) { toast(e.message, true); } finally { setBusy(false); }
  };
  useEffect(() => { if (vis !== "private") apply(vis); /* fetch link */ // eslint-disable-next-line
  }, []);
  const lan = location.hostname === "127.0.0.1" || location.hostname === "localhost";
  return (
    <Dialog open onClose={onClose} title={`Share ${m.name}`} footer={<button className="btn" onClick={onClose}>Done</button>}>
      <div className="col">
        <Seg value={vis} onChange={apply} options={[{ value: "private", label: "Private" }, { value: "link", label: "Link" }, { value: "public", label: "Public" }]} />
        <div className="dim small">
          {vis === "private" && "Only you can see and use this AI."}
          {vis === "link" && "Anyone with the link can view the AI page and download its .makeai package. It is not listed in Discover."}
          {vis === "public" && "Listed in Discover and in this computer's public index; anyone who can reach this MakeAI instance can download it."}
        </div>
        {busy && <Spinner />}
        {card?.page && vis !== "private" && (
          <>
            <Field label="Share page"><div className="row"><input className="in mono" readOnly value={card.page} onFocus={(e) => e.target.select()} /><button className="btn sm" onClick={() => navigator.clipboard?.writeText(card.page).then(() => toast("Link copied"))}>Copy</button><a className="btn sm" href={card.page} target="_blank" rel="noreferrer">Open</a></div></Field>
            {lan && <div className="note">This link works on this computer. To let other computers open it, start MakeAI with <span className="mono">python -m makeai --host 0.0.0.0</span> (Settings → Sharing) and replace 127.0.0.1 with this PC's network address. Other people can also receive the .makeai file directly (Export).</div>}
          </>
        )}
        <div className="note">Created by <b>{m.creator.name}</b> @{m.creator.username}{(m.original_creator?.username || m.creator.username) !== m.creator.username && <> · originally created by <b>{m.original_creator.name}</b></>}. Attribution travels with every copy and cannot be replaced by the importer.</div>
      </div>
    </Dialog>
  );
}

export function ExportDialog({ m, onClose }: { m: any; onClose: () => void }) {
  const { track, toast } = useApp();
  const av = useFetch<any>(`/api/models/${m.uid}/export`);
  const [busy, setBusy] = useState<string | null>(null);
  const [done, setDone] = useState<Record<string, any>>({});
  const run = async (fmtKey: string) => {
    setBusy(fmtKey);
    try {
      const r = await track(await api.post(`/api/models/${m.uid}/export`, { format: fmtKey }), `Export ${m.name} → ${fmtKey}`);
      setDone((d) => ({ ...d, [fmtKey]: r }));
      toast(`Exported ${r.file} (${fmt.bytes(r.bytes)})`);
    } catch (e: any) { toast(e.message, true); } finally { setBusy(null); }
  };
  return (
    <Dialog open onClose={onClose} wide title={`Export ${m.name}`} footer={<button className="btn" onClick={onClose}>Close</button>}>
      {!av.data && <Spinner />}
      <table className="t"><tbody>
        {av.data && Object.entries(av.data).map(([k, v]: any) => (
          <tr key={k}>
            <td><b>{v.label}</b>{v.reason && <div className={`small ${v.available ? "dim" : "warn-t"}`}>{v.reason}</div>}</td>
            <td style={{ width: 230, textAlign: "right", whiteSpace: "nowrap" }}>
              {done[k] && <a className="btn sm" href={done[k].download} download>Download {fmt.bytes(done[k].bytes)}</a>}{" "}
              <button className="btn sm primary" disabled={!v.available || !!busy} onClick={() => run(k)}>{busy === k ? <Spinner /> : "Export"}</button>
            </td>
          </tr>
        ))}
      </tbody></table>
      <div className="dim small" style={{ marginTop: 8 }}>Files are written to your MakeAI exports folder. Creator attribution is embedded in each format's metadata (safetensors header, GGUF keys, ONNX metadata, PEFT side file, package manifest).</div>
    </Dialog>
  );
}

export function EditDialog({ m, onClose }: { m: any; onClose: () => void }) {
  const { toast, settings, open } = useApp();
  const [f, setF] = useState({ description: m.description || "", tags: (m.tags || []).join(", "), icon: m.icon || "", license: m.license || "", name: m.display_name || m.name });
  const [ver, setVer] = useState("");
  const [del, setDel] = useState(false);
  const save = async () => {
    try { await api.patch(`/api/models/${m.uid}`, { ...f, tags: f.tags.split(",").map((s: string) => s.trim()).filter(Boolean) }); toast("Saved"); onClose(); }
    catch (e: any) { toast(e.message, true); }
  };
  return (
    <Dialog open onClose={onClose} title={`Edit ${m.name}`} footer={<><button className="btn danger" style={{ marginRight: "auto" }} onClick={() => setDel(true)}>Delete…</button><button className="btn" onClick={onClose}>Cancel</button><button className="btn primary" onClick={save}>Save</button></>}>
      <div className="col">
        <div className="note">ID <span className="mono">{m.id}</span> · created by <b>{m.creator.name}</b> @{m.creator.username} - the creator and original creator are permanent and cannot be edited.</div>
        <Field label="Display name" hint="the AI ID stays the same"><input className="in" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} /></Field>
        <Field label="Description"><textarea className="in" rows={3} value={f.description} onChange={(e) => setF({ ...f, description: e.target.value })} /></Field>
        <div className="grid g2">
          <Field label="Tags"><input className="in" value={f.tags} onChange={(e) => setF({ ...f, tags: e.target.value })} /></Field>
          <Field label="Icon (emoji)"><input className="in" value={f.icon.startsWith("data:") ? "" : f.icon} onChange={(e) => setF({ ...f, icon: e.target.value })} /></Field>
        </div>
        <Field label="License"><input className="in" value={f.license} onChange={(e) => setF({ ...f, license: e.target.value })} /></Field>
        <div className="row"><Field label="New version" style={{ flex: 1 }}><input className="in mono" placeholder="e.g. 1.1" value={ver} onChange={(e) => setVer(e.target.value)} /></Field>
          <button className="btn" style={{ alignSelf: "flex-end" }} disabled={!ver} onClick={async () => { try { const n = await api.post(`/api/models/${m.uid}/version`, { version: ver }); toast(`Created ${n.id}`); onClose(); open({ view: "ai", id: n.uid, title: n.name }); } catch (e: any) { toast(e.message, true); } }}>Create version</button></div>
      </div>
      <Dialog open={del} onClose={() => setDel(false)} title={`Delete ${m.name}?`} footer={<><button className="btn" onClick={() => setDel(false)} autoFocus>Cancel</button><button className="btn danger" onClick={async () => { try { await api.del(`/api/models/${m.uid}`); toast("Deleted"); setDel(false); onClose(); } catch (e: any) { toast(e.message, true); } }}>Delete permanently</button></>}>
        This removes the AI's weights, adapters and metadata from this computer. Exported files and training runs are kept.
      </Dialog>
    </Dialog>
  );
}

export function TrainDialog({ m, onClose }: { m: any; onClose: () => void }) {
  const { open, toast } = useApp();
  const datasets = useFetch<any[]>("/api/datasets");
  const plan = useFetch<any>(`/api/models/${m.uid}/plan`);
  const trained = m.runnable && (m.method === "from_scratch" || m.method === "imported" || m.method === "full");
  const [mode, setMode] = useState<"continue" | "adapter">(trained ? "continue" : "continue");
  const [sel, setSel] = useState<string[]>([]);
  const [complexity, setComplexity] = useState(4);
  const [rec, setRec] = useState<any>(null);
  const [steps, setSteps] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const usePlan = !trained && plan.data?.training;
  useEffect(() => { if (plan.data?.datasets) setSel(plan.data.datasets.map((d: any) => d.id)); }, [plan.data]);
  useEffect(() => {
    if (usePlan || !sel.length) return;
    const method = trained ? "full" : m.method;
    api.post("/api/recommend", { complexity, method, base_model: method === "from_scratch" ? undefined : (m.base_model || m.uid), dataset_ids: sel })
      .then((r) => { setRec(r); setSteps(r.training.max_steps); }).catch((e) => toast(e.message, true));
  }, [sel.join(","), complexity, usePlan]); // eslint-disable-line
  const start = async () => {
    setBusy(true);
    try {
      const body = usePlan
        ? { model_uid: m.uid, datasets: plan.data.datasets, training: { ...plan.data.training, save_every: plan.data.checkpoint?.save_every }, checkpoint: plan.data.checkpoint, data_prep: plan.data.data_prep }
        : { model_uid: m.uid, datasets: sel.map((id) => ({ id, weight: 1 })), training: { ...rec.training, max_steps: steps }, checkpoint: { save_every: rec.training.save_every, keep: 3 } };
      const r = await api.post("/api/runs", body);
      onClose();
      open({ view: "run", id: r.run_id, title: `Run · ${m.name}` });
    } catch (e: any) { toast(e.message, true); } finally { setBusy(false); }
  };
  return (
    <Dialog open onClose={onClose} title={`Train ${m.name}`} footer={<><button className="btn" onClick={onClose}>Cancel</button>
      {mode === "adapter" ? <button className="btn primary" onClick={() => { onClose(); open({ view: "create", id: `base:${m.uid}`, title: "Create AI" }); }}>Open Create AI</button>
        : <button className="btn primary" disabled={busy || (!usePlan && (!rec || !sel.length))} onClick={start}>{busy ? <Spinner /> : "Start training"}</button>}</>}>
      <div className="col">
        {trained && <Seg value={mode} onChange={setMode} options={[{ value: "continue", label: "Continue training this AI" }, { value: "adapter", label: "Fine-tune into a new AI (LoRA / QLoRA / adapter)" }]} />}
        {mode === "adapter" && <div className="dim">Creates a new AI that uses {m.name} as its base model. The new AI records {m.original_creator?.name || m.creator.name} as creator of the base.</div>}
        {mode === "continue" && usePlan && <div className="note ok">Uses the plan saved when this AI was created: {plan.data.training.max_steps ? `${fmt.int(plan.data.training.max_steps)} steps` : `${plan.data.training.epochs} epochs`}, {plan.data.training.optimizer}, lr {plan.data.training.learning_rate}.</div>}
        {mode === "continue" && !usePlan && (
          <>
            <Field label="Datasets">
              <div className="col" style={{ gap: 4, maxHeight: 180, overflow: "auto" }}>{datasets.data?.map((d) => <Check key={d.id} checked={sel.includes(d.id)} onChange={(v) => setSel(v ? [...sel, d.id] : sel.filter((x) => x !== d.id))}>{d.name} <span className="dim small">{fmt.int(d.stats?.samples)} samples</span></Check>)}</div>
            </Field>
            <div className="grid g2">
              <Field label="Complexity"><Sel value={complexity} onChange={setComplexity} options={[1, 4, 16, 64, 256]} /></Field>
              <Field label="Steps"><Num value={steps} onChange={setSteps} min={1} /></Field>
            </div>
            {rec && <div className="dim small">{rec.training.optimizer} · lr {rec.training.learning_rate} · batch {rec.training.micro_batch_size}×{rec.training.gradient_accumulation} · {rec.training.precision.toUpperCase()} · est. VRAM {fmt.bytes(rec.estimates.vram_bytes)}. Full settings are editable in Create AI.</div>}
          </>
        )}
      </div>
    </Dialog>
  );
}
