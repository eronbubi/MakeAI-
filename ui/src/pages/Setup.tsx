import React, { useEffect, useState } from "react";
import { api, fmt } from "../api";
import { Field, Spinner } from "../components/ui";
import { useApp } from "../store";

/** First launch: who creates the AIs (stored as creator metadata), then the real hardware scan. */
export default function Setup() {
  const { reloadSettings, setHw } = useApp();
  const [name, setName] = useState("");
  const [user, setUser] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [hw, setLocalHw] = useState<any>(null);
  const [scanning, setScanning] = useState(true);
  useEffect(() => {
    api.get("/api/hardware?refresh=true").then((h) => { setLocalHw(h); setHw(h); }).catch((e) => setErr(e.message)).finally(() => setScanning(false));
  }, [setHw]);
  const save = async () => {
    try {
      await api.patch("/api/settings", { profile: { name: name.trim(), username: user.trim() } });
      await reloadSettings();
    } catch (e: any) { setErr(e.message); }
  };
  const g = hw?.gpus?.[0];
  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 16 }}>
      <div className="card" style={{ width: "min(560px, 100%)", padding: 24 }}>
        <div className="row" style={{ gap: 10, marginBottom: 6 }}><img src="/favicon.svg" width={28} height={28} alt="" /><h1 className="title">Welcome to MakeAI</h1></div>
        <p className="dim" style={{ marginTop: 0 }}>Create, train and run your own AI models on this computer. Your name and username are stored as the permanent creator of every AI you make.</p>
        <div className="grid g2" style={{ marginTop: 14 }}>
          <Field label="Your name"><input className="in" value={name} onChange={(e) => setName(e.target.value)} placeholder="Eron" autoFocus /></Field>
          <Field label="Username" hint="a–z, 0–9, _ - ."><input className="in mono" value={user} onChange={(e) => setUser(e.target.value.replace(/^@/, "").toLowerCase())} placeholder="eron" /></Field>
        </div>
        <h2 className="sec">Hardware scan</h2>
        {scanning && <div className="row dim"><Spinner /> Scanning GPU, CPU, memory and storage…</div>}
        {hw && (
          <dl className="kv">
            <dt>GPU</dt><dd>{g ? `${g.name} · ${fmt.mb(g.vram_total_mb)} VRAM · compute ${g.compute_capability || "?"}` : "no CUDA GPU detected - training runs on CPU"}</dd>
            {g && <><dt>Precision</dt><dd>{["fp16", "bf16", "tf32"].filter((k) => g[k]).join(" · ").toUpperCase()}{g.tensor_cores ? " · Tensor Cores" : ""}</dd></>}
            <dt>CPU</dt><dd>{hw.cpu.name} · {hw.cpu.cores} cores / {hw.cpu.threads} threads</dd>
            <dt>RAM</dt><dd>{fmt.mb(hw.ram.total_mb)}</dd>
            <dt>CUDA</dt><dd>{hw.torch.cuda ? `${hw.torch.cuda} (driver ${hw.driver_version}, supports ${hw.driver_cuda_version})` : "not available"}</dd>
            <dt>Storage</dt><dd>{hw.makeai_home.free_gb} GB free at {hw.makeai_home.path}</dd>
          </dl>
        )}
        {err && <div className="note bad" style={{ marginTop: 10 }}>{err}</div>}
        <div className="row" style={{ marginTop: 18, justifyContent: "flex-end" }}>
          <button className="btn primary lg" disabled={!name.trim() || !user.trim()} onClick={save}>Continue</button>
        </div>
      </div>
    </div>
  );
}
