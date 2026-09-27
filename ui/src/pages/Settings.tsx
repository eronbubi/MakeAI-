import React, { useState } from "react";
import { api } from "../api";
import { Check, Field, Seg } from "../components/ui";
import { useApp, useFetch } from "../store";

export default function Settings() {
  const { settings, setSettings, toast } = useApp();
  const health = useFetch<any>("/api/health");
  const [p, setP] = useState(settings.profile);
  const [hub, setHub] = useState("");
  const [theme, setTheme] = useState<string>(() => { try { return localStorage.getItem("makeai.theme") || "light"; } catch { return "dark"; } });
  const save = async (patch: any) => {
    try { setSettings(await api.patch("/api/settings", patch)); toast("Settings saved"); } catch (e: any) { toast(e.message, true); }
  };
  return (
    <div className="page" style={{ maxWidth: 820 }}>
      <div className="head"><h1 className="title">Settings</h1></div>
      <div className="card"><h3>Profile</h3>
        <div className="dim small" style={{ marginBottom: 8 }}>Used as creator of new AIs and recorded as importer. Existing AIs keep their creator.</div>
        <div className="grid g2">
          <Field label="Name"><input className="in" value={p.name} onChange={(e) => setP({ ...p, name: e.target.value })} /></Field>
          <Field label="Username"><input className="in mono" value={p.username} onChange={(e) => setP({ ...p, username: e.target.value.toLowerCase() })} /></Field>
        </div>
        <button className="btn primary" style={{ marginTop: 10 }} onClick={() => save({ profile: p })}>Save profile</button>
      </div>
      <div className="card" style={{ marginTop: 12 }}><h3>Training</h3>
        <div className="col">
          <Check checked={!!settings.instant_kill} onChange={(v) => save({ instant_kill: v })}>Instant kill - the Kill Switch stops training without a confirmation dialog</Check>
          <Check checked={!!settings.auto_optimize} onChange={(v) => save({ auto_optimize: v })}>Auto Optimization - allow MakeAI to apply live-adjustable recommendations (DataLoader workers) during training</Check>
        </div>
      </div>
      <div className="card" style={{ marginTop: 12 }}><h3>Sharing &amp; Discover</h3>
        <Field label="Network address when started" hint="restart MakeAI to apply. 0.0.0.0 lets other computers open Link/Public share pages and download packages; everything else stays local-only.">
          <Seg value={settings.share_host === "0.0.0.0" ? "lan" : "local"} onChange={(v) => save({ share_host: v === "lan" ? "0.0.0.0" : "127.0.0.1" })} options={[{ value: "local", label: "This computer only" }, { value: "lan", label: "Local network (share pages)" }]} />
        </Field>
        <div className="lbl" style={{ marginTop: 12 }}>Discover hubs - other MakeAI instances whose public AIs appear in Discover</div>
        {(settings.discover_hubs || []).map((h: string) => <div key={h} className="row small mono">{h}<span className="sp" /><button className="btn ghost sm" onClick={() => save({ discover_hubs: settings.discover_hubs.filter((x: string) => x !== h) })}>Remove</button></div>)}
        <div className="row" style={{ marginTop: 6 }}><input className="in mono" style={{ flex: 1 }} value={hub} onChange={(e) => setHub(e.target.value)} placeholder="http://192.168.1.20:7860" /><button className="btn" disabled={!/^https?:\/\//.test(hub)} onClick={() => { save({ discover_hubs: [...(settings.discover_hubs || []), hub.replace(/\/$/, "")] }); setHub(""); }}>Add hub</button></div>
      </div>
      <div className="card" style={{ marginTop: 12 }}><h3>Appearance</h3>
        <Seg value={theme} onChange={(v) => { setTheme(v); document.documentElement.dataset.theme = v; try { localStorage.setItem("makeai.theme", v); } catch { /* ignore */ } }} options={[{ value: "light", label: "White / lilac" }, { value: "dark", label: "Dark" }]} />
      </div>
      <div className="card" style={{ marginTop: 12 }}><h3>Keyboard shortcuts</h3>
        <dl className="kv">
          <dt><kbd>Ctrl K</kbd></dt><dd>Command palette</dd><dt><kbd>Ctrl J</kbd></dt><dd>Toggle log panel</dd><dt><kbd>Ctrl B</kbd></dt><dd>Collapse sidebar</dd>
          <dt><kbd>Ctrl Alt N</kbd></dt><dd>Create AI</dd><dt><kbd>Alt 1-6</kbd></dt><dd>Dashboard, Hardware, My AIs, Training, Playground, Datasets</dd><dt><kbd>Ctrl Alt W</kbd></dt><dd>Close tab</dd><dt><kbd>Ctrl S</kbd></dt><dd>Save file (Projects)</dd>
        </dl>
      </div>
      <div className="card" style={{ marginTop: 12 }}><h3>About</h3>
        <dl className="kv"><dt>Product</dt><dd>MakeAI {health.data?.version} by Convergent</dd><dt>Data folder</dt><dd>{health.data?.home}</dd>
          <dt>Runtime</dt><dd>local PyTorch engine · llama.cpp · ONNX Runtime - no cloud AI API</dd><dt>Development agent</dt><dd>{health.data?.dev_agent ? "installed (optional)" : "not installed"}</dd></dl>
      </div>
    </div>
  );
}
