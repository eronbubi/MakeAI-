import React, { useEffect, useRef, useState } from "react";
import { api, fmt, streamPost } from "../api";
import { Check, Field, Icon, Num, Seg, Sel, Spinner } from "../components/ui";
import { Route, useApp, useFetch } from "../store";

type Msg = { role: "user" | "assistant" | "system"; content: string; meta?: any };
const DEFAULTS = { temperature: 0.7, top_k: 40, top_p: 0.95, min_p: 0.05, repetition_penalty: 1.1, frequency_penalty: 0, presence_penalty: 0, max_tokens: 512, seed: null as number | null };

export default function Playground({ route }: { route: Route }) {
  const { toast } = useApp();
  const models = useFetch<any[]>("/api/models");
  const runnable = (models.data || []).filter((m) => m.runnable);
  const [uid, setUid] = useState<string>(route.id || "");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [p, setP] = useState(DEFAULTS);
  const [system, setSystem] = useState("");
  const [stop, setStop] = useState("");
  const [streaming, setStreaming] = useState(true);
  const [mode, setMode] = useState<"chat" | "complete">("chat");
  const [quant, setQuant] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const model = runnable.find((m) => m.uid === uid);

  useEffect(() => { if (!uid && runnable.length) setUid(runnable[0].uid); }, [runnable.length]); // eslint-disable-line
  useEffect(() => { endRef.current?.scrollIntoView({ block: "end" }); }, [msgs]);
  useEffect(() => { api.get("/api/playground/status").then(setStatus).catch(() => {}); }, [uid]);

  const load = async () => {
    setLoading(true);
    try { setStatus(await api.post("/api/playground/load", { uid, quant: quant || null })); }
    catch (e: any) { toast(e.message, true); } finally { setLoading(false); }
  };

  const send = async () => {
    if (!uid || busy || (!input.trim() && mode === "chat")) return;
    const history: Msg[] = mode === "chat" ? [...msgs.filter((m) => m.role !== "system"), { role: "user", content: input }] : [];
    const prompt = mode === "complete" ? input : undefined;
    setMsgs(mode === "chat" ? [...history, { role: "assistant", content: "" }] : [{ role: "user", content: input }, { role: "assistant", content: "" }]);
    if (mode === "chat") setInput("");
    setBusy(true);
    abort.current = new AbortController();
    let text = "";
    const params = { ...p, stop: stop.split("\\n").join("\n").split("|").map((s) => s).filter(Boolean) };
    try {
      await streamPost("/api/playground/chat", {
        uid, messages: history.map(({ role, content }) => ({ role, content })), params, system_prompt: system || null,
        raw_prompt: prompt, quant: quant || null,
      }, (d) => {
        if (d.error) throw new Error(d.error);
        if (d.info) toast(d.info);
        if (d.delta) {
          text += d.delta;
          if (streaming) setMsgs((m) => { const c = m.slice(); c[c.length - 1] = { role: "assistant", content: text }; return c; });
        }
        if (d.done) setMsgs((m) => { const c = m.slice(); c[c.length - 1] = { role: "assistant", content: d.text ?? text, meta: d }; return c; });
      }, abort.current.signal);
      api.get("/api/playground/status").then(setStatus).catch(() => {});
    } catch (e: any) {
      if (e.name !== "AbortError") toast(e.message, true);
    } finally { setBusy(false); }
  };
  const halt = () => { api.post("/api/playground/stop"); abort.current?.abort(); };
  const setK = (k: keyof typeof DEFAULTS, v: any) => setP({ ...p, [k]: v });

  return (
    <div className="pg">
      <div className="chat">
        <div className="row" style={{ padding: "8px 16px", borderBottom: "1px solid var(--line)" }}>
          <div style={{ width: 300 }}><Sel value={uid} onChange={(v) => { setUid(v); setMsgs([]); }} options={[{ value: "", label: runnable.length ? "Choose an AI…" : "No runnable AI yet" }, ...runnable.map((m) => ({ value: m.uid, label: `${m.name} ${m.version} · ${fmt.params(m.param_count)}` }))]} /></div>
          <Seg value={mode} onChange={(v) => { setMode(v); setMsgs([]); }} options={[{ value: "chat", label: "Chat" }, { value: "complete", label: "Completion" }]} />
          {model && <span className="dim small">by {model.creator.name} @{model.creator.username}</span>}
          <span className="sp" />
          {status?.loaded && status.uid === uid ? <span className="pill good">{status.backend} · {status.device}{status.quant ? ` · ${status.quant}` : ""} · {status.dtype}{status.vram_allocated_mb ? ` · ${fmt.mb(status.vram_allocated_mb)}` : ""}</span>
            : <button className="btn sm" disabled={!uid || loading} onClick={load}>{loading ? <Spinner /> : "Load"}</button>}
          {status?.loaded && <button className="btn sm ghost" onClick={async () => { await api.post("/api/playground/unload"); setStatus({ loaded: false }); }}>Unload</button>}
          <button className="btn sm ghost" onClick={() => setMsgs([])}>Clear</button>
        </div>
        <div className="msgs">
          {!msgs.length && <div className="empty" style={{ maxWidth: 560, margin: "40px auto" }}>
            {model ? <>Everything runs on this computer with {model.backend || "the MakeAI engine"} - no Claude, no cloud.<br />{mode === "chat" ? "Uses the model's own chat template." : "Completion mode continues your text exactly."}</> : "Choose an AI to chat with."}
          </div>}
          {msgs.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <div className="who">{m.role === "user" ? (mode === "chat" ? "You" : "Prompt") : model?.name || "AI"}</div>
              <div className="txt">{m.content || (busy && i === msgs.length - 1 ? <Spinner /> : "")}</div>
              {m.meta && <div className="meta">{m.meta.completion_tokens} tokens · {m.meta.tokens_per_s ?? "–"} tok/s · first token {Math.round((m.meta.time_to_first_token_s || 0) * 1000)} ms · {m.meta.finish_reason}</div>}
            </div>
          ))}
          <div ref={endRef} />
        </div>
        <div className="composer">
          <div className="box">
            <textarea className="in" rows={mode === "chat" ? 2 : 5} value={input} placeholder={mode === "chat" ? "Message… (Enter to send, Shift+Enter for a new line)" : "Text to continue…"}
              onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && mode === "chat") { e.preventDefault(); send(); } }} aria-label="Message" />
            {busy ? <button className="btn danger" onClick={halt}><Icon n="stop" s={13} />Stop</button>
              : <button className="btn primary" disabled={!uid || (!input.trim())} onClick={send}>Send</button>}
          </div>
        </div>
      </div>
      <div className="pg-side">
        <Field label="System prompt"><textarea className="in" rows={3} value={system} onChange={(e) => setSystem(e.target.value)} placeholder="You are a helpful assistant." disabled={mode !== "chat"} /></Field>
        <Slider label="Temperature" v={p.temperature} min={0} max={2} step={0.05} on={(v) => setK("temperature", v)} hint="0 = greedy" />
        <Slider label="Top-K" v={p.top_k} min={0} max={200} step={1} on={(v) => setK("top_k", v)} hint="0 = off" />
        <Slider label="Top-P" v={p.top_p} min={0} max={1} step={0.01} on={(v) => setK("top_p", v)} />
        <Slider label="Min-P" v={p.min_p} min={0} max={0.5} step={0.01} on={(v) => setK("min_p", v)} />
        <Slider label="Repetition penalty" v={p.repetition_penalty} min={1} max={2} step={0.01} on={(v) => setK("repetition_penalty", v)} />
        <Slider label="Frequency penalty" v={p.frequency_penalty} min={0} max={2} step={0.05} on={(v) => setK("frequency_penalty", v)} />
        <Slider label="Presence penalty" v={p.presence_penalty} min={0} max={2} step={0.05} on={(v) => setK("presence_penalty", v)} />
        <div className="grid g2">
          <Field label="Max tokens"><Num value={p.max_tokens} onChange={(v) => setK("max_tokens", v)} min={1} /></Field>
          <Field label="Seed" hint="empty = random"><input className="in mono" value={p.seed ?? ""} onChange={(e) => setK("seed", e.target.value === "" ? null : Number(e.target.value))} /></Field>
        </div>
        <Field label="Context" hint="the model's trained context; older turns are dropped to fit"><input className="in mono" disabled value={status?.uid === uid ? status.context_length : model?.context_length ?? ""} /></Field>
        <Field label="Stop sequences" hint="separate with |"><input className="in mono" value={stop} onChange={(e) => setStop(e.target.value)} placeholder="###|\nUser:" /></Field>
        <Check checked={streaming} onChange={setStreaming}>Streaming</Check>
        <Field label="Load quantized" hint="MakeAI engine only; needs bitsandbytes + CUDA">
          <Sel value={quant} onChange={(v) => { setQuant(v); setStatus(null); }} options={[{ value: "", label: "Full precision (bf16/fp16)" }, { value: "int8", label: "INT8" }, { value: "nf4", label: "NF4 (4-bit)" }, { value: "fp4", label: "FP4 (4-bit)" }]} />
        </Field>
        <button className="btn ghost sm" onClick={() => { setP(DEFAULTS); setStop(""); }}>Reset controls</button>
      </div>
    </div>
  );
}

function Slider({ label, v, min, max, step, on, hint }: { label: string; v: number; min: number; max: number; step: number; on: (v: number) => void; hint?: string }) {
  return (
    <div className="field">
      <label>{label}<span className="sp" /><span className="mono">{v}</span></label>
      <input type="range" min={min} max={max} step={step} value={v} onChange={(e) => on(Number(e.target.value))} aria-label={label} />
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}
