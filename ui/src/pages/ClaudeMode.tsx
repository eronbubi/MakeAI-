import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, fmt } from "../api";
import { Dialog, Icon, LineChart, Spinner } from "../components/ui";
import { useApp } from "../store";

type Ev = { id: number; t: number; kind: "system" | "say" | "action" | "event"; text: string; icon?: string; status?: string; detail?: string };

/** Claude Mode: a coding agent (Claude Code, Codex, Cursor, ...) works in MakeAI, you only watch.
 *  You talk to the agent in its own app. */
export default function ClaudeMode({ onExit }: { pages?: unknown; onExit: () => void }) {
  const [events, setEvents] = useState<Ev[]>([]);
  const [state, setState] = useState<any>(null);
  const [help, setHelp] = useState(false);
  const since = useRef(0);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const s = await api.get(`/api/claude/state?since=${since.current}`);
        if (stop) return;
        setState(s);
        if (s.events.length) {
          since.current = s.events[s.events.length - 1].id;
          setEvents((e) => {
            const byId = new Map(e.map((x) => [x.id, x]));
            s.events.forEach((x: Ev) => byId.set(x.id, x));
            return [...byId.values()].sort((a, b) => a.id - b.id).slice(-300);
          });
        }
      } catch { /* server restarting */ }
      if (!stop) setTimeout(tick, 1000);
    };
    tick();
    return () => { stop = true; };
  }, []);
  useEffect(() => { const el = listRef.current; if (el) el.scrollTop = el.scrollHeight; }, [events.length]);

  // only this session's activity; hide technical noise
  const start = state?.session?.started_at || 0;
  const shown = events.filter((e) => e.t >= start - 1 && !(e.kind === "action" && e.status === "running"));
  const focus = state?.focus;
  const connected = state?.connected;
  const agent: string = state?.agent?.name || "Claude";

  return (
    <div className="cm">
      <aside className="cm-left" aria-label={`${agent} activity`}>
        <div className="cm-left-head">
          <Icon n="spark" s={22} c="var(--acc)" /><h2>{agent} activity</h2>
          <span className="cm-lock"><Icon n="lock" s={13} />VIEW ONLY</span>
        </div>
        <div className="cm-list" ref={listRef}>
          {!shown.length && (
            <div className="cm-item say">
              <span className="tx">No activity yet. Ask your agent (Claude Code, Codex, Cursor, …) in its own app to work in MakeAI - its steps appear here.</span>
            </div>
          )}
          {shown.map((e, i) => <Item key={e.id} e={e} latest={i === shown.length - 1} />)}
        </div>
        <div className="cm-foot">
          <span className={`dot ${connected ? "run" : ""}`} />
          {state?.session?.active ? (connected ? `${agent} is working` : `${agent} is thinking`) : connected ? `${agent} finished` : "No agent connected"}
          <span className="sp" />
          <button className="btn ghost sm" onClick={() => setHelp(true)}>How to connect</button>
          <button className="btn sm" onClick={onExit}>Exit</button>
        </div>
      </aside>
      <section className="cm-right" aria-label={`What ${agent} is working on`} {...({ inert: "" } as any)}>
        {focus?.view === "run" && focus.id ? <RunView runId={focus.id} />
          : focus?.view === "ai" && focus.id ? <AiView uid={focus.id} />
          : focus?.view === "dataset" && focus.id ? <DatasetView id={focus.id} />
          : <Idle agent={agent} connected={connected} goal={state?.session?.active ? state?.session?.goal : null} />}
      </section>
      {help && <ConnectDialog onClose={() => setHelp(false)} />}
    </div>
  );
}

function Item({ e, latest }: { e: Ev; latest: boolean }) {
  const icon = e.icon || (e.kind === "say" ? "spark" : e.kind === "event" ? "check" : e.kind === "system" ? "spark" : "play");
  return (
    <div className={`cm-item${latest ? " latest" : ""}${e.status === "error" ? " err" : ""}${e.kind === "say" ? " say" : ""}`}
      title={new Date(e.t * 1000).toLocaleTimeString()}>
      <span className="ic"><Icon n={e.status === "error" ? "x" : icon} s={17} /></span>
      <div className="tx">{e.text}{e.detail && e.status === "error" && <div className="sub">{e.detail}</div>}</div>
    </div>
  );
}

function Idle({ agent, connected, goal }: { agent: string; connected: boolean; goal?: string | null }) {
  return (
    <div className="cm-idle">
      <div style={{ maxWidth: 440 }}>
        <Icon n="spark" s={40} c="var(--acc)" />
        <h2>{goal ? `${agent} is getting started` : connected ? `${agent} is connected` : "Waiting for an agent"}</h2>
        <div>{goal ? goal : "When your agent creates or trains an AI, you see it here live - progress, loss curve, GPU and temperature."}</div>
      </div>
    </div>
  );
}

function RunView({ runId }: { runId: string }) {
  const { telemetry } = useApp();
  const [info, setInfo] = useState<any>(null);
  const [steps, setSteps] = useState<any[]>([]);
  const next = useRef(0);
  useEffect(() => {
    let stop = false;
    next.current = 0;
    setSteps([]);
    const tick = async () => {
      try {
        const [i, m] = await Promise.all([api.get(`/api/runs/${runId}`), api.get(`/api/runs/${runId}/metrics?since=${next.current}`)]);
        if (stop) return;
        setInfo(i);
        next.current = m.next;
        const st = m.records.filter((r: any) => r.type === "step");
        if (st.length) setSteps((s) => [...s, ...st]);
      } catch { /* ignore */ }
      if (!stop) setTimeout(tick, 2000);
    };
    tick();
    return () => { stop = true; };
  }, [runId]);
  const last = steps[steps.length - 1];
  const status = info?.status || {};
  const total = last?.total_steps || status.total_steps || info?.config?.training?.max_steps || 0;
  const step = last?.step || status.step || 0;
  const pct = total ? (100 * step) / total : 0;
  const pts = useMemo(() => {
    let e: number | null = null;
    return steps.map((s) => { e = e == null ? s.loss : 0.96 * e + 0.04 * s.loss; return [s.step, e] as [number, number]; });
  }, [steps]);
  const g = telemetry?.gpus?.[0];
  const params = status.params ? ` ${fmt.params(status.params)}` : "";
  const running = status.state === "running";
  return (
    <div>
      <div className="run-head">
        <span className="name">{info?.config?.model_name || "…"}{params}</span>
        <span className="pct">{Math.round(pct)}%</span>
      </div>
      <div className="progress-big"><i style={{ width: `${pct}%` }} /></div>
      <div className="chart-card" style={{ marginTop: 22 }}>
        <div className="cap">training loss</div>
        {pts.length > 1 ? <LineChart bare height={300} series={[{ name: "loss", color: "var(--acc)", points: pts }]} />
          : <div className="cm-idle" style={{ height: 300 }}>{status.message || "Waiting for the first steps…"}</div>}
      </div>
      <div className="grid g3" style={{ marginTop: 16 }}>
        {running || status.state === "starting" || status.state === "paused" ? (
          <>
            <Card k="GPU" v={g?.util_pct != null ? `${g.util_pct}%` : "–"} />
            <Card k="Temp" v={g?.temp_c != null ? `${g.temp_c}°C` : "–"} />
            <Card k="Tokens/sec" v={last ? fmt.int(last.tokens_per_s) : "–"} />
          </>
        ) : (
          <>
            <Card k="Final loss" v={fmt.num(last?.loss, 3)} />
            <Card k="Best val loss" v={fmt.num(status.best_val_loss, 3)} />
            <Card k="Duration" v={fmt.dur(last?.elapsed)} />
          </>
        )}
      </div>
      <div className="row small dim" style={{ marginTop: 14, gap: 18 }}>
        <span>Step {fmt.int(step)} / {fmt.int(total)}</span>
        {last && <span>Loss {fmt.num(last.loss, 3)}</span>}
        {status.best_val_loss != null && <span>Best val loss {fmt.num(status.best_val_loss, 3)}</span>}
        {running && last?.eta_s != null && <span>Done in ~{fmt.dur(last.eta_s)}</span>}
        {!running && status.state && <span>{status.state === "completed" ? "Finished" : status.state}</span>}
      </div>
    </div>
  );
}

function Card({ k, v }: { k: string; v: string }) {
  return <div className="stat-card"><div className="stat"><span className="k mono" style={{ fontSize: 12 }}>{k}</span><span className="v">{v}</span></div></div>;
}

function AiView({ uid }: { uid: string }) {
  const [m, setM] = useState<any>(null);
  useEffect(() => {
    let stop = false;
    const t = () => api.get(`/api/models/${uid}`).then((x) => { if (!stop) setM(x); }).catch(() => {});
    t();
    const i = setInterval(t, 4000);
    return () => { stop = true; clearInterval(i); };
  }, [uid]);
  if (!m) return <Spinner />;
  return (
    <div>
      <div className="run-head"><span className="name">{m.name}</span><span className="pct" style={{ fontSize: 15 }}>{m.status}</span></div>
      <div className="dim" style={{ marginBottom: 20 }}>{m.description || "No description"} · by {m.creator.name}</div>
      <div className="grid g3">
        <Card k="Parameters" v={fmt.params(m.param_count)} />
        <Card k="Context" v={m.context_length ? String(m.context_length) : "–"} />
        <Card k="Val loss" v={m.last_eval ? fmt.num(m.last_eval.val_loss, 3) : "–"} />
      </div>
    </div>
  );
}

function DatasetView({ id }: { id: string }) {
  const [d, setD] = useState<any>(null);
  useEffect(() => { api.get(`/api/datasets/${id}`).then(setD).catch(() => {}); }, [id]);
  if (!d) return <Spinner />;
  const s = d.stats || {};
  return (
    <div>
      <div className="run-head"><span className="name">{d.name}</span></div>
      <div className="grid g3" style={{ marginTop: 8 }}>
        <Card k="Samples" v={fmt.int(s.samples)} />
        <Card k="Tokens" v={s.tokens ? fmt.params(s.tokens) : "–"} />
        <Card k="Size" v={fmt.bytes(s.bytes)} />
      </div>
    </div>
  );
}

type AgentInfo = { id: string; name: string; installed: boolean; configured: boolean; config_path: string | null;
  restart: string; snippet: string; docs: string; kind: string; last_seen: number | null };

function ConnectDialog({ onClose }: { onClose: () => void }) {
  const { toast } = useApp();
  const [data, setData] = useState<{ agents: AgentInfo[]; prompt_template: string } | null>(null);
  const [goal, setGoal] = useState("Train a small code model on my Python dataset and show me how good it gets.");
  const [busy, setBusy] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const load = () => api.get("/api/claude/agents").then(setData).catch((e) => toast(e.message, true));
  useEffect(() => { load(); }, []); // eslint-disable-line
  const prompt = (data?.prompt_template || "").replace("{goal}", goal || "(ask me)");
  const copy = (t: string) => navigator.clipboard?.writeText(t).then(() => toast("Copied"));
  const act = async (a: AgentInfo, what: "connect" | "disconnect") => {
    setBusy(a.id);
    try {
      await api.post(`/api/claude/agents/${a.id}/${what}`);
      toast(what === "connect" ? `${a.name} linked. ${a.restart}` : `${a.name} unlinked`);
      await load();
    } catch (e: any) { toast(e.message, true); } finally { setBusy(null); }
  };
  const list = data ? [...data.agents].sort((x, y) => Number(y.installed) - Number(x.installed)) : [];
  return (
    <Dialog open onClose={onClose} wide title="Connect an agent" footer={<button className="btn" onClick={onClose}>Close</button>}>
      {!data ? <Spinner /> : (
        <div className="col" style={{ gap: 18 }}>
          <div>
            <div className="lbl" style={{ marginBottom: 8 }}>1 · Link MakeAI to your coding agent (once)</div>
            <div className="agent-list">
              {list.map((a) => (
                <div key={a.id} className={`agent-row${a.installed ? "" : " off"}`}>
                  <div className="agent-main">
                    <span className="agent-name">{a.name}</span>
                    <span className={`agent-state${a.configured ? " ok" : ""}`}>
                      {a.configured ? (a.last_seen ? `Linked · last used ${new Date(a.last_seen * 1000).toLocaleString()}` : "Linked")
                        : a.installed ? "Not linked" : "Not installed"}
                    </span>
                    <span className="sp" />
                    <button className="btn ghost sm" onClick={() => setOpen(open === a.id ? null : a.id)}>{open === a.id ? "Hide" : "Manual"}</button>
                    {a.configured
                      ? <button className="btn sm" disabled={busy === a.id} onClick={() => act(a, "disconnect")}>Unlink</button>
                      : <button className="btn primary sm" disabled={busy === a.id || !a.installed} onClick={() => act(a, "connect")}>
                          {busy === a.id ? <Spinner /> : "Link"}</button>}
                  </div>
                  {open === a.id && (
                    <div className="agent-manual small">
                      {a.kind === "aider" ? "Aider has no MCP support - MakeAI gives it a conventions file with the commands instead:"
                        : a.kind === "cli" ? "Run this once in a terminal:" : <>Add this to <span className="mono">{a.config_path}</span>:</>}
                      <pre className="cm-pre" style={{ maxHeight: 180, overflow: "auto" }}>{a.snippet}</pre>
                      <div className="row" style={{ gap: 8 }}>
                        <button className="btn sm" onClick={() => copy(a.snippet)}>Copy</button>
                        <span className="dim">{a.restart}</span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="lbl" style={{ marginBottom: 6 }}>2 · Open a new chat in that agent and send this</div>
            <textarea className="in" rows={2} value={goal} onChange={(e) => setGoal(e.target.value)} aria-label="Goal" />
            <pre className="cm-pre" style={{ maxHeight: 200, overflow: "auto" }}>{prompt}</pre>
            <button className="btn primary" onClick={() => copy(prompt)}>Copy prompt</button>
          </div>
          <div className="small dim">You talk to the agent in its own app. MakeAI only shows what it does.</div>
        </div>
      )}
    </Dialog>
  );
}
