import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, fmt, stateLabel } from "../api";
import { Bar, Confirm, Dialog, Icon, LineChart, Num, Spinner } from "../components/ui";
import { Route, useApp } from "../store";
import { StatePill } from "./Dashboard";

type Rec = any;

export default function RunMonitor({ route }: { route: Route }) {
  const runId = route.id!;
  const { subscribe, setRunFeedHandler, telemetry, history, settings, toast, log, open } = useApp();
  const [info, setInfo] = useState<any>(null);
  const [steps, setSteps] = useState<Rec[]>([]);
  const [evals, setEvals] = useState<Rec[]>([]);
  const [events, setEvents] = useState<Rec[]>([]);
  const [status, setStatus] = useState<any>(null);
  const [healthData, setHealth] = useState<any>(null);
  const [alive, setAlive] = useState(false);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [ckpts, setCkpts] = useState<any>(null);
  const [confirmKill, setConfirmKill] = useState(false);
  const [killing, setKilling] = useState<number | null>(null);
  const [workers, setWorkers] = useState<number | null>(null);
  const [showCkpt, setShowCkpt] = useState(false);
  const logCount = useRef(0);
  const lastLine = useRef("");

  const ingest = (recs: Rec[]) => {
    const s = recs.filter((r) => r.type === "step"), e = recs.filter((r) => r.type === "eval"), ev = recs.filter((r) => r.type === "event" || r.type === "checkpoint");
    if (s.length) setSteps((x) => [...x, ...s]);
    if (e.length) setEvals((x) => [...x, ...e]);
    if (ev.length) setEvents((x) => [...x, ...ev]);
  };

  useEffect(() => {
    let alive = true;
    (async () => {
      const i = await api.get(`/api/runs/${runId}`);
      if (!alive) return;
      setInfo(i); setStatus(i.status); setAlive(i.alive); setWorkers(i.config.training.dataloader_workers ?? 2);
      const m = await api.get(`/api/runs/${runId}/metrics?since=0`);
      if (!alive) return;
      setSteps([]); setEvals([]); setEvents([]);
      ingest(m.records);
      subscribe(runId, m.next);
      api.get(`/api/runs/${runId}/health`).then((h) => alive && setHealth(h)).catch(() => {});
    })().catch((e) => toast(e.message, true));
    setRunFeedHandler((feed: any) => {
      ingest(feed.records || []);
      setStatus(feed.status);
      setAlive(feed.alive);
      if (feed.health) setHealth(feed.health);
    });
    return () => { alive = false; setRunFeedHandler(null); subscribe(null); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  // training log tail + checkpoint list
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const l = await api.get(`/api/runs/${runId}/log?lines=400`);
        if (stop) return;
        setLogLines(l.lines);
        const lines: string[] = l.lines;
        const lastSeen = logCount.current ? lines.lastIndexOf(lastLine.current) : -1;
        const fresh = logCount.current && lastSeen < 0 ? lines : lines.slice(lastSeen + 1);
        if (logCount.current || fresh.length < 60) fresh.forEach((line) => log("run", line));
        logCount.current = 1;
        if (lines.length) lastLine.current = lines[lines.length - 1];
        setCkpts(await api.get(`/api/runs/${runId}/checkpoints`));
      } catch { /* run may be gone */ }
      if (!stop) setTimeout(tick, 3000);
    };
    tick();
    return () => { stop = true; };
  }, [runId, log]);

  const last = steps.at(-1);
  const lastEval = evals.at(-1);
  const state = status?.state;
  const active = ["starting", "running", "paused", "stopping"].includes(state) && alive;
  const total = last?.total_steps || status?.total_steps || info?.config?.training?.max_steps || 0;
  const step = last?.step || status?.step || 0;
  const pct = total ? (100 * step) / total : 0;
  const eta = state === "running" ? last?.eta_s : null;
  const g = telemetry?.gpus?.[0];
  const proc = telemetry?.procs?.[runId];

  const act = async (a: string, body: any = {}) => {
    try { await api.post(`/api/runs/${runId}/${a}`, body); toast(a === "kill" ? "Kill Switch activated" : `${a} requested`); }
    catch (e: any) { toast(e.message, true); }
  };
  const kill = async (instant = false) => {
    setKilling(Date.now());
    await act("kill", { instant });
  };
  useEffect(() => { if (!active) setKilling(null); }, [active]);

  const lossSeries = useMemo(() => [
    { name: "train loss", color: "var(--acc)", points: smooth(steps.map((s) => [s.step, s.loss] as [number, number]), 0.85) },
    { name: "raw", color: "var(--acc-dim)", points: steps.length < 400 ? steps.map((s) => [s.step, s.loss] as [number, number]) : [] },
    { name: "val loss", color: "var(--warn)", points: evals.map((e) => [e.step, e.val_loss] as [number, number]) },
  ], [steps, evals]);
  const t0 = history[0]?.t || 0;

  if (!info) return <div className="page"><Spinner /></div>;
  const cfg = info.config;
  const tr = cfg.training;
  const h = healthData?.available ? healthData : null;
  const statusColor = h ? { excellent: "var(--good)", good: "var(--good)", moderate: "var(--warn)", bottleneck: "var(--orange)", critical: "var(--bad)" }[h.status as string] : "var(--faint)";

  const lossPts = lossSeries[0].points;
  const running = state === "running";
  return (
    <div className="page" style={{ maxWidth: 1180 }}>
      <div className="run-head">
        <span className="name">{cfg.model_name}{status?.params ? ` ${fmt.params(status.params)}` : ""}</span>
        <StatePill s={state} />
        <span className="pct">{Math.round(pct)}%</span>
      </div>
      <div className="progress-big" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${pct}%` }} /></div>
      {status?.message && !running && state !== "completed" && (
        <div className={`note ${["failed", "crashed"].includes(state) ? "bad" : ["terminated", "interrupted"].includes(state) ? "warn" : ""}`} style={{ marginTop: 16 }}>
          {status.message}
        </div>
      )}

      <div className="row" style={{ marginTop: 18 }}>
        {state === "paused"
          ? <button className="btn primary" disabled={!active} onClick={() => act("resume")}><Icon n="play" s={14} />Resume</button>
          : <button className="btn" disabled={!active || state !== "running"} onClick={() => act("pause")}><Icon n="pause" s={14} />Pause</button>}
        <button className="kill" disabled={!active || state === "stopping"} onClick={() => (settings.instant_kill ? kill(true) : setConfirmKill(true))} aria-label="Kill switch: stop training">⛔ KILL</button>
        {state === "stopping" && <span className="row small dim"><Spinner /> saving a safe checkpoint…
          {killing && Date.now() - killing > 5000 && <button className="btn sm danger" onClick={() => kill(true)}>Force stop</button>}</span>}
        {!active && ckpts?.latest && ["terminated", "interrupted", "crashed", "failed"].includes(state) && (
          <button className="btn" onClick={() => act("restart")}>Continue from last checkpoint</button>
        )}
        <span className="sp" />
        {state === "completed" && <button className="btn primary" onClick={() => open({ view: "playground", id: cfg.model_uid, title: `Chat · ${cfg.model_name}` })}><Icon n="play" s={14} />Chat with it</button>}
        <button className="btn ghost" onClick={() => open({ view: "ai", id: cfg.model_uid, title: cfg.model_name })}>Open AI</button>
      </div>

      <div className="chart-card" style={{ marginTop: 20 }}>
        <div className="row" style={{ marginBottom: 4 }}>
          <span className="cap" style={{ margin: 0 }}>training loss</span>
          {evals.length > 0 && <span className="legend" style={{ marginLeft: "auto" }}><span><i style={{ background: "var(--acc)" }} />train</span><span><i style={{ background: "var(--warn)" }} />validation</span></span>}
        </div>
        {lossPts.length > 1
          ? <LineChart height={280} xFmt={(x) => fmt.int(x)} series={[lossSeries[0], lossSeries[2]]} />
          : <div className="empty" style={{ height: 280, display: "grid", placeItems: "center", border: 0 }}>{status?.message || "Waiting for the first steps…"}</div>}
      </div>

      <div className="grid g4" style={{ marginTop: 16 }}>
        <BigStat k="Loss" v={fmt.num(last?.loss, 3)} sub={lastEval ? `val ${fmt.num(lastEval.val_loss, 3)}` : undefined} />
        {active ? (
          <>
            <BigStat k="GPU" v={g?.util_pct != null ? `${g.util_pct}%` : "–"} sub={g ? `${fmt.mb(g.vram_used_mb)} / ${fmt.mb(g.vram_total_mb)} VRAM` : undefined} />
            <BigStat k="Temp" v={g?.temp_c != null ? `${g.temp_c}°C` : "–"} sub={g?.power_w != null ? `${Math.round(g.power_w)} W` : undefined} />
            <BigStat k="Time left" v={eta != null ? fmt.dur(eta) : "–"} sub={eta != null ? `done ~${fmt.clock(Date.now() / 1000 + eta)} · ${fmt.int(last?.tokens_per_s)} tok/s` : undefined} />
          </>
        ) : (
          <>
            <BigStat k="Best val loss" v={fmt.num(status?.best_val_loss, 3)} sub={lastEval ? `token accuracy ${(100 * lastEval.token_accuracy).toFixed(1)}%` : undefined} />
            <BigStat k="Duration" v={fmt.dur(last?.elapsed)} sub={status?.ended_at ? `ended ${fmt.clock(status.ended_at)}` : undefined} />
            <BigStat k="Tokens/sec" v={fmt.int(last?.tokens_per_s)} sub={last?.mem_peak_mb ? `peak memory ${fmt.mb(last.mem_peak_mb)}` : undefined} />
          </>
        )}
      </div>

      {h && (
        <div className="card row" style={{ marginTop: 16, gap: 20, alignItems: "center" }}>
          <div className="score">{h.score.toFixed(1)}<small> / 10</small></div>
          <div style={{ flex: 1, minWidth: 220 }}>
            <div className="status-chip" style={{ color: statusColor }}><span className="dot" style={{ background: statusColor }} />Training performance: {h.status_label}</div>
            <div className="dim small" style={{ marginTop: 4 }}>{h.reason}{h.recommendation ? ` ${h.recommendation}` : ""}</div>
          </div>
        </div>
      )}

      <details className="more">
        <summary>Details</summary>
        <div className="col" style={{ gap: 16 }}>
          <div className="card">
            <h3>All numbers</h3>
            <div className="grid g4">
              <Stat k="Epoch" v={`${last?.epoch ?? status?.epoch ?? 0} / ${status?.total_epochs ?? tr.epochs ?? "?"}`} />
              <Stat k="Step" v={`${fmt.int(step)} / ${fmt.int(total)}`} />
              <Stat k="Perplexity" v={fmt.num(lastEval?.val_ppl ?? last?.ppl, 2)} />
              <Stat k="Token accuracy" v={lastEval ? `${(100 * lastEval.token_accuracy).toFixed(1)}%` : "–"} />
              <Stat k="Learning rate" v={fmt.lr(last?.lr)} />
              <Stat k="Grad norm" v={fmt.num(last?.grad_norm, 2)} />
              <Stat k="Steps/sec" v={fmt.num(last?.steps_per_s, 2)} />
              <Stat k="Samples/sec" v={fmt.num(last?.samples_per_s, 1)} />
              <Stat k="Tokens seen" v={fmt.params(last?.tokens_seen)} />
              <Stat k="Elapsed" v={fmt.dur(last?.elapsed)} />
              <Stat k="Run memory" v={last?.mem_peak_mb ? fmt.mb(last.mem_peak_mb) : "–"} />
              <Stat k="CPU / RAM" v={telemetry ? `${Math.round(telemetry.cpu_pct)}% / ${Math.round(telemetry.ram_pct)}%` : "–"} sub={proc ? `trainer ${fmt.mb(proc.rss_mb)}` : undefined} />
            </div>
          </div>
          <div className="grid g2">
            <div className="card"><h3>Speed</h3>
              <LineChart height={140} xFmt={(x) => fmt.int(x)} series={[{ name: "tokens/s", color: "var(--blue)", points: smooth(steps.map((s) => [s.step, s.tokens_per_s] as [number, number]), 0.8) }]} />
            </div>
            <div className="card"><h3>Hardware (last minutes)</h3>
              <LineChart height={140} yMin={0} yMax={100} xFmt={(x) => `${Math.round(x)}s`} series={[
                { name: "GPU %", color: "var(--acc)", points: history.map((s) => [s.t - t0, s.gpus?.[0]?.util_pct ?? null] as [number, number]) },
                { name: "GPU °C", color: "var(--orange)", points: history.map((s) => [s.t - t0, s.gpus?.[0]?.temp_c ?? null] as [number, number]) },
              ]} />
            </div>
          </div>
          {h && (
            <div className="card"><h3>Performance breakdown<span className="r">technical health of this run, not model quality</span></h3>
              <div className="col" style={{ gap: 8 }}>
                {Object.entries(h.components).map(([k, c]: any) => (
                  <div className="comp" key={k}><span className="dim">{c.label}</span><Bar v={c.score * 10} color={c.score < 5 ? "var(--warn)" : undefined} /><span className="mono small" style={{ textAlign: "right" }}>{c.score.toFixed(1)}</span></div>
                ))}
              </div>
              {h.issues.length > 1 && <div className="small dim" style={{ marginTop: 10 }}>{h.issues.slice(1).map((i: any, n: number) => <div key={n}>• {i.reason} {i.recommendation}</div>)}</div>}
            </div>
          )}
          <div className="grid g2">
            <div className="card"><h3>Checkpoints<span className="r"><button className="btn sm" disabled={!active} onClick={() => act("checkpoint")}>Save now</button></span></h3>
              <table className="t"><tbody>
                {ckpts?.checkpoints?.slice().reverse().map((c: any) => (
                  <tr key={c.name}>
                    <td>{c.kind}{ckpts.best?.name === c.name && <span className="pill good" style={{ marginLeft: 6 }}>best</span>}</td>
                    <td className="num">step {fmt.int(c.step)}</td><td className="num">{fmt.bytes(c.bytes)}</td>
                    <td style={{ whiteSpace: "nowrap", textAlign: "right" }}>
                      <button className="btn sm ghost" onClick={async () => { const r = await api.post(`/api/runs/${runId}/checkpoints/${c.name}/backup`); toast(`Backup saved: ${r.path}`); }}>Backup</button>
                      {!active && <button className="btn sm" onClick={() => act("restart", { checkpoint: c.name })}>Resume</button>}
                    </td>
                  </tr>
                ))}
                {!ckpts?.checkpoints?.length && <tr><td className="dim">No checkpoints yet.</td></tr>}
              </tbody></table>
            </div>
            <div className="card"><h3>Log</h3>
              <div className="logs-body" style={{ height: 220 }}>
                {logLines.slice(-150).map((l, i) => <div key={i} className={/FAILED|TERMINATED|HARD KILL/.test(l) ? "l-err" : /complete|resumed/.test(l) ? "l-ok" : undefined}>{l}</div>)}
              </div>
            </div>
          </div>
          <div className="card row">
            <span className="dim small">DataLoader workers</span>
            <div style={{ width: 80 }}><Num value={workers} onChange={setWorkers} min={1} max={32} /></div>
            <button className="btn sm" disabled={!active} onClick={() => act("set", { dataloader_workers: workers })}>Apply</button>
            <span className="sp" />
            <button className="btn ghost sm" onClick={() => setShowCkpt(true)}>Run configuration</button>
          </div>
        </div>
      </details>

      <Confirm open={confirmKill} onClose={() => setConfirmKill(false)} danger title="Stop training?" confirmLabel="⛔ Stop training"
        onConfirm={() => kill(false)}
        body={<div className="col">
          <div>Training stops now. MakeAI saves a safe checkpoint, frees the GPU and marks the run <b>TERMINATED BY USER</b>.</div>
          <div className="dim small">You can continue later from that checkpoint. Instant stop without this question can be turned on in Settings.</div>
        </div>} />
      <Dialog open={showCkpt} onClose={() => setShowCkpt(false)} title="Run configuration" wide>
        <pre className="mono small" style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(cfg, null, 2)}</pre>
      </Dialog>
    </div>
  );
}

function BigStat({ k, v, sub }: { k: string; v: React.ReactNode; sub?: string }) {
  return <div className="stat-card"><div className="stat"><span className="k">{k}</span><span className="v">{v}</span>{sub && <span className="faint small">{sub}</span>}</div></div>;
}

function Stat({ k, v, sub, bar }: { k: string; v: React.ReactNode; sub?: string; bar?: number | null }) {
  return <div className="stat"><span className="k">{k}</span><span className="v" style={{ fontSize: 16 }}>{v}</span>{bar != null && <Bar v={bar} />}{sub && <span className="faint small">{sub}</span>}</div>;
}

function smooth(pts: [number, number][], a: number): [number, number][] {
  let e: number | null = null;
  return pts.map(([x, y]) => {
    if (y == null || !isFinite(y)) return [x, e ?? NaN];
    e = e == null ? y : a * e + (1 - a) * y;
    return [x, e];
  });
}

export { stateLabel };
