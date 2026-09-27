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

  return (
    <div className="page wide">
      <div className="head">
        <h1 className="title">Training: {cfg.model_name}</h1>
        <StatePill s={state} />
        <span className="sub mono small">{runId} · {cfg.method}</span>
        <div className="actions">
          <button className="btn" onClick={() => open({ view: "ai", id: cfg.model_uid, title: cfg.model_name })}>Open AI</button>
          {state === "completed" && <button className="btn primary" onClick={() => open({ view: "playground", id: cfg.model_uid, title: `Chat · ${cfg.model_name}` })}><Icon n="play" s={14} />RUN</button>}
        </div>
      </div>
      {status?.message && <div className={`note ${["failed", "crashed"].includes(state) ? "bad" : state === "terminated" || state === "interrupted" ? "warn" : ""}`} style={{ marginBottom: 12 }}>{status.message}{status.stdout_tail && <pre className="mono small" style={{ whiteSpace: "pre-wrap", margin: "6px 0 0" }}>{status.stdout_tail.slice(-600)}</pre>}</div>}

      <div className="split" style={{ gridTemplateColumns: "minmax(0,1fr) 330px" }}>
        <div className="col" style={{ gap: 12 }}>
          <div className="card">
            <div className="row" style={{ marginBottom: 8 }}>
              <b>{cfg.model_name}</b><span className="dim">{fmt.params(status?.params)} parameters{status?.trainable_params && status.trainable_params !== status.params ? ` · ${fmt.params(status.trainable_params)} trainable` : ""}</span>
              <span className="sp" /><span className="mono" style={{ fontSize: 18 }}>{pct.toFixed(1)}%</span>
            </div>
            <div className="progress-big" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${pct}%` }} /></div>
            <div className="grid g4" style={{ marginTop: 14 }}>
              <Stat k="Epoch" v={`${last?.epoch ?? status?.epoch ?? 0} / ${status?.total_epochs ?? tr.epochs ?? "?"}`} />
              <Stat k="Step" v={`${fmt.int(step)} / ${fmt.int(total)}`} />
              <Stat k="Loss" v={fmt.num(last?.loss, 3)} />
              <Stat k="Validation loss" v={fmt.num(lastEval?.val_loss, 3)} sub={status?.best_val_loss != null ? `best ${fmt.num(status.best_val_loss, 3)}` : undefined} />
              <Stat k="Perplexity" v={fmt.num(lastEval?.val_ppl ?? last?.ppl, 2)} sub={lastEval ? "validation" : "train"} />
              <Stat k="Token accuracy" v={lastEval ? `${(100 * lastEval.token_accuracy).toFixed(1)}%` : "–"} />
              <Stat k="Learning rate" v={fmt.lr(last?.lr)} />
              <Stat k="Grad norm" v={fmt.num(last?.grad_norm, 2)} />
              <Stat k="Tokens/sec" v={fmt.int(last?.tokens_per_s)} />
              <Stat k="Steps/sec" v={fmt.num(last?.steps_per_s, 2)} />
              <Stat k="Samples/sec" v={fmt.num(last?.samples_per_s, 1)} />
              <Stat k="Tokens seen" v={fmt.params(last?.tokens_seen)} />
              <Stat k="Elapsed" v={fmt.dur(last?.elapsed)} />
              <Stat k="Remaining" v={eta != null ? fmt.dur(eta) : "–"} sub="from measured speed" />
              <Stat k="ETA" v={eta != null ? `${Math.round(eta / 60)} min` : "–"} />
              <Stat k="Est. completion" v={eta != null ? fmt.clock(Date.now() / 1000 + eta) : state === "completed" ? fmt.clock(status?.ended_at) : "–"} />
            </div>
          </div>

          <div className="grid g2">
            <div className="card"><h3>Loss<span className="r legend"><span><i style={{ background: "var(--acc)" }} />train (smoothed)</span><span><i style={{ background: "var(--warn)" }} />validation</span></span></h3>
              <LineChart series={lossSeries} height={200} xFmt={(x) => fmt.int(x)} />
            </div>
            <div className="card"><h3>Throughput &amp; learning rate<span className="r legend"><span><i style={{ background: "var(--blue)" }} />tokens/s</span></span></h3>
              <LineChart series={[{ name: "tokens/s", color: "var(--blue)", points: smooth(steps.map((s) => [s.step, s.tokens_per_s] as [number, number]), 0.8) }]} height={120} xFmt={(x) => fmt.int(x)} />
              <LineChart series={[{ name: "lr", color: "var(--violet)", points: steps.map((s) => [s.step, s.lr] as [number, number]) }]} height={78} xFmt={(x) => fmt.int(x)} yFmt={(y) => y.toExponential(0)} />
            </div>
          </div>

          <div className="card"><h3>Live hardware<span className="r">from the OS / NVIDIA driver, every second</span></h3>
            <div className="grid g4">
              <Stat k="GPU" v={g ? `${g.util_pct ?? "–"}%` : "–"} sub={info && g ? undefined : "no GPU"} bar={g?.util_pct} />
              <Stat k="VRAM" v={g ? `${fmt.mb(g.vram_used_mb)} / ${fmt.mb(g.vram_total_mb)}` : "–"} sub={last?.mem_reserved_mb ? `this run ${fmt.mb(last.mem_reserved_mb)} (peak ${fmt.mb(last.mem_peak_mb)})` : undefined} bar={g ? (100 * g.vram_used_mb) / g.vram_total_mb : undefined} />
              <Stat k="GPU temperature" v={g?.temp_c != null ? `${g.temp_c}°C` : "n/a"} sub={g?.throttle?.length ? `throttle: ${g.throttle.join(", ")}` : undefined} />
              <Stat k="GPU power" v={g?.power_w != null ? `${Math.round(g.power_w)} W` : "n/a"} sub={g?.power_limit_w ? `limit ${g.power_limit_w} W` : undefined} />
              <Stat k="GPU clock" v={g?.clock_mhz ? `${fmt.int(g.clock_mhz)} MHz` : "n/a"} />
              <Stat k="CPU" v={telemetry ? `${Math.round(telemetry.cpu_pct)}%` : "–"} sub={proc ? `trainer ${Math.round(proc.cpu_pct)}% · ${proc.threads} threads` : undefined} bar={telemetry?.cpu_pct} />
              <Stat k="CPU temperature" v={telemetry?.cpu_temp_c != null ? `${telemetry.cpu_temp_c}°C` : "n/a"} sub={telemetry?.cpu_temp_c == null ? "not exposed by this OS" : undefined} />
              <Stat k="RAM" v={telemetry ? `${fmt.mb(telemetry.ram_used_mb)} / ${fmt.mb(telemetry.ram_total_mb)}` : "–"} sub={proc ? `trainer ${fmt.mb(proc.rss_mb)} · ${Math.round(telemetry!.ram_pct)}%` : telemetry ? `${Math.round(telemetry.ram_pct)}%` : undefined} bar={telemetry?.ram_pct} />
            </div>
            <div style={{ marginTop: 10 }}>
              <LineChart height={110} yMin={0} yMax={100} xFmt={(x) => `${Math.round(x)}s`} series={[
                { name: "GPU %", color: "var(--acc)", points: history.map((s) => [s.t - t0, s.gpus?.[0]?.util_pct ?? null] as [number, number]) },
                { name: "VRAM %", color: "var(--violet)", points: history.map((s) => [s.t - t0, s.gpus?.[0] ? (100 * s.gpus[0].vram_used_mb) / s.gpus[0].vram_total_mb : null] as [number, number]) },
                { name: "CPU %", color: "var(--blue)", points: history.map((s) => [s.t - t0, s.cpu_pct] as [number, number]) },
                { name: "GPU °C", color: "var(--orange)", points: history.map((s) => [s.t - t0, s.gpus?.[0]?.temp_c ?? null] as [number, number]), dashed: true },
              ]} />
            </div>
          </div>

          <div className="grid g2">
            <div className="card"><h3>Checkpoints<span className="r"><button className="btn sm" disabled={!active} onClick={() => act("checkpoint")}><Icon n="save" s={13} />Save now</button></span></h3>
              <table className="t"><thead><tr><th>Checkpoint</th><th className="num">Step</th><th className="num">Val loss</th><th className="num">Size</th><th /></tr></thead><tbody>
                {ckpts?.checkpoints?.slice().reverse().map((c: any) => (
                  <tr key={c.name}>
                    <td><span className="mono small">{c.kind}</span>{ckpts.best?.name === c.name && <span className="pill good" style={{ marginLeft: 6 }}>best</span>}</td>
                    <td className="num">{fmt.int(c.step)}</td><td className="num">{fmt.num(c.val_loss, 3)}</td><td className="num">{fmt.bytes(c.bytes)}</td>
                    <td style={{ whiteSpace: "nowrap" }}>
                      <button className="btn sm ghost" onClick={async () => { const r = await api.post(`/api/runs/${runId}/checkpoints/${c.name}/backup`); toast(`Backup: ${r.path}`); }}>Backup</button>
                      {!active && <button className="btn sm" onClick={() => act("restart", { checkpoint: c.name })}>Resume</button>}
                    </td>
                  </tr>
                ))}
                {!ckpts?.checkpoints?.length && <tr><td colSpan={5} className="dim">No checkpoints yet.</td></tr>}
              </tbody></table>
            </div>
            <div className="card"><h3>Training log</h3>
              <div className="logs-body" style={{ height: 210, border: "1px solid var(--line)", borderRadius: 6, background: "var(--bg)" }}>
                {logLines.slice(-150).map((l, i) => <div key={i} className={/FAILED|TERMINATED|HARD KILL/.test(l) ? "l-err" : /complete|resumed/.test(l) ? "l-ok" : undefined}>{l}</div>)}
              </div>
            </div>
          </div>
        </div>

        <aside className="col" style={{ gap: 12, position: "sticky", top: 0 }}>
          <div className="card" style={{ padding: 12 }}>
            <button className="kill" disabled={!active || state === "stopping"} onClick={() => (settings.instant_kill ? kill(true) : setConfirmKill(true))} aria-label="Kill switch: stop training">
              <span className="big">⛔ KILL</span><span className="small2">STOP TRAINING</span>
            </button>
            {state === "stopping" && <div className="row small dim" style={{ marginTop: 8 }}><Spinner /> stopping - saving the latest safe checkpoint…
              {killing && Date.now() - killing > 5000 && <button className="btn sm danger" onClick={() => kill(true)}>Force kill now</button>}</div>}
            <div className="row" style={{ marginTop: 10 }}>
              {state === "paused"
                ? <button className="btn primary" style={{ flex: 1 }} disabled={!active} onClick={() => act("resume")}><Icon n="play" s={13} />RESUME</button>
                : <button className="btn" style={{ flex: 1 }} disabled={!active || state !== "running"} onClick={() => act("pause")}><Icon n="pause" s={13} />PAUSE</button>}
              {!active && ckpts?.latest && ["terminated", "interrupted", "crashed", "failed"].includes(state) && (
                <button className="btn" style={{ flex: 1 }} onClick={() => act("restart")}>Resume from {ckpts.latest.kind}</button>
              )}
            </div>
            {state === "paused" && <div className="dim small" style={{ marginTop: 6 }}>Model, optimizer, scheduler and step are held in memory{info.config.checkpoint?.checkpoint_on_pause !== false ? " and saved to a checkpoint" : ""}.</div>}
          </div>

          <div className="card">
            <h3>Training performance<span className="r" title="Technical health of this run - not a measure of model quality">?</span></h3>
            {h ? (
              <>
                <div className="row" style={{ alignItems: "baseline" }}><span className="score">{h.score.toFixed(1)}<small> / 10</small></span></div>
                <div style={{ margin: "8px 0" }}><Bar v={h.score * 10} color={statusColor} /></div>
                <div className="status-chip" style={{ color: statusColor }}><span className="dot" style={{ background: statusColor }} />{h.status_label}</div>
                <div className="col" style={{ marginTop: 10, gap: 5 }}>
                  {Object.entries(h.components).map(([k, c]: any) => (
                    <div className="comp" key={k}><span className="dim">{c.label}</span><Bar v={c.score * 10} color={c.score < 5 ? "var(--warn)" : undefined} /><span className="mono small" style={{ textAlign: "right" }}>{c.score.toFixed(1)}</span></div>
                  ))}
                </div>
                <div className={`note ${h.score < 5.5 ? "warn" : ""}`} style={{ marginTop: 10 }}>
                  {h.score < 8.5 && "⚠ "}{h.reason}
                  {h.recommendation && <div style={{ marginTop: 4 }}><b>Recommendation:</b> {h.recommendation}</div>}
                </div>
                {h.issues.length > 1 && <details style={{ marginTop: 6 }}><summary className="dim small">{h.issues.length - 1} more</summary>{h.issues.slice(1).map((i: any, n: number) => <div key={n} className="small" style={{ marginTop: 4 }}>{i.reason} <span className="dim">{i.recommendation}</span></div>)}</details>}
                <dl className="kv" style={{ marginTop: 10 }}>
                  {h.detail.mfu_pct != null && <><dt>Model FLOP use</dt><dd>{h.detail.mfu_pct}% of measured peak</dd></>}
                  {h.detail.data_wait_pct != null && <><dt>Waiting for data</dt><dd>{h.detail.data_wait_pct}%</dd></>}
                </dl>
                <div className="faint small" style={{ marginTop: 6 }}>{h.note} {settings.auto_optimize ? "Auto Optimization is on: live-adjustable settings may be changed." : "Settings are never changed automatically unless Auto Optimization is enabled."}</div>
              </>
            ) : <div className="dim small">{active ? healthData?.reason || "Measuring…" : "Available while training runs."}</div>}
          </div>

          <div className="card"><h3>Live settings</h3>
            <div className="row">
              <span className="dim small" style={{ flex: 1 }}>DataLoader workers</span>
              <div style={{ width: 70 }}><Num value={workers} onChange={setWorkers} min={1} max={32} /></div>
              <button className="btn sm" disabled={!active} onClick={() => act("set", { dataloader_workers: workers })}>Apply</button>
            </div>
            <button className="btn ghost sm" style={{ marginTop: 8 }} onClick={() => setShowCkpt(true)}>Show run configuration</button>
          </div>
        </aside>
      </div>

      <Confirm open={confirmKill} onClose={() => setConfirmKill(false)} danger title="Stop training?" confirmLabel="⛔ Kill training"
        onConfirm={() => kill(false)}
        body={<div className="col">
          <div>New steps stop immediately. MakeAI saves the latest safe checkpoint, releases GPU memory and ends the training process. The run will be marked <b>TERMINATED BY USER</b>.</div>
          <div className="dim small">If the process does not stop within the grace period it is force-terminated and marked as interrupted. Existing checkpoints are never modified. You can turn on instant kill (no confirmation) in Settings.</div>
        </div>} />
      <Dialog open={showCkpt} onClose={() => setShowCkpt(false)} title="Run configuration" wide>
        <pre className="mono small" style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(cfg, null, 2)}</pre>
      </Dialog>
    </div>
  );
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
