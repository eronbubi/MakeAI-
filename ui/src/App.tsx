import React, { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, fmt } from "./api";
import { Icon, Spinner } from "./components/ui";
import { Route, routeKey, useApp } from "./store";
import Setup from "./pages/Setup";

const pages: Record<string, React.LazyExoticComponent<React.ComponentType<{ route: Route }>>> = {
  dashboard: lazy(() => import("./pages/Dashboard")),
  hardware: lazy(() => import("./pages/Hardware")),
  create: lazy(() => import("./pages/CreateAI")),
  models: lazy(() => import("./pages/MyAIs")),
  ai: lazy(() => import("./pages/AiDetail")),
  datasets: lazy(() => import("./pages/Datasets")),
  dataset: lazy(() => import("./pages/DatasetDetail")),
  tokenizers: lazy(() => import("./pages/Tokenizers")),
  training: lazy(() => import("./pages/Training")),
  run: lazy(() => import("./pages/RunMonitor")),
  playground: lazy(() => import("./pages/Playground")),
  evaluate: lazy(() => import("./pages/Evaluate")),
  import: lazy(() => import("./pages/Import")),
  discover: lazy(() => import("./pages/Discover")),
  projects: lazy(() => import("./pages/Projects")),
  develop: lazy(() => import("./pages/Develop")),
  requirements: lazy(() => import("./pages/Requirements")),
  settings: lazy(() => import("./pages/Settings")),
};

export const NAV: { group: string; items: { view: string; label: string; icon: string; key?: string }[] }[] = [
  { group: "Lab", items: [
    { view: "dashboard", label: "Dashboard", icon: "dash", key: "1" },
    { view: "hardware", label: "Hardware", icon: "cpu", key: "2" },
    { view: "create", label: "Create AI", icon: "plus", key: "n" },
    { view: "models", label: "My AIs", icon: "bot", key: "3" },
    { view: "training", label: "Training", icon: "train", key: "4" },
    { view: "playground", label: "Playground", icon: "play", key: "5" },
  ] },
  { group: "Data", items: [
    { view: "datasets", label: "Datasets", icon: "data", key: "6" },
    { view: "tokenizers", label: "Tokenizers", icon: "tok" },
    { view: "evaluate", label: "Evaluation", icon: "eval" },
  ] },
  { group: "Share", items: [
    { view: "import", label: "Import", icon: "import" },
    { view: "discover", label: "Discover", icon: "globe" },
    { view: "projects", label: "Projects", icon: "folder" },
  ] },
  { group: "Build", items: [
    { view: "develop", label: "Development Agent", icon: "code" },
    { view: "requirements", label: "Requirements & Tests", icon: "check" },
    { view: "settings", label: "Settings", icon: "gear" },
  ] },
];
const TITLES: Record<string, string> = Object.fromEntries(NAV.flatMap((g) => g.items.map((i) => [i.view, i.label])));

function useDrag(initial: number, key: string, axis: "x" | "y", min: number, max: number, invert = false) {
  const [size, setSize] = useState<number>(() => {
    try { return Number(localStorage.getItem(key)) || initial; } catch { return initial; }
  });
  const [drag, setDrag] = useState(false);
  const onDown = (e: React.PointerEvent) => {
    e.preventDefault();
    setDrag(true);
    const start = axis === "x" ? e.clientX : e.clientY;
    const s0 = size;
    const move = (ev: PointerEvent) => {
      const d = (axis === "x" ? ev.clientX : ev.clientY) - start;
      setSize(Math.max(min, Math.min(max, s0 + (invert ? -d : d))));
    };
    const up = () => { setDrag(false); window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  useEffect(() => { try { localStorage.setItem(key, String(size)); } catch { /* ignore */ } }, [key, size]);
  return { size, setSize, onDown, drag };
}

export default function App() {
  const app = useApp();
  const { tabs, active, open, close, settings, telemetry, activeRuns, logs, toasts, jobs } = app;
  const side = useDrag(210, "makeai.side", "x", 52, 380);
  const bottom = useDrag(170, "makeai.logs", "y", 60, 600, true);
  const [logsOpen, setLogsOpen] = useState<boolean>(() => { try { return localStorage.getItem("makeai.logsOpen") !== "0"; } catch { return true; } });
  const [logTab, setLogTab] = useState<"all" | "jobs" | "run">("all");
  const [palette, setPalette] = useState(false);
  const [health, setHealth] = useState<any>(null);
  useEffect(() => { try { localStorage.setItem("makeai.logsOpen", logsOpen ? "1" : "0"); } catch { /* ignore */ } }, [logsOpen]);
  useEffect(() => { api.get("/api/health").then(setHealth).catch(() => setHealth(null)); }, []);

  const go = useCallback((view: string, id?: string, title?: string) => open({ view, id, title: title || TITLES[view] || view }), [open]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "k") { e.preventDefault(); setPalette((p) => !p); }
      else if (mod && e.key.toLowerCase() === "j") { e.preventDefault(); setLogsOpen((o) => !o); }
      else if (mod && e.key.toLowerCase() === "b") { e.preventDefault(); side.setSize((s) => (s > 60 ? 52 : 210)); }
      else if (mod && e.altKey && e.key.toLowerCase() === "n") { e.preventDefault(); go("create"); }
      else if (mod && e.key.toLowerCase() === "w" && e.altKey) { e.preventDefault(); close(active); }
      else if (e.altKey && /^[1-6]$/.test(e.key)) {
        const it = NAV.flatMap((g) => g.items).find((i) => i.key === e.key);
        if (it) { e.preventDefault(); go(it.view); }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, close, active, side]);

  if (!settings) return <div style={{ display: "grid", placeItems: "center", height: "100vh" }}><Spinner /></div>;
  if (!settings.profile?.username) return <Setup />;

  const cur = tabs.find((t) => routeKey(t) === active) || tabs[0];
  const Page = pages[cur.view] || pages.dashboard;
  const g = telemetry?.gpus?.[0];
  const collapsed = side.size < 90;
  const runLine = activeRuns[0];
  const shown = logs.filter((l) => logTab === "all" || (logTab === "jobs" ? l.src === "job" : l.src === "run"));
  const runningJobs = jobs.filter((j) => j.state === "running").length;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand"><img src="/favicon.svg" width={20} height={20} alt="" />MakeAI <small>by Convergent</small></div>
        <button className="palette-btn" onClick={() => setPalette(true)} aria-label="Open command palette">
          <Icon n="search" s={14} /> Search or run a command <kbd>Ctrl K</kbd>
        </button>
        <div className="top-stats" aria-label="live hardware">
          {g && <span>GPU <b>{g.util_pct ?? "–"}%</b></span>}
          {g && <span>VRAM <b>{fmt.mb(g.vram_used_mb)}</b>/{fmt.mb(g.vram_total_mb)}</span>}
          {g?.temp_c != null && <span><b>{g.temp_c}°C</b></span>}
          {telemetry && <span>CPU <b>{Math.round(telemetry.cpu_pct)}%</b></span>}
          {telemetry && <span>RAM <b>{Math.round(telemetry.ram_pct)}%</b></span>}
        </div>
      </header>
      <div className="body">
        <nav className={`sidebar${collapsed ? " collapsed" : ""}`} style={{ width: side.size }} aria-label="Main">
          {NAV.map((grp) => (
            <div className="nav-group" key={grp.group}>
              <div className="nav-title">{grp.group}</div>
              {grp.items.map((it) => (
                <button key={it.view} className={`nav-item${cur.view === it.view ? " active" : ""}`} onClick={() => go(it.view)} title={it.label}>
                  <Icon n={it.icon} /><span className="label">{it.label}</span>
                  {it.view === "training" && activeRuns.length > 0 && <span className="count"><span className="dot run" /></span>}
                </button>
              ))}
            </div>
          ))}
          <div className="sp" />
          {runLine && !collapsed && (
            <button className="nav-item" style={{ margin: 6, width: "auto", border: "1px solid var(--line)" }} onClick={() => go("run", runLine.run_id, `Run · ${runLine.model_name}`)}>
              <span className="dot run" /><span className="label">{runLine.model_name} · {runLine.step ?? 0}/{runLine.total_steps ?? "?"}</span>
            </button>
          )}
        </nav>
        <div className={`resizer-v${side.drag ? " drag" : ""}`} onPointerDown={side.onDown} role="separator" aria-orientation="vertical" aria-label="Resize sidebar" />
        <main className="main">
          <div className="tabs" role="tablist">
            {tabs.map((t) => {
              const k = routeKey(t);
              return (
                <div key={k} className={`tab${k === active ? " active" : ""}`} role="tab" aria-selected={k === active}
                  onClick={() => open(t)} onAuxClick={(e) => e.button === 1 && close(k)}>
                  <span>{t.title || TITLES[t.view]}</span>
                  <button className="x" aria-label={`Close ${t.title}`} onClick={(e) => { e.stopPropagation(); close(k); }}>×</button>
                </div>
              );
            })}
          </div>
          <div className="content">
            <PageBoundary key={active}>
              <Suspense fallback={<div className="page"><Spinner /></div>}>
                <Page route={cur} />
              </Suspense>
            </PageBoundary>
          </div>
          {logsOpen && (
            <>
              <div className={`resizer-h${bottom.drag ? " drag" : ""}`} onPointerDown={bottom.onDown} role="separator" aria-label="Resize log panel" />
              <section className="logs" style={{ height: bottom.size }} aria-label="Logs">
                <div className="logs-head">
                  {(["all", "jobs", "run"] as const).map((t) => <button key={t} className={`t${logTab === t ? " on" : ""}`} onClick={() => setLogTab(t)}>{t === "all" ? "Output" : t === "jobs" ? `Jobs${runningJobs ? ` (${runningJobs})` : ""}` : "Training"}</button>)}
                  <span className="sp" />
                  <button className="btn ghost sm" onClick={() => setLogsOpen(false)} aria-label="Hide logs">Hide <kbd>Ctrl J</kbd></button>
                </div>
                <LogBody lines={shown} />
              </section>
            </>
          )}
        </main>
      </div>
      <footer className="statusbar">
        <span><span className={`dot ${activeRuns.length ? "run" : "ok"}`} /> {activeRuns.length ? `training ${activeRuns[0].model_name}` : "idle"}</span>
        {runningJobs > 0 && <span><Spinner /> {runningJobs} job{runningJobs > 1 ? "s" : ""}</span>}
        {g && <span>{g.power_w != null ? `${Math.round(g.power_w)} W` : ""} {g.clock_mhz ? `${g.clock_mhz} MHz` : ""}</span>}
        {!logsOpen && <button className="btn ghost sm" onClick={() => setLogsOpen(true)}>Logs</button>}
        <span className="sp" />
        <span>@{settings.profile.username}</span>
        <span>MakeAI {health?.version || ""} · runtime local · no cloud</span>
      </footer>
      <Palette open={palette} onClose={() => setPalette(false)} go={go} toggleLogs={() => setLogsOpen((o) => !o)} />
      <div className="toast-wrap" role="status" aria-live="polite">{toasts.map((t) => <div key={t.id} className={`toast${t.err ? " err" : ""}`}>{t.text}</div>)}</div>
    </div>
  );
}

function LogBody({ lines }: { lines: { t: number; src: string; text: string; level?: string }[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { const el = ref.current; if (el) el.scrollTop = el.scrollHeight; }, [lines.length]);
  return <div className="logs-body" ref={ref}>
    {lines.length === 0 && <span className="faint">Output of jobs, imports, exports and training appears here.</span>}
    {lines.map((l, i) => <div key={i} className={l.level ? `l-${l.level}` : undefined}><span className="faint">{new Date(l.t).toLocaleTimeString()} {l.src.padEnd(3)} </span>{l.text}</div>)}
  </div>;
}

function Palette({ open, onClose, go, toggleLogs }: { open: boolean; onClose: () => void; go: (v: string, id?: string, t?: string) => void; toggleLogs: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const [models, setModels] = useState<any[]>([]);
  const [runs, setRuns] = useState<any[]>([]);
  const { activeRuns, settings, toast } = useApp();
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) { d.showModal(); setQ(""); setSel(0); api.get("/api/models").then(setModels).catch(() => {}); api.get("/api/runs").then(setRuns).catch(() => {}); }
    if (!open && d.open) d.close();
  }, [open]);
  useEffect(() => { const d = ref.current; if (!d) return; const c = () => onClose(); d.addEventListener("close", c); return () => d.removeEventListener("close", c); }, [onClose]);
  const items = useMemo(() => {
    const out: { label: string; grp: string; icon: string; run: () => void; hint?: string }[] = [];
    NAV.forEach((g) => g.items.forEach((i) => out.push({ label: i.label, grp: "Go to", icon: i.icon, run: () => go(i.view), hint: i.key ? `Alt ${i.key.toUpperCase()}` : undefined })));
    out.push({ label: "New AI…", grp: "Action", icon: "plus", run: () => go("create"), hint: "Ctrl Alt N" });
    out.push({ label: "Import a model file…", grp: "Action", icon: "import", run: () => go("import") });
    out.push({ label: "Toggle log panel", grp: "Action", icon: "log", run: toggleLogs, hint: "Ctrl J" });
    out.push({ label: "Run acceptance tests", grp: "Action", icon: "check", run: () => go("requirements") });
    if (activeRuns[0]) {
      const r = activeRuns[0];
      out.push({ label: `Open live monitor: ${r.model_name}`, grp: "Training", icon: "train", run: () => go("run", r.run_id, `Run · ${r.model_name}`) });
      out.push({ label: `KILL training: ${r.model_name}`, grp: "Training", icon: "stop", run: async () => {
        if (!settings?.instant_kill && !confirm(`Stop training ${r.model_name}? The latest safe checkpoint will be saved.`)) return;
        await api.post(`/api/runs/${r.run_id}/kill`, { instant: false }); toast("Kill requested");
      } });
    }
    models.forEach((m) => {
      out.push({ label: `${m.name} ${m.version}`, grp: "AI", icon: "bot", run: () => go("ai", m.uid, m.name) });
      if (m.runnable) out.push({ label: `Run ${m.name} in Playground`, grp: "AI", icon: "play", run: () => go("playground", m.uid, `Chat · ${m.name}`) });
    });
    runs.slice(0, 15).forEach((r) => out.push({ label: `Run ${r.model_name} (${r.state})`, grp: "Run", icon: "train", run: () => go("run", r.run_id, `Run · ${r.model_name}`) }));
    const s = q.trim().toLowerCase();
    return s ? out.filter((i) => i.label.toLowerCase().includes(s) || i.grp.toLowerCase().includes(s)) : out;
  }, [q, models, runs, activeRuns, go, toggleLogs, settings, toast]);
  const exec = (i: number) => { const it = items[i]; if (!it) return; onClose(); it.run(); };
  return (
    <dialog ref={ref} className="dlg palette" aria-label="Command palette" {...({ closedby: "any" } as any)}
      onClick={(e) => { if (e.target === ref.current && !("closedBy" in HTMLDialogElement.prototype)) ref.current?.close(); }}>
      {open && <>
        <input autoFocus placeholder="Type a command, an AI or a page…" value={q} onChange={(e) => { setQ(e.target.value); setSel(0); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(items.length - 1, s + 1)); }
            if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); }
            if (e.key === "Enter") { e.preventDefault(); exec(sel); }
          }} aria-label="Command" />
        <ul role="listbox">
          {items.slice(0, 60).map((it, i) => (
            <li key={i} role="option" aria-selected={i === sel} className={i === sel ? "sel" : ""} onMouseEnter={() => setSel(i)} onClick={() => exec(i)}>
              <Icon n={it.icon} s={14} />{it.label}{it.hint && <kbd>{it.hint}</kbd>}<span className="grp">{it.grp}</span>
            </li>
          ))}
          {!items.length && <li className="faint">No matches</li>}
        </ul>
      </>}
    </dialog>
  );
}

/** Keeps one broken page from blanking the app; reloads once when MakeAI was updated underneath. */
class PageBoundary extends React.Component<{ children: React.ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error) {
    if (/dynamically imported module|Importing a module script failed/.test(error.message)) {
      try {
        if (!sessionStorage.getItem("makeai.reloaded")) { sessionStorage.setItem("makeai.reloaded", "1"); location.reload(); }
      } catch { /* storage unavailable */ }
    }
  }
  render() {
    if (!this.state.error) {
      try { sessionStorage.removeItem("makeai.reloaded"); } catch { /* ignore */ }
      return this.props.children;
    }
    return <div className="page"><div className="note bad">This page failed to load: {this.state.error.message}</div>
      <button className="btn" style={{ marginTop: 10 }} onClick={() => location.reload()}>Reload MakeAI</button></div>;
  }
}
