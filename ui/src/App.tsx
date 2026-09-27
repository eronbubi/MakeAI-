import React, { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, fmt } from "./api";
import { Bar, Dialog, Icon, Spinner } from "./components/ui";
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
const ClaudeModeView = lazy(() => import("./pages/ClaudeMode"));

export const NAV: { group: string; items: { view: string; label: string; icon: string; key?: string }[] }[] = [
  { group: "", items: [
    { view: "dashboard", label: "Home", icon: "dash", key: "1" },
    { view: "create", label: "Create AI", icon: "plus", key: "n" },
    { view: "models", label: "My AIs", icon: "bot", key: "2" },
    { view: "training", label: "Training", icon: "train", key: "3" },
    { view: "playground", label: "Chat", icon: "play", key: "4" },
  ] },
  { group: "Library", items: [
    { view: "datasets", label: "Datasets", icon: "data", key: "5" },
    { view: "tokenizers", label: "Tokenizers", icon: "tok" },
    { view: "evaluate", label: "Evaluation", icon: "eval" },
    { view: "hardware", label: "Hardware", icon: "cpu" },
  ] },
  { group: "Share", items: [
    { view: "import", label: "Import", icon: "import" },
    { view: "discover", label: "Discover", icon: "globe" },
    { view: "projects", label: "Projects", icon: "folder" },
  ] },
];
const EXTRA_TITLES: Record<string, string> = { settings: "Settings", develop: "Development Agent", requirements: "Requirements & Tests" };
const TITLES: Record<string, string> = { ...Object.fromEntries(NAV.flatMap((g) => g.items.map((i) => [i.view, i.label]))), ...EXTRA_TITLES };
const PARENT: Record<string, string> = { ai: "models", run: "training", dataset: "datasets", develop: "settings", requirements: "settings" };

export default function App() {
  const { route, open, back, canBack, settings, telemetry, activeRuns, logs, toasts, jobs } = useApp();
  const [palette, setPalette] = useState(false);
  const [logsOpen, setLogsOpen] = useState(false);
  const [health, setHealth] = useState<any>(null);
  useEffect(() => { api.get("/api/health").then(setHealth).catch(() => setHealth(null)); }, []);

  // Claude Mode: warm white/orange, watch-only view of what the agent (Claude, Codex, Cursor, ...) does in MakeAI
  const [claudeMode, setClaudeMode] = useState<boolean>(() => {
    try { return new URLSearchParams(location.search).get("claude") === "1" || sessionStorage.getItem("makeai.claudeMode") === "1"; } catch { return false; }
  });
  const [claude, setClaude] = useState<{ connected: boolean; active: boolean; agent: string } | null>(null);
  useEffect(() => {
    try { sessionStorage.setItem("makeai.claudeMode", claudeMode ? "1" : "0"); } catch { /* ignore */ }
    const root = document.documentElement;
    if (claudeMode) { if (root.dataset.theme !== "claude") root.dataset.prevTheme = root.dataset.theme || ""; root.dataset.theme = "claude"; }
    else if (root.dataset.theme === "claude") { root.dataset.theme = root.dataset.prevTheme || ""; if (!root.dataset.theme) delete root.dataset.theme; }
    if (!claudeMode && new URLSearchParams(location.search).has("claude")) history.replaceState(null, "", location.pathname);
  }, [claudeMode]);
  useEffect(() => {
    if (!health?.claude_mode) return;
    let stop = false;
    const tick = async () => {
      try { const s = await api.get("/api/claude/state?since=999999999"); if (!stop) setClaude({ connected: s.connected, active: !!s.session?.active, agent: s.agent?.name || "Claude" }); } catch { /* ignore */ }
      if (!stop) setTimeout(tick, 3000);
    };
    tick();
    return () => { stop = true; };
  }, [health?.claude_mode]);

  const go = useCallback((view: string, id?: string, title?: string) => open({ view, id, title: title || TITLES[view] || view }), [open]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (claudeMode) return;           // Claude Mode is watch-only
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "k") { e.preventDefault(); setPalette((p) => !p); }
      else if (mod && e.altKey && e.key.toLowerCase() === "n") { e.preventDefault(); go("create"); }
      else if (e.altKey && e.key === "ArrowLeft" && canBack) { e.preventDefault(); back(); }
      else if (e.altKey && /^[1-5]$/.test(e.key)) {
        const it = NAV.flatMap((g) => g.items).find((i) => i.key === e.key);
        if (it) { e.preventDefault(); go(it.view); }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, back, canBack, claudeMode]);

  if (!settings) return <div style={{ display: "grid", placeItems: "center", height: "100vh" }}><Spinner /></div>;
  if (!settings.profile?.username) return <Setup />;

  const Page = pages[route.view] || pages.dashboard;
  const g = telemetry?.gpus?.[0];
  const navActive = PARENT[route.view] || route.view;
  const run = activeRuns[0];
  const runningJobs = jobs.filter((j) => j.state === "running").length;
  const claudeBtn = health?.claude_mode && (
    <button className={`claude-btn${claudeMode ? " on" : ""}`} onClick={() => setClaudeMode((c) => !c)} aria-pressed={claudeMode}
      title={claude?.connected ? `${claude.agent} is connected` : "No agent connected"}>
      <span className={`dot ${claude?.connected ? "run" : ""}`} />ClaudeMode
    </button>
  );

  return (
    <div className="shell" style={claudeMode ? { gridTemplateColumns: "minmax(0, 1fr)" } : undefined}>
      {claudeMode ? (
        <div className="main">
          <header className="topbar">
            <div className="page-title"><img src="/favicon.svg" width={22} height={22} alt="" /><span>MakeAI</span></div>
            <div className="top-stats">{g && <span>GPU <b>{g.util_pct ?? "–"}%</b></span>}{g?.temp_c != null && <span>Temp <b>{g.temp_c}°C</b></span>}</div>
            {claudeBtn}
          </header>
          <Suspense fallback={<div className="page"><Spinner /></div>}>
            <ClaudeModeView pages={pages} onExit={() => setClaudeMode(false)} />
          </Suspense>
        </div>
      ) : (
        <>
          <nav className="sidebar" aria-label="Main">
            <div className="brand"><img src="/favicon.svg" width={28} height={28} alt="" /><div className="brand-text">MakeAI<small>by Convergent</small></div></div>
            {NAV.map((grp) => (
              <React.Fragment key={grp.group || "main"}>
                {grp.group && <div className="nav-title">{grp.group}</div>}
                {grp.items.map((it) => (
                  <button key={it.view} className={`nav-item${navActive === it.view ? " active" : ""}`} onClick={() => go(it.view)} title={it.label}>
                    <Icon n={it.icon} /><span className="label">{it.label}</span>
                    {it.view === "training" && activeRuns.length > 0 && <span className="count"><span className="dot run" /></span>}
                  </button>
                ))}
              </React.Fragment>
            ))}
            {run ? (
              <button className="side-run" onClick={() => go("run", run.run_id, run.model_name)}>
                <div className="row small" style={{ gap: 8 }}><span className="dot run" /><b style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{run.model_name}</b></div>
                <Bar v={run.total_steps ? (100 * (run.step || 0)) / run.total_steps : 0} />
              </button>
            ) : <div style={{ marginTop: "auto" }} />}
            <button className={`nav-item${navActive === "settings" ? " active" : ""}`} style={{ marginTop: 10 }} onClick={() => go("settings")}>
              <Icon n="gear" /><span className="label">Settings</span>
            </button>
          </nav>
          <div className="main">
            <header className="topbar">
              <div className="page-title">
                {canBack && PARENT[route.view] && <button className="back" onClick={back} aria-label="Back"><Icon n="left" s={14} /></button>}
                <span>{route.title || TITLES[route.view] || "MakeAI"}</span>
              </div>
              <div className="top-stats" aria-label="live hardware">
                {g && <span>GPU <b>{g.util_pct ?? "–"}%</b></span>}
                {g && <span>VRAM <b>{fmt.mb(g.vram_used_mb)}</b></span>}
                {g?.temp_c != null && <span><b>{g.temp_c}°C</b></span>}
              </div>
              {claudeBtn}
            </header>
            <div className="content">
              <PageBoundary key={routeKey(route)}>
                <Suspense fallback={<div className="page"><Spinner /></div>}>
                  <Page route={route} />
                </Suspense>
              </PageBoundary>
            </div>
          </div>
        </>
      )}
      <footer className="statusbar">
        <span className="row" style={{ gap: 7 }}><span className={`dot ${activeRuns.length ? "run" : "ok"}`} />{activeRuns.length ? <>Training <b>{activeRuns[0].model_name}</b> · step {fmt.int(activeRuns[0].step)} / {fmt.int(activeRuns[0].total_steps)}</> : "Ready"}</span>
        {runningJobs > 0 && <span className="row" style={{ gap: 7 }}><Spinner /> {runningJobs} task{runningJobs > 1 ? "s" : ""} running</span>}
        {!claudeMode && <button className="btn ghost sm" onClick={() => setLogsOpen(true)}>Activity log</button>}
        <span className="sp" />
        <span>@{settings.profile.username}</span>
        <span>{claudeMode ? (claude?.connected ? `${claude.agent} connected` : "waiting for an agent") : "Runs locally · no cloud"}</span>
      </footer>
      {!claudeMode && <Palette open={palette} onClose={() => setPalette(false)} go={go} />}
      <Dialog open={logsOpen} onClose={() => setLogsOpen(false)} title="Activity log" wide footer={<button className="btn" onClick={() => setLogsOpen(false)}>Close</button>}>
        <LogBody lines={logs} />
      </Dialog>
      <div className="toast-wrap" role="status" aria-live="polite">{toasts.map((t) => <div key={t.id} className={`toast${t.err ? " err" : ""}`}>{t.text}</div>)}</div>
    </div>
  );
}

function LogBody({ lines }: { lines: { t: number; src: string; text: string; level?: string }[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { const el = ref.current; if (el) el.scrollTop = el.scrollHeight; }, [lines.length]);
  return <div className="logs-body" ref={ref} style={{ height: 420 }}>
    {lines.length === 0 && <span className="faint">Imports, exports, training and other tasks appear here.</span>}
    {lines.map((l, i) => <div key={i} className={l.level ? `l-${l.level}` : undefined}><span className="faint">{new Date(l.t).toLocaleTimeString()} </span>{l.text}</div>)}
  </div>;
}

function Palette({ open, onClose, go }: { open: boolean; onClose: () => void; go: (v: string, id?: string, t?: string) => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const [models, setModels] = useState<any[]>([]);
  const [runs, setRuns] = useState<any[]>([]);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) { d.showModal(); setQ(""); setSel(0); api.get("/api/models").then(setModels).catch(() => {}); api.get("/api/runs").then(setRuns).catch(() => {}); }
    if (!open && d.open) d.close();
  }, [open]);
  useEffect(() => { const d = ref.current; if (!d) return; const c = () => onClose(); d.addEventListener("close", c); return () => d.removeEventListener("close", c); }, [onClose]);
  const items = useMemo(() => {
    const out: { label: string; grp: string; icon: string; run: () => void }[] = [];
    NAV.forEach((g) => g.items.forEach((i) => out.push({ label: i.label, grp: "Go to", icon: i.icon, run: () => go(i.view) })));
    out.push({ label: "Settings", grp: "Go to", icon: "gear", run: () => go("settings") });
    models.forEach((m) => {
      out.push({ label: m.name, grp: "AI", icon: "bot", run: () => go("ai", m.uid, m.name) });
      if (m.runnable) out.push({ label: `Chat with ${m.name}`, grp: "AI", icon: "play", run: () => go("playground", m.uid, `Chat · ${m.name}`) });
    });
    runs.slice(0, 10).forEach((r) => out.push({ label: `${r.model_name} training (${r.state})`, grp: "Training", icon: "train", run: () => go("run", r.run_id, r.model_name) }));
    const s = q.trim().toLowerCase();
    return s ? out.filter((i) => i.label.toLowerCase().includes(s)) : out;
  }, [q, models, runs, go]);
  const exec = (i: number) => { const it = items[i]; if (!it) return; onClose(); it.run(); };
  return (
    <dialog ref={ref} className="dlg palette" aria-label="Search" {...({ closedby: "any" } as any)}
      onClick={(e) => { if (e.target === ref.current && !("closedBy" in HTMLDialogElement.prototype)) ref.current?.close(); }}>
      {open && <>
        <input autoFocus placeholder="Search pages and AIs…" value={q} onChange={(e) => { setQ(e.target.value); setSel(0); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(items.length - 1, s + 1)); }
            if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); }
            if (e.key === "Enter") { e.preventDefault(); exec(sel); }
          }} aria-label="Search" />
        <ul role="listbox">
          {items.slice(0, 40).map((it, i) => (
            <li key={i} role="option" aria-selected={i === sel} className={i === sel ? "sel" : ""} onMouseEnter={() => setSel(i)} onClick={() => exec(i)}>
              <Icon n={it.icon} s={14} />{it.label}<span className="grp">{it.grp}</span>
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
