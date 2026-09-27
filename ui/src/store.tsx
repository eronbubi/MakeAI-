import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api, Job } from "./api";

// ------------------------------------------------------------------ routing (one page at a time + back)
export type Route = { view: string; id?: string; title?: string };
export const routeKey = (r: Route) => `${r.view}:${r.id ?? ""}`;

export type Telemetry = {
  t: number;
  gpus: { index: number; util_pct: number | null; vram_used_mb: number; vram_total_mb: number; temp_c: number | null;
    power_w: number | null; power_limit_w: number | null; clock_mhz: number | null; mem_clock_mhz: number | null;
    fan_pct: number | null; throttle: string[]; mem_util_pct: number | null }[];
  cpu_pct: number; cpu_per_core: number[]; cpu_temp_c: number | null; cpu_mhz: number | null;
  ram_used_mb: number; ram_total_mb: number; ram_pct: number;
  procs: Record<string, { pid: number; rss_mb: number; cpu_pct: number; threads: number }>;
};

type LogLine = { t: number; src: string; text: string; level?: "err" | "ok" };
type Toast = { id: number; text: string; err?: boolean };

type Ctx = {
  route: Route; tabs: Route[]; active: string; open: (r: Route) => void; close: (key?: string) => void;
  back: () => void; canBack: boolean;
  settings: any; setSettings: (s: any) => void; reloadSettings: () => Promise<void>;
  telemetry: Telemetry | null; history: Telemetry[]; activeRuns: any[];
  subscribe: (runId: string | null, since?: number) => void; runFeed: any; setRunFeedHandler: (fn: ((m: any) => void) | null) => void;
  logs: LogLine[]; log: (src: string, text: string, level?: "err" | "ok") => void;
  toast: (text: string, err?: boolean) => void; toasts: Toast[];
  jobs: Job[]; track: (job: Job, label?: string) => Promise<any>;
  bump: number; refresh: () => void;
  hw: any; setHw: (h: any) => void;
};

const AppCtx = createContext<Ctx>(null as any);
export const useApp = () => useContext(AppCtx);

function loadRoute(): Route {
  try {
    const r = JSON.parse(localStorage.getItem("makeai.route") || "null");
    if (r?.view) return r;
  } catch { /* storage unavailable */ }
  return { view: "dashboard", title: "Home" };
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [route, setRoute] = useState<Route>(loadRoute);
  const [stack, setStack] = useState<Route[]>([]);
  const [settings, setSettings] = useState<any>(null);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [history, setHistory] = useState<Telemetry[]>([]);
  const [activeRuns, setActiveRuns] = useState<any[]>([]);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [bump, setBump] = useState(0);
  const [hw, setHw] = useState<any>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const subRef = useRef<{ id: string | null; since: number }>({ id: null, since: 0 });
  const feedRef = useRef<((m: any) => void) | null>(null);
  const [runFeed, setRunFeed] = useState<any>(null);

  useEffect(() => {
    try { localStorage.setItem("makeai.route", JSON.stringify(route)); } catch { /* ignore */ }
  }, [route]);

  const open = useCallback((r: Route) => {
    setRoute((cur) => {
      if (routeKey(cur) === routeKey(r)) return { ...cur, ...r };
      setStack((st) => [...st.slice(-30), cur]);
      return r;
    });
  }, []);
  const back = useCallback(() => {
    setStack((st) => {
      const prev = st[st.length - 1] || { view: "dashboard", title: "Home" };
      setRoute(prev);
      return st.slice(0, -1);
    });
  }, []);
  const close = useCallback((_k?: string) => back(), [back]);

  const log = useCallback((src: string, text: string, level?: "err" | "ok") => {
    setLogs((l) => [...l.slice(-800), { t: Date.now(), src, text, level }]);
  }, []);
  const toast = useCallback((text: string, err?: boolean) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, text, err }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), err ? 7000 : 3500);
    log("ui", text, err ? "err" : undefined);
  }, [log]);

  const reloadSettings = useCallback(async () => setSettings(await api.get("/api/settings")), []);
  useEffect(() => { reloadSettings(); }, [reloadSettings]);

  // Live telemetry + subscribed run over one WebSocket (reconnects).
  useEffect(() => {
    let stop = false;
    let timer: number | undefined;
    const connect = () => {
      const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`);
      wsRef.current = ws;
      let ping: number | undefined;
      ws.onopen = () => {
        ws.send(JSON.stringify(subRef.current.id ? { subscribe: subRef.current.id, since: subRef.current.since } : { ping: 1 }));
        ping = window.setInterval(() => { if (ws.readyState === 1) ws.send(JSON.stringify({ ping: 1 })); }, 5000);
      };
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data);
        if (m.telemetry) {
          setTelemetry(m.telemetry);
          setHistory((h) => [...h.slice(-299), m.telemetry]);
        }
        setActiveRuns(m.active_runs || []);
        if (m.run && m.run.id === subRef.current.id) {
          subRef.current.since = m.run.next;
          feedRef.current?.(m.run);
          setRunFeed(m.run);
        }
      };
      ws.onclose = () => { clearInterval(ping); if (!stop) timer = window.setTimeout(connect, 1500); };
    };
    connect();
    return () => { stop = true; clearTimeout(timer); wsRef.current?.close(); };
  }, []);

  const subscribe = useCallback((id: string | null, since = 0) => {
    subRef.current = { id, since };
    setRunFeed(null);
    const ws = wsRef.current;
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ subscribe: id, since }));
  }, []);
  const setRunFeedHandler = useCallback((fn: ((m: any) => void) | null) => { feedRef.current = fn; }, []);

  const track = useCallback(async (job: Job, label?: string) => {
    const name = label || job.title;
    log("job", `▶ ${name}`);
    setJobs((j) => [job, ...j.filter((x) => x.id !== job.id)].slice(0, 50));
    let lastLen = 0;
    for (;;) {
      const j = await api.get<Job>(`/api/jobs/${job.id}`);
      setJobs((all) => all.map((x) => (x.id === j.id ? j : x)));
      const lines = j.log || [];
      for (const l of lines.slice(lastLen)) log("job", l);
      lastLen = lines.length;
      if (j.state === "done") { log("job", `✓ ${name}`, "ok"); return j.result; }
      if (j.state === "failed" || j.state === "cancelled") {
        log("job", `✗ ${name}: ${j.error || j.state}`, "err");
        throw new Error(j.error || j.state);
      }
      await new Promise((r) => setTimeout(r, 500));
    }
  }, [log]);

  const value: Ctx = {
    route, tabs: [route], active: routeKey(route), open, close, back, canBack: stack.length > 0, settings, setSettings, reloadSettings, telemetry, history, activeRuns, subscribe, runFeed,
    setRunFeedHandler, logs, log, toast, toasts, jobs, track, bump, refresh: () => setBump((b) => b + 1), hw, setHw,
  };
  return <AppCtx.Provider value={value}>{children}</AppCtx.Provider>;
}

/** Fetch helper hook with reload. */
export function useFetch<T = any>(path: string | null, deps: unknown[] = []): { data: T | null; error: string | null; loading: boolean; reload: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [n, setN] = useState(0);
  const { bump } = useApp();
  useEffect(() => {
    if (!path) return;
    let alive = true;
    setLoading(true);
    api.get<T>(path).then((d) => { if (alive) { setData(d); setError(null); } })
      .catch((e) => { if (alive) setError(e.message); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, n, bump, ...deps]);
  return { data, error, loading, reload: () => setN((x) => x + 1) };
}
