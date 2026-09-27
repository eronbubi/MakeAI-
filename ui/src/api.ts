// Thin client for the local MakeAI server. Every state-changing request carries the
// X-MakeAI-Client header the server requires (CSRF protection).

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: { "X-MakeAI-Client": "1" } };
  if (body !== undefined) {
    (init.headers as Record<string, string>)["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const r = await fetch(path, init);
  const text = await r.text();
  let data: any = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!r.ok) throw new ApiError(r.status, (data && data.detail) || r.statusText);
  return data as T;
}

export const api = {
  get: <T = any>(p: string) => req<T>("GET", p),
  post: <T = any>(p: string, b: unknown = {}) => req<T>("POST", p, b),
  patch: <T = any>(p: string, b: unknown) => req<T>("PATCH", p, b),
  put: <T = any>(p: string, b: unknown) => req<T>("PUT", p, b),
  del: <T = any>(p: string) => req<T>("DELETE", p),
  async upload(files: FileList | File[]): Promise<{ paths: string[]; dir: string }> {
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f, f.name));
    const r = await fetch("/api/upload", { method: "POST", body: fd, headers: { "X-MakeAI-Client": "1" } });
    if (!r.ok) throw new ApiError(r.status, (await r.json()).detail);
    return r.json();
  },
};

export type Job = { id: string; kind: string; title: string; state: string; progress: string; error: string | null; result?: any; log?: string[] };

/** Poll a background job until it ends. Calls onProgress on every change. */
export async function waitJob(id: string, onProgress?: (j: Job) => void): Promise<any> {
  for (;;) {
    const j = await api.get<Job>(`/api/jobs/${id}`);
    onProgress?.(j);
    if (j.state === "done") return j.result;
    if (j.state === "failed") throw new Error(j.error || "job failed");
    if (j.state === "cancelled") throw new Error("cancelled");
    await new Promise((r) => setTimeout(r, 500));
  }
}

/** Server-sent events over POST (fetch streaming). */
export async function streamPost(path: string, body: unknown, onData: (d: any) => void, signal?: AbortSignal) {
  const r = await fetch(path, {
    method: "POST", signal,
    headers: { "Content-Type": "application/json", "X-MakeAI-Client": "1" },
    body: JSON.stringify(body),
  });
  if (!r.ok || !r.body) throw new ApiError(r.status, (await r.text()) || r.statusText);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      if (chunk.startsWith("data: ")) onData(JSON.parse(chunk.slice(6)));
    }
  }
}

// ------------------------------------------------------------------ formatting
export const fmt = {
  params(n?: number | null) {
    if (!n) return "–";
    if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 2)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
    return String(n);
  },
  int(n?: number | null) { return n == null ? "–" : Math.round(n).toLocaleString("en-US"); },
  num(n?: number | null, d = 2) { return n == null || !isFinite(n) ? "–" : n.toFixed(d); },
  bytes(b?: number | null) {
    if (b == null) return "–";
    const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; let v = b;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
  },
  mb(mb?: number | null) { return mb == null ? "–" : mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`; },
  dur(s?: number | null) {
    if (s == null || !isFinite(s)) return "–";
    s = Math.max(0, Math.round(s));
    const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    if (d) return `${d}d ${h}h`;
    if (h) return `${h}h ${m}m`;
    if (m) return `${m}m ${sec}s`;
    return `${sec}s`;
  },
  clock(t?: number | null) {
    if (!t) return "–";
    return new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  },
  date(s?: string | null) { return s ? new Date(s).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "–"; },
  lr(n?: number | null) { return n == null ? "–" : n.toExponential(2); },
};

export const stateLabel: Record<string, string> = {
  created: "Created", starting: "Starting", running: "Running", paused: "Paused", stopping: "Stopping",
  completed: "Completed", terminated: "TERMINATED BY USER", interrupted: "Interrupted", failed: "Failed", crashed: "Crashed",
};
