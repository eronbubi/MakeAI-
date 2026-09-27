import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";

// ------------------------------------------------------------------ icons (stroke, 16px)
const P: Record<string, string> = {
  dash: "M3 3h4v6H3zM9 3h4v3H9zM9 8h4v5H9zM3 11h4v2H3z",
  cpu: "M5 5h6v6H5zM3 7H1.5M3 9H1.5M13 7h1.5M13 9h1.5M7 3V1.5M9 3V1.5M7 13v1.5M9 13v1.5M3 3h10v10H3z",
  plus: "M8 3v10M3 8h10",
  bot: "M4 6h8v7H4zM8 3v3M6 9h.01M10 9h.01M2 9v2M14 9v2",
  data: "M3 4c0-1 2.2-1.8 5-1.8S13 3 13 4v8c0 1-2.2 1.8-5 1.8S3 13 3 12zM3 4c0 1 2.2 1.8 5 1.8S13 5 13 4M3 8c0 1 2.2 1.8 5 1.8S13 9 13 8",
  tok: "M2 4h5M2 8h8M2 12h4M9 11l2 2 3-4",
  train: "M2 13l3-4 3 2 3-6 3 3",
  play: "M5 3l8 5-8 5z",
  eval: "M3 13V7M7 13V3M11 13V9M2 13h12",
  import: "M8 2v8M5 7l3 3 3-3M3 12v2h10v-2",
  globe: "M8 1.8a6.2 6.2 0 1 0 0 12.4A6.2 6.2 0 0 0 8 1.8zM1.8 8h12.4M8 1.8c1.8 2 2.6 4 2.6 6.2S9.8 12.2 8 14.2M8 1.8C6.2 3.8 5.4 5.8 5.4 8s.8 4.2 2.6 6.2",
  folder: "M2 4h4l1.5 1.5H14V13H2z",
  code: "M6 4L2 8l4 4M10 4l4 4-4 4",
  check: "M3 8.5l3 3 7-7",
  gear: "M8 5.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5zM8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.4 1.4M11.6 11.6L13 13M3 13l1.4-1.4M11.6 4.4L13 3",
  search: "M7 2.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9zM10.3 10.3L14 14",
  x: "M4 4l8 8M12 4l-8 8",
  pause: "M5 3v10M11 3v10",
  stop: "M4 4h8v8H4z",
  save: "M3 3h8l2 2v8H3zM5 3v3h5V3M5 13V9h6v4",
  share: "M11 3l3 3-3 3M14 6H7a4 4 0 0 0-4 4v2",
  export: "M8 10V2M5 5l3-3 3 3M3 10v4h10v-4",
  edit: "M3 13l1-3 7-7 2 2-7 7zM10 4l2 2",
  trash: "M3 4h10M6 4V2.5h4V4M4.5 4l.7 9.5h5.6l.7-9.5",
  bolt: "M9 1.5L3 9h4.5L7 14.5 13 7H8.5z",
  layers: "M8 2l6 3-6 3-6-3zM2 8l6 3 6-3M2 11l6 3 6-3",
  log: "M3 3h10v10H3zM5 6h6M5 8.5h6M5 11h3",
  sidebar: "M2 3h12v10H2zM6 3v10",
  file: "M4 1.5h5l3 3v10H4zM9 1.5v3h3",
  dir: "M2 4h4l1.5 1.5H14V13H2z",
  up: "M8 13V3M4 7l4-4 4 4",
  left: "M10 3L5 8l5 5",
  chip: "M4 4h8v8H4zM6 6h4v4H6zM6 2v2M10 2v2M6 12v2M10 12v2M2 6h2M2 10h2M12 6h2M12 10h2",
  sliders: "M4 2v12M8 2v12M12 2v12M2.5 5h3M6.5 10h3M10.5 6h3",
  eye: "M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8zM8 6a2 2 0 1 0 0 4 2 2 0 0 0 0-4z",
  spark: "M8 1.5v13M1.5 8h13M3.4 3.4l9.2 9.2M12.6 3.4l-9.2 9.2",
  done: "M8 1.8a6.2 6.2 0 1 0 0 12.4A6.2 6.2 0 0 0 8 1.8zM5.3 8.2l1.9 1.9 3.6-3.8",
  chat: "M2.5 3h11v7.5H7L4 13v-2.5H2.5z",
  box: "M2 5l6-3 6 3v6l-6 3-6-3zM2 5l6 3 6-3M8 8v6",
  control: "M5 4v8M11 4v8",
  lock: "M4.5 7V5a3.5 3.5 0 0 1 7 0v2M3.5 7h9v7h-9z",
};
export function Icon({ n, s = 16, c }: { n: string; s?: number; c?: string }) {
  return (
    <svg width={s} height={s} viewBox="0 0 16 16" fill="none" stroke={c || "currentColor"} strokeWidth={1.4}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={P[n] || P.dash} /></svg>
  );
}

// ------------------------------------------------------------------ dialog (native <dialog>, modal)
export function Dialog({ open, onClose, title, children, footer, wide, dismissible = true, labelId }: {
  open: boolean; onClose: () => void; title?: React.ReactNode; children: React.ReactNode; footer?: React.ReactNode;
  wide?: boolean; dismissible?: boolean; labelId?: string;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const id = useMemo(() => labelId || `dlg-${Math.random().toString(36).slice(2)}`, [labelId]);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    const onCancel = (e: Event) => { if (!dismissible) e.preventDefault(); };
    const onCloseEv = () => onClose();
    // light-dismiss fallback for browsers without <dialog closedby>
    const onClick = (e: MouseEvent) => {
      if (!dismissible || "closedBy" in HTMLDialogElement.prototype || e.target !== d) return;
      const r = d.getBoundingClientRect();
      const inside = r.top <= e.clientY && e.clientY <= r.bottom && r.left <= e.clientX && e.clientX <= r.right;
      if (!inside) d.close();
    };
    d.addEventListener("cancel", onCancel);
    d.addEventListener("close", onCloseEv);
    d.addEventListener("click", onClick);
    return () => { d.removeEventListener("cancel", onCancel); d.removeEventListener("close", onCloseEv); d.removeEventListener("click", onClick); };
  }, [onClose, dismissible]);
  return (
    // closedby is a newer attribute; React passes it through as-is.
    <dialog ref={ref} className={`dlg${wide ? " wide" : ""}`} aria-labelledby={id} {...({ closedby: dismissible ? "any" : "none" } as any)}>
      {open && (
        <>
          {title && <div className="dlg-head" id={id}>{title}</div>}
          <div className="dlg-body">{children}</div>
          {footer && <div className="dlg-foot">{footer}</div>}
        </>
      )}
    </dialog>
  );
}

// ------------------------------------------------------------------ form fields
export function Field({ label, hint, children, style }: { label: React.ReactNode; hint?: React.ReactNode; children: React.ReactNode; style?: React.CSSProperties }) {
  return <div className="field" style={style}><label>{label}</label>{children}{hint && <div className="hint">{hint}</div>}</div>;
}

export function Num({ value, onChange, step, min, max, disabled, placeholder }: {
  value: number | null | undefined; onChange: (v: number) => void; step?: number | string; min?: number; max?: number; disabled?: boolean; placeholder?: string;
}) {
  const [txt, setTxt] = useState(value == null ? "" : String(value));
  useEffect(() => { if (Number(txt) !== value) setTxt(value == null ? "" : String(value)); /* eslint-disable-next-line */ }, [value]);
  return (
    <input className="in mono" inputMode="decimal" value={txt} disabled={disabled} placeholder={placeholder}
      onChange={(e) => {
        setTxt(e.target.value);
        const v = Number(e.target.value);
        if (e.target.value.trim() !== "" && isFinite(v)) onChange(min != null ? Math.max(min, max != null ? Math.min(max, v) : v) : v);
      }} step={step as any} />
  );
}

export function Sel<T extends string | number>({ value, onChange, options, disabled }: {
  value: T; onChange: (v: T) => void; options: (T | { value: T; label: string })[]; disabled?: boolean;
}) {
  return (
    <select className="in" value={String(value)} disabled={disabled} onChange={(e) => {
      const o = options.map((x) => (typeof x === "object" ? x.value : x)).find((x) => String(x) === e.target.value);
      onChange(o as T);
    }}>
      {options.map((o) => {
        const v = typeof o === "object" ? o.value : o;
        const l = typeof o === "object" ? o.label : String(o);
        return <option key={String(v)} value={String(v)}>{l}</option>;
      })}
    </select>
  );
}

export function Seg<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return <div className="seg" role="group">{options.map((o) => <button key={o.value} type="button" className={o.value === value ? "on" : ""} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>{o.label}</button>)}</div>;
}

export function Check({ checked, onChange, children, disabled }: { checked: boolean; onChange: (v: boolean) => void; children: React.ReactNode; disabled?: boolean }) {
  return <label className="check"><input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />{children}</label>;
}

export function Spinner() { return <span className="spinner" role="status" aria-label="working" />; }

export function Bar({ v, color }: { v: number; color?: string }) {
  return <div className="bar"><i style={{ width: `${Math.max(0, Math.min(100, v))}%`, background: color }} /></div>;
}

// ------------------------------------------------------------------ charts (own SVG)
export type Series = { name: string; color: string; points: [number, number][]; dashed?: boolean };

export function LineChart({ series, height = 180, yLabel, xFmt, yFmt, logY, yMin, yMax, bare }: {
  series: Series[]; height?: number; yLabel?: string; xFmt?: (x: number) => string; yFmt?: (y: number) => string;
  logY?: boolean; yMin?: number; yMax?: number; bare?: boolean;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(600);
  const [hover, setHover] = useState<number | null>(null);
  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth || 600));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const pad = bare ? { l: 6, r: 6, t: 8, b: 8 } : { l: 46, r: 10, t: 8, b: 20 };
  const all = series.flatMap((s) => s.points).filter((p) => p[1] != null && isFinite(p[1]) && (!logY || p[1] > 0));
  if (!all.length) return <div ref={wrap} className="empty" style={{ height }}>No data yet</div>;
  const tr = (y: number) => (logY ? Math.log10(y) : y);
  let x0 = Math.min(...all.map((p) => p[0])), x1 = Math.max(...all.map((p) => p[0]));
  if (x1 === x0) x1 = x0 + 1;
  let y0 = yMin ?? Math.min(...all.map((p) => tr(p[1]))), y1 = yMax ?? Math.max(...all.map((p) => tr(p[1])));
  if (y1 === y0) { y1 += 1; y0 -= 1; }
  const span = y1 - y0; if (yMin == null) y0 -= span * 0.05; if (yMax == null) y1 += span * 0.05;
  const W = Math.max(200, w), H = height;
  const sx = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - (tr(y) - y0) / (y1 - y0)) * (H - pad.t - pad.b);
  const ticks = 4;
  const yt = Array.from({ length: ticks + 1 }, (_, i) => y0 + ((y1 - y0) * i) / ticks);
  const xt = Array.from({ length: 5 }, (_, i) => x0 + ((x1 - x0) * i) / 4);
  const yf = yFmt || ((v: number) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(0)}k` : Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2)));
  const hx = hover != null ? x0 + ((hover - pad.l) / (W - pad.l - pad.r)) * (x1 - x0) : null;
  return (
    <div ref={wrap} style={{ position: "relative" }}>
      <svg className="chart" width={W} height={H} role="img" aria-label={yLabel || "chart"}
        onMouseMove={(e) => { const r = (e.currentTarget as SVGElement).getBoundingClientRect(); const x = e.clientX - r.left; setHover(x >= pad.l && x <= W - pad.r ? x : null); }}
        onMouseLeave={() => setHover(null)}>
        {!bare && yt.map((v, i) => (
          <g key={i}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(logY ? 10 ** v : v)} y2={sy(logY ? 10 ** v : v)} stroke="var(--line)" />
            <text x={pad.l - 6} y={sy(logY ? 10 ** v : v) + 3} textAnchor="end">{yf(logY ? 10 ** v : v)}</text>
          </g>
        ))}
        {!bare && xt.map((v, i) => <text key={i} x={sx(v)} y={H - 5} textAnchor={i === 0 ? "start" : i === 4 ? "end" : "middle"}>{xFmt ? xFmt(v) : Math.round(v)}</text>)}
        {series.map((s) => {
          const pts = s.points.filter((p) => p[1] != null && isFinite(p[1]) && (!logY || p[1] > 0));
          if (!pts.length) return null;
          const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
          return <g key={s.name}>
            <path d={d} fill="none" stroke={s.color} strokeWidth={bare ? 3 : 1.8} strokeLinecap="round" strokeLinejoin="round" strokeDasharray={s.dashed ? "4 3" : undefined} />
            {!bare && pts.length < 40 && pts.map((p, i) => <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={2.2} fill={s.color} />)}
          </g>;
        })}
        {hx != null && <line x1={hover!} x2={hover!} y1={pad.t} y2={H - pad.b} stroke="var(--line2)" />}
      </svg>
      {hx != null && (
        <div style={{ position: "absolute", top: 6, left: Math.min(hover! + 10, W - 170), background: "var(--panel2)", border: "1px solid var(--line2)", borderRadius: 6, padding: "5px 8px", fontSize: 11.5, pointerEvents: "none", fontFamily: "var(--mono)" }}>
          <div className="dim">{xFmt ? xFmt(hx) : Math.round(hx)}</div>
          {series.map((s) => {
            const pts = s.points.filter((p) => p[1] != null && isFinite(p[1]));
            if (!pts.length) return null;
            let best = pts[0];
            for (const p of pts) if (Math.abs(p[0] - hx) < Math.abs(best[0] - hx)) best = p;
            return <div key={s.name}><span style={{ color: s.color }}>●</span> {s.name}: {yf(best[1])}</div>;
          })}
        </div>
      )}
    </div>
  );
}

export function Spark({ values, max, color = "var(--acc)", h = 28 }: { values: (number | null)[]; max?: number; color?: string; h?: number }) {
  const v = values.map((x) => (x == null ? 0 : x));
  if (v.length < 2) return <svg width="100%" height={h} />;
  const m = max ?? Math.max(1, ...v);
  const W = 120;
  const d = v.map((x, i) => `${i ? "L" : "M"}${((i / (v.length - 1)) * W).toFixed(1)},${(h - 2 - (x / m) * (h - 4)).toFixed(1)}`).join("");
  return <svg width="100%" height={h} viewBox={`0 0 ${W} ${h}`} preserveAspectRatio="none" aria-hidden="true">
    <path d={`${d}L${W},${h}L0,${h}Z`} fill={color} opacity={0.12} /><path d={d} fill="none" stroke={color} strokeWidth={1.3} vectorEffect="non-scaling-stroke" />
  </svg>;
}

// ------------------------------------------------------------------ local file/folder picker
export function PathPicker({ open, onClose, onPick, mode = "any", multiple, title }: {
  open: boolean; onClose: () => void; onPick: (paths: string[]) => void; mode?: "file" | "dir" | "any"; multiple?: boolean; title?: string;
}) {
  const [path, setPath] = useState<string>("");
  const [data, setData] = useState<any>(null);
  const [sel, setSel] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const go = (p?: string) => api.post("/api/fs/list", { path: p || null }).then((d) => { setData(d); setPath(d.path); setErr(null); }).catch((e) => setErr(e.message));
  useEffect(() => { if (open) { setSel([]); go(path || undefined); } /* eslint-disable-next-line */ }, [open]);
  const toggle = (p: string) => setSel((s) => (s.includes(p) ? s.filter((x) => x !== p) : multiple ? [...s, p] : [p]));
  return (
    <Dialog open={open} onClose={onClose} title={title || "Choose " + (mode === "dir" ? "a folder" : "files")} wide
      footer={<>
        <span className="dim small" style={{ marginRight: "auto" }}>{sel.length ? `${sel.length} selected` : mode === "dir" ? "select folders, or use the current folder" : ""}</span>
        {mode !== "file" && <button className="btn" onClick={() => { onPick([path]); onClose(); }}>Use this folder</button>}
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={!sel.length} onClick={() => { onPick(sel); onClose(); }}>Choose</button>
      </>}>
      <div className="row" style={{ marginBottom: 8 }}>
        <button className="btn sm" disabled={!data?.parent} onClick={() => go(data.parent)} aria-label="Parent folder"><Icon n="up" s={14} /></button>
        <input className="in mono" value={path} onChange={(e) => setPath(e.target.value)} onKeyDown={(e) => e.key === "Enter" && go(path)} style={{ flex: 1 }} />
        {data?.roots?.map((r: string) => <button key={r} className="btn sm" onClick={() => go(r)}>{r}</button>)}
        <button className="btn sm" onClick={() => go(data?.home)}>Home</button>
      </div>
      {err && <div className="note bad">{err}</div>}
      <div style={{ maxHeight: 380, overflow: "auto", border: "1px solid var(--line)", borderRadius: 6 }}>
        <table className="t"><tbody>
          {data?.items?.map((it: any) => (
            <tr key={it.path} onDoubleClick={() => (it.dir ? go(it.path) : (onPick([it.path]), onClose()))} style={{ cursor: "pointer" }}>
              <td style={{ width: 28 }}>
                {(mode === "any" || (mode === "dir") === it.dir) && <input type="checkbox" checked={sel.includes(it.path)} onChange={() => toggle(it.path)} aria-label={`select ${it.name}`} />}
              </td>
              <td onClick={() => (it.dir ? go(it.path) : toggle(it.path))}><span className="row" style={{ gap: 7 }}><Icon n={it.dir ? "dir" : "file"} s={14} />{it.name}</span></td>
              <td className="num dim">{it.size != null ? fmtSize(it.size) : ""}</td>
            </tr>
          ))}
        </tbody></table>
      </div>
      <div className="hint faint small" style={{ marginTop: 6 }}>Double-click a folder to open it. Files stay where they are; MakeAI reads them locally.</div>
    </Dialog>
  );
}
function fmtSize(b: number) { return b > 2 ** 30 ? `${(b / 2 ** 30).toFixed(1)} GB` : b > 2 ** 20 ? `${(b / 2 ** 20).toFixed(1)} MB` : `${Math.ceil(b / 1024)} KB`; }

export function Confirm({ open, title, body, confirmLabel = "Confirm", danger, onConfirm, onClose }: {
  open: boolean; title: string; body: React.ReactNode; confirmLabel?: string; danger?: boolean; onConfirm: () => void; onClose: () => void;
}) {
  return <Dialog open={open} onClose={onClose} title={title}
    footer={<><button className="btn" onClick={onClose} autoFocus>Cancel</button><button className={`btn ${danger ? "danger" : "primary"}`} onClick={() => { onConfirm(); onClose(); }}>{confirmLabel}</button></>}>
    {body}
  </Dialog>;
}

export function AiIcon({ icon, name }: { icon?: string; name: string }) {
  if (icon && (icon.startsWith("data:") || icon.startsWith("http"))) return <div className="icon"><img src={icon} alt="" /></div>;
  return <div className="icon" aria-hidden="true">{icon || name.slice(0, 1).toUpperCase()}</div>;
}
