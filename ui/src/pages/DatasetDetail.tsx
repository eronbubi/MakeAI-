import React, { useState } from "react";
import { api, fmt } from "../api";
import { Check, Confirm, Field, Num, Sel, Spinner } from "../components/ui";
import { Route, useApp, useFetch } from "../store";

export default function DatasetDetail({ route }: { route: Route }) {
  const { track, toast, close } = useApp();
  const meta = useFetch<any>(`/api/datasets/${route.id}`);
  const [offset, setOffset] = useState(0);
  const prev = useFetch<any[]>(`/api/datasets/${route.id}/preview?offset=${offset}&limit=20`, [meta.data?.updated]);
  const toks = useFetch<any[]>("/api/tokenizers");
  const models = useFetch<any[]>("/api/models");
  const [busy, setBusy] = useState<string | null>(null);
  const [f, setF] = useState({ min_chars: 0, max_chars: 0, min_words: 0, max_words: 0, include_regex: "", exclude_regex: "", max_symbol_ratio: 0 });
  const [clean, setClean] = useState({ unicode_nfc: true, strip_control: true, strip_trailing_whitespace: true, collapse_blank_lines: true, strip_html: false });
  const [near, setNear] = useState(true);
  const [seed, setSeed] = useState(42);
  const [tok, setTok] = useState("");
  const [del, setDel] = useState(false);
  const run = async (op: string, body: any) => {
    setBusy(op);
    try { const r = await track(await api.post(`/api/datasets/${route.id}/${op}`, body), `${op} ${meta.data?.name}`); if (r?.removed != null) toast(`${op}: ${r.removed} removed, ${r.after} kept`); meta.reload(); }
    catch (e: any) { toast(e.message, true); } finally { setBusy(null); }
  };
  const d = meta.data;
  if (!d) return <div className="page"><Spinner /></div>;
  const s = d.stats || {};
  const tokOptions = [{ value: "", label: "Choose tokenizer…" }, ...(toks.data || []).map((t) => ({ value: t.id, label: `${t.name} (${t.kind}, ${t.vocab_size})` })),
    ...(models.data || []).filter((m) => m.backend !== "llama.cpp").map((m) => ({ value: `model:${m.uid}`, label: `${m.name} tokenizer` }))];
  return (
    <div className="page">
      <div className="head"><h1 className="title">{d.name}</h1><span className="sub mono small">{d.id}</span>
        <div className="actions"><button className="btn danger" onClick={() => setDel(true)}>Delete</button></div></div>
      <div className="grid g4">
        <div className="card stat"><span className="k">Samples</span><span className="v">{fmt.int(s.samples)}</span></div>
        <div className="card stat"><span className="k">Tokens</span><span className="v">{s.tokens ? fmt.int(s.tokens) : "–"}</span><span className="faint small">{s.tokenizer_vocab ? `vocab ${s.tokenizer_vocab}` : "count below"}</span></div>
        <div className="card stat"><span className="k">Size</span><span className="v">{fmt.bytes(s.bytes)}</span><span className="faint small">{fmt.int(s.characters)} characters</span></div>
        <div className="card stat"><span className="k">Avg / max length</span><span className="v" style={{ fontSize: 16 }}>{s.avg_tokens ? `${fmt.int(s.avg_tokens)} / ${fmt.int(s.max_tokens)} tok` : `${fmt.int(s.avg_chars)} / ${fmt.int(s.max_chars)} ch`}</span></div>
      </div>

      <div className="grid g2" style={{ marginTop: 12 }}>
        <div className="card"><h3>Cleaning</h3>
          <div className="col" style={{ gap: 4 }}>
            <Check checked={clean.unicode_nfc} onChange={(v) => setClean({ ...clean, unicode_nfc: v })}>Unicode NFC normalization</Check>
            <Check checked={clean.strip_control} onChange={(v) => setClean({ ...clean, strip_control: v })}>Remove control characters</Check>
            <Check checked={clean.strip_trailing_whitespace} onChange={(v) => setClean({ ...clean, strip_trailing_whitespace: v })}>Strip trailing whitespace</Check>
            <Check checked={clean.collapse_blank_lines} onChange={(v) => setClean({ ...clean, collapse_blank_lines: v })}>Collapse long runs of blank lines</Check>
            <Check checked={clean.strip_html} onChange={(v) => setClean({ ...clean, strip_html: v })}>Strip HTML tags</Check>
          </div>
          <button className="btn" style={{ marginTop: 10 }} disabled={!!busy} onClick={() => run("clean", clean)}>{busy === "clean" ? <Spinner /> : "Clean"}</button>
        </div>
        <div className="card"><h3>Filtering</h3>
          <div className="grid g4">
            <Field label="Min chars"><Num value={f.min_chars} onChange={(v) => setF({ ...f, min_chars: v })} min={0} /></Field>
            <Field label="Max chars"><Num value={f.max_chars} onChange={(v) => setF({ ...f, max_chars: v })} min={0} /></Field>
            <Field label="Min words"><Num value={f.min_words} onChange={(v) => setF({ ...f, min_words: v })} min={0} /></Field>
            <Field label="Max words"><Num value={f.max_words} onChange={(v) => setF({ ...f, max_words: v })} min={0} /></Field>
          </div>
          <div className="grid g3" style={{ marginTop: 8 }}>
            <Field label="Keep if matches"><input className="in mono" value={f.include_regex} onChange={(e) => setF({ ...f, include_regex: e.target.value })} placeholder="regex" /></Field>
            <Field label="Drop if matches"><input className="in mono" value={f.exclude_regex} onChange={(e) => setF({ ...f, exclude_regex: e.target.value })} placeholder="regex" /></Field>
            <Field label="Max symbol ratio" hint="0 = off"><Num value={f.max_symbol_ratio} onChange={(v) => setF({ ...f, max_symbol_ratio: v })} min={0} max={1} /></Field>
          </div>
          <button className="btn" style={{ marginTop: 10 }} disabled={!!busy} onClick={() => run("filter", f)}>{busy === "filter" ? <Spinner /> : "Apply filter"}</button>
        </div>
        <div className="card"><h3>Deduplication &amp; shuffle</h3>
          <Check checked={near} onChange={setNear}>Also remove near-duplicates (MinHash, ≥85% similar)</Check>
          <div className="row" style={{ marginTop: 10 }}>
            <button className="btn" disabled={!!busy} onClick={() => run("dedup", { near })}>{busy === "dedup" ? <Spinner /> : "Deduplicate"}</button>
            <span className="sp" />
            <div style={{ width: 90 }}><Num value={seed} onChange={setSeed} /></div>
            <button className="btn" disabled={!!busy} onClick={() => run("shuffle", { seed })}>{busy === "shuffle" ? <Spinner /> : "Shuffle"}</button>
          </div>
          <div className="dim small" style={{ marginTop: 8 }}>Train/validation/test split, minimum/maximum sequence length and sequence packing are chosen per training run (Create AI → Dataset).</div>
        </div>
        <div className="card"><h3>Token counting</h3>
          <div className="row"><div style={{ flex: 1 }}><Sel value={tok} onChange={setTok} options={tokOptions} /></div>
            <button className="btn" disabled={!tok || !!busy} onClick={() => run("stats", { tokenizer: tok })}>{busy === "stats" ? <Spinner /> : "Count tokens"}</button></div>
          {d.token_stats && <table className="t" style={{ marginTop: 8 }}><tbody>{Object.entries(d.token_stats).filter(([k]) => k !== "none").map(([k, v]: any) => <tr key={k}><td className="small">{k}</td><td className="num">{fmt.int(v.tokens)} tokens</td><td className="num">avg {fmt.int(v.avg_tokens)} · max {fmt.int(v.max_tokens)}</td></tr>)}</tbody></table>}
        </div>
      </div>

      <h2 className="sec">Preview</h2>
      <div className="card" style={{ padding: 0 }}>
        <table className="t"><tbody>
          {prev.data?.map((r) => (
            <tr key={r.index}><td className="num faint" style={{ width: 60 }}>{r.index}</td>
              <td><pre className="mono small" style={{ margin: 0, whiteSpace: "pre-wrap", maxHeight: 160, overflow: "auto" }}>{r.text != null ? r.text.slice(0, 2000) : r.messages.map((m: any) => `[${m.role}] ${m.content}`).join("\n").slice(0, 2000)}</pre></td></tr>
          ))}
        </tbody></table>
        <div className="row" style={{ padding: 8 }}>
          <button className="btn sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 20))}>Previous</button>
          <span className="dim small">{offset + 1}–{Math.min(offset + 20, s.samples || 0)} of {fmt.int(s.samples)}</span>
          <button className="btn sm" disabled={offset + 20 >= (s.samples || 0)} onClick={() => setOffset(offset + 20)}>Next</button>
        </div>
      </div>

      <h2 className="sec">History</h2>
      <table className="t"><tbody>{d.history?.slice().reverse().map((h: any, i: number) => <tr key={i}><td>{h.op}</td><td className="small dim">{h.before != null ? `${fmt.int(h.before)} → ${fmt.int(h.after)} (${fmt.int(h.removed)} removed)` : `${fmt.int(h.records)} records`}</td><td className="small dim">{fmt.date(h.at)}</td></tr>)}</tbody></table>
      <div className="dim small" style={{ marginTop: 6 }}>Sources: <span className="mono">{d.sources?.join(", ")}</span></div>
      <Confirm open={del} onClose={() => setDel(false)} danger title={`Delete ${d.name}?`} confirmLabel="Delete" body="The imported copy of the records is removed. Your original files are not touched."
        onConfirm={async () => { await api.del(`/api/datasets/${d.id}`); toast("Dataset deleted"); close(`dataset:${d.id}`); }} />
    </div>
  );
}
