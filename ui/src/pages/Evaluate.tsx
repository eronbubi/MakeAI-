import React, { useState } from "react";
import { api, fmt } from "../api";
import { Field, Num, PathPicker, Sel, Spinner } from "../components/ui";
import { useApp, useFetch } from "../store";

const SAMPLE = `{"prompt": "What does json.dumps do in Python?", "expected": "JSON", "match": "contains"}
{"prompt": "What is 2 + 2? Answer with a number.", "expected": "4", "match": "contains"}`;

export default function Evaluate() {
  const { track, toast } = useApp();
  const models = useFetch<any[]>("/api/models");
  const ds = useFetch<any[]>("/api/datasets");
  const native = (models.data || []).filter((m) => m.runnable && (m.backend || "native") === "native");
  const runnable = (models.data || []).filter((m) => m.runnable);
  const [uid, setUid] = useState("");
  const [dsId, setDsId] = useState("");
  const [split, setSplit] = useState("val");
  const [batches, setBatches] = useState(50);
  const [bench, setBench] = useState(SAMPLE);
  const [benchUid, setBenchUid] = useState("");
  const [maxTok, setMaxTok] = useState(64);
  const [busy, setBusy] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [cmp, setCmp] = useState<string[]>([]);
  const [pick, setPick] = useState(false);
  const comparison = useFetch<any[]>(cmp.length ? `/api/eval/compare?uids=${cmp.join(",")}` : null, [result]);

  const runDs = async () => {
    setBusy("ds");
    try { const r = await track(await api.post("/api/eval/dataset", { uid, dataset_id: dsId, split, max_batches: batches }), "Evaluate on dataset"); setResult(r); if (!cmp.includes(uid)) setCmp([...cmp, uid]); }
    catch (e: any) { toast(e.message, true); } finally { setBusy(null); }
  };
  const runBench = async (path?: string) => {
    setBusy("bench");
    try {
      const items = path ? undefined : bench.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
      const r = await track(await api.post("/api/eval/benchmark", { uid: benchUid, items, path, params: { max_tokens: maxTok } }), "Benchmark");
      setResult(r); if (!cmp.includes(benchUid)) setCmp([...cmp, benchUid]);
    } catch (e: any) { toast(e.message, true); } finally { setBusy(null); }
  };
  const opts = (list: any[]) => [{ value: "", label: "Choose AI…" }, ...list.map((m) => ({ value: m.uid, label: `${m.name} ${m.version}` }))];
  return (
    <div className="page">
      <div className="head"><h1 className="title">Evaluation</h1><span className="sub">Validation loss, perplexity, token accuracy, custom benchmarks and model comparison.</span></div>
      <div className="grid g2">
        <div className="card"><h3>Dataset metrics</h3>
          <div className="grid g2">
            <Field label="AI"><Sel value={uid} onChange={setUid} options={opts(native)} /></Field>
            <Field label="Dataset"><Sel value={dsId} onChange={setDsId} options={[{ value: "", label: "Choose dataset…" }, ...(ds.data || []).map((d) => ({ value: d.id, label: d.name }))]} /></Field>
            <Field label="Split"><Sel value={split} onChange={setSplit} options={[{ value: "val", label: "validation (2%)" }, { value: "train", label: "whole dataset" }]} /></Field>
            <Field label="Max batches"><Num value={batches} onChange={setBatches} min={1} /></Field>
          </div>
          <button className="btn primary" style={{ marginTop: 10 }} disabled={!uid || !dsId || !!busy} onClick={runDs}>{busy === "ds" ? <Spinner /> : "Evaluate"}</button>
          <div className="dim small" style={{ marginTop: 6 }}>Tokenised with the AI's own tokenizer. Test prompts: use the Playground with temperature 0.</div>
        </div>
        <div className="card"><h3>Custom benchmark</h3>
          <div className="grid g2"><Field label="AI"><Sel value={benchUid} onChange={setBenchUid} options={opts(runnable)} /></Field><Field label="Max tokens per answer"><Num value={maxTok} onChange={setMaxTok} min={1} /></Field></div>
          <Field label="Items (JSONL)" hint='one per line: {"prompt", "expected", "match": contains | exact | exact_ci | starts_with | regex}' style={{ marginTop: 8 }}>
            <textarea className="in mono" rows={6} value={bench} onChange={(e) => setBench(e.target.value)} />
          </Field>
          <div className="row" style={{ marginTop: 10 }}>
            <button className="btn primary" disabled={!benchUid || !!busy} onClick={() => runBench()}>{busy === "bench" ? <Spinner /> : "Run benchmark"}</button>
            <button className="btn" disabled={!benchUid || !!busy} onClick={() => setPick(true)}>Run a test dataset file…</button>
          </div>
        </div>
      </div>

      {result && (
        <div className="card" style={{ marginTop: 12 }}><h3>Result · {result.name}</h3>
          {result.kind === "dataset" ? (
            <div className="grid g4">
              <div className="stat"><span className="k">Loss</span><span className="v">{fmt.num(result.loss, 4)}</span></div>
              <div className="stat"><span className="k">Perplexity</span><span className="v">{fmt.num(result.perplexity, 2)}</span></div>
              <div className="stat"><span className="k">Token accuracy</span><span className="v">{(100 * result.token_accuracy).toFixed(2)}%</span></div>
              <div className="stat"><span className="k">Tokens</span><span className="v">{fmt.int(result.tokens)}</span><span className="faint small">{result.split} · ctx {result.context_length} · {result.seconds}s</span></div>
            </div>
          ) : (
            <>
              <div className="stat"><span className="k">Accuracy</span><span className="v">{result.accuracy != null ? `${(100 * result.accuracy).toFixed(1)}%` : "–"} <small>{result.scored} scored of {result.n}</small></span></div>
              <table className="t" style={{ marginTop: 8 }}><thead><tr><th>Prompt</th><th>Expected</th><th>Output</th><th /></tr></thead><tbody>
                {result.results?.map((r: any, i: number) => <tr key={i}><td className="small">{r.prompt}</td><td className="mono small">{r.expected}</td><td className="small" style={{ whiteSpace: "pre-wrap", maxWidth: 480 }}>{r.output}</td><td>{r.correct == null ? "" : r.correct ? <span className="pill good">✓</span> : <span className="pill bad">✗</span>}</td></tr>)}
              </tbody></table>
            </>
          )}
        </div>
      )}

      <div className="card" style={{ marginTop: 12 }}><h3>Model comparison</h3>
        <div className="row" style={{ marginBottom: 8 }}>
          <div style={{ width: 260 }}><Sel value="" onChange={(v: string) => v && !cmp.includes(v) && setCmp([...cmp, v])} options={[{ value: "", label: "Add AI to compare…" }, ...runnable.map((m) => ({ value: m.uid, label: m.name }))]} /></div>
          {cmp.map((u) => <span key={u} className="pill">{u}<button className="btn ghost sm" onClick={() => setCmp(cmp.filter((x) => x !== u))} aria-label="remove">×</button></span>)}
        </div>
        {comparison.data && (
          <table className="t"><thead><tr><th>AI</th><th className="num">Parameters</th><th className="num">Training val loss</th><th>Evaluations</th></tr></thead><tbody>
            {comparison.data.map((r) => <tr key={r.uid}><td><b>{r.name}</b> {r.version}</td><td className="num">{fmt.params(r.params)}</td><td className="num">{fmt.num(r.last_eval?.val_loss, 4)}</td>
              <td className="small">{r.evaluations.map((e: any) => <div key={e.id}>{e.name}: {e.kind === "dataset" ? `loss ${fmt.num(e.loss, 3)} / ppl ${fmt.num(e.perplexity, 1)} / acc ${(100 * e.token_accuracy).toFixed(1)}%` : `accuracy ${e.accuracy != null ? (100 * e.accuracy).toFixed(1) + "%" : "–"}`}</div>)}</td></tr>)}
          </tbody></table>
        )}
      </div>
      <PathPicker open={pick} mode="file" title="Choose a .json or .jsonl test file" onClose={() => setPick(false)} onPick={(p) => runBench(p[0])} />
    </div>
  );
}
