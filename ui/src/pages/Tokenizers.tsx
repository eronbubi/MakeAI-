import React, { useState } from "react";
import { api, fmt } from "../api";
import { Check, Dialog, Field, Icon, Num, PathPicker, Sel, Spinner } from "../components/ui";
import { useApp, useFetch } from "../store";

export default function Tokenizers() {
  const { track, toast } = useApp();
  const toks = useFetch<any[]>("/api/tokenizers");
  const ds = useFetch<any[]>("/api/datasets");
  const [train, setTrain] = useState(false);
  const [pick, setPick] = useState(false);
  const [sel, setSel] = useState<string>("");
  const [text, setText] = useState("def hello(name):\n    return f\"Hello, {name}!\"");
  const [enc, setEnc] = useState<any>(null);
  const [form, setForm] = useState({ name: "", kind: "bpe", vocab_size: 16000, dataset_ids: [] as string[], min_frequency: 2, pad: "<|pad|>", bos: "<|bos|>", eos: "<|eos|>", unk: "<|unk|>", extra: "<|system|>, <|user|>, <|assistant|>, <|end|>", spm_model_type: "bpe" });
  const [busy, setBusy] = useState(false);
  const doTrain = async () => {
    setBusy(true);
    try {
      await track(await api.post("/api/tokenizers/train", { name: form.name, kind: form.kind, vocab_size: form.vocab_size, dataset_ids: form.dataset_ids, min_frequency: form.min_frequency,
        specials: { pad: form.pad, bos: form.bos, eos: form.eos, unk: form.unk }, extra_specials: form.extra.split(",").map((s) => s.trim()).filter(Boolean), spm_model_type: form.spm_model_type }), `Train tokenizer ${form.name}`);
      setTrain(false); toks.reload(); toast("Tokenizer trained");
    } catch (e: any) { toast(e.message, true); } finally { setBusy(false); }
  };
  const encode = async (id: string, t: string) => { try { setEnc(await api.post(`/api/tokenizers/${id}/encode`, { text: t })); } catch (e: any) { toast(e.message, true); } };
  return (
    <div className="page">
      <div className="head"><h1 className="title">Tokenizers</h1><span className="sub">BPE · WordPiece · Unigram · SentencePiece · custom</span>
        <div className="actions"><button className="btn" onClick={() => setPick(true)}>Import custom…</button><button className="btn primary" onClick={() => setTrain(true)}><Icon n="plus" s={14} />Train tokenizer</button></div></div>
      <div className="split">
        <div className="card" style={{ padding: 0 }}>
          <table className="t"><thead><tr><th>Name</th><th>Type</th><th className="num">Vocabulary</th><th>Special tokens</th><th>Created</th></tr></thead><tbody>
            {toks.data?.map((t) => (
              <tr key={t.id} style={{ cursor: "pointer", background: sel === t.id ? "var(--hover)" : undefined }} onClick={() => { setSel(t.id); encode(t.id, text); }}>
                <td><b>{t.name}</b>{!t.lossless && <span className="pill warn" style={{ marginLeft: 6 }}>lossy spacing</span>}</td><td>{t.kind}</td><td className="num">{fmt.int(t.vocab_size)}</td>
                <td className="mono small">{Object.values(t.specials || {}).join(" ")}</td><td className="small">{fmt.date(t.created)}</td>
              </tr>
            ))}
            {toks.data && !toks.data.length && <tr><td colSpan={5} className="dim">No tokenizers yet. Train one on a dataset, or import a Hugging Face tokenizer.json.</td></tr>}
          </tbody></table>
        </div>
        <div className="card"><h3>Test</h3>
          <textarea className="in mono" rows={5} value={text} onChange={(e) => { setText(e.target.value); if (sel) encode(sel, e.target.value); }} aria-label="Text to tokenize" />
          {!sel && <div className="dim small" style={{ marginTop: 6 }}>Select a tokenizer.</div>}
          {enc && sel && <>
            <div className="row small" style={{ margin: "8px 0" }}><span className="pill acc">{enc.count} tokens</span><span className={enc.roundtrip === text ? "ok-t" : "warn-t"}>{enc.roundtrip === text ? "lossless round trip" : "round trip differs"}</span></div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>{enc.tokens.map((t: string, i: number) => <span key={i} className="mono small" title={String(enc.ids[i])} style={{ background: `hsl(${(enc.ids[i] * 47) % 360} 55% var(--chip-l))`, padding: "1px 4px", borderRadius: 3 }}>{t}</span>)}</div>
          </>}
        </div>
      </div>
      <Dialog open={train} onClose={() => setTrain(false)} title="Train tokenizer" wide footer={<><button className="btn" onClick={() => setTrain(false)}>Cancel</button><button className="btn primary" disabled={busy || !form.dataset_ids.length} onClick={doTrain}>{busy ? <Spinner /> : "Train"}</button></>}>
        <div className="grid g3">
          <Field label="Name"><input className="in" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
          <Field label="Type" hint={form.kind === "wordpiece" ? "whitespace is not preserved" : form.kind === "bpe" ? "byte-level, lossless, GGUF export" : ""}>
            <Sel value={form.kind} onChange={(v) => setForm({ ...form, kind: v })} options={[{ value: "bpe", label: "BPE (byte-level)" }, { value: "wordpiece", label: "WordPiece" }, { value: "unigram", label: "Unigram" }, { value: "sentencepiece", label: "SentencePiece" }]} /></Field>
          <Field label="Vocabulary size"><Num value={form.vocab_size} onChange={(v) => setForm({ ...form, vocab_size: v })} min={256} /></Field>
          {form.kind === "sentencepiece" && <Field label="SentencePiece model" hint="BPE exports to GGUF"><Sel value={form.spm_model_type} onChange={(v) => setForm({ ...form, spm_model_type: v })} options={["bpe", "unigram"]} /></Field>}
          <Field label="Min frequency"><Num value={form.min_frequency} onChange={(v) => setForm({ ...form, min_frequency: v })} min={1} /></Field>
          <Field label="BOS"><input className="in mono" value={form.bos} onChange={(e) => setForm({ ...form, bos: e.target.value })} /></Field>
          <Field label="EOS"><input className="in mono" value={form.eos} onChange={(e) => setForm({ ...form, eos: e.target.value })} /></Field>
          <Field label="PAD"><input className="in mono" value={form.pad} onChange={(e) => setForm({ ...form, pad: e.target.value })} /></Field>
          <Field label="UNK"><input className="in mono" value={form.unk} onChange={(e) => setForm({ ...form, unk: e.target.value })} /></Field>
        </div>
        <Field label="Additional special tokens" hint="comma separated; the chat role tokens are used by MakeAI's chat template" style={{ marginTop: 8 }}><input className="in mono" value={form.extra} onChange={(e) => setForm({ ...form, extra: e.target.value })} /></Field>
        <div className="lbl" style={{ marginTop: 10 }}>Train on</div>
        <div className="col" style={{ gap: 4 }}>{ds.data?.map((d) => <Check key={d.id} checked={form.dataset_ids.includes(d.id)} onChange={(v) => setForm({ ...form, dataset_ids: v ? [...form.dataset_ids, d.id] : form.dataset_ids.filter((x) => x !== d.id) })}>{d.name} <span className="dim small">{fmt.bytes(d.stats?.bytes)}</span></Check>)}</div>
      </Dialog>
      <PathPicker open={pick} mode="any" title="Choose tokenizer.json / tokenizer.model or its folder" onClose={() => setPick(false)} onPick={async (p) => { try { await api.post("/api/tokenizers/import", { path: p[0] }); toks.reload(); toast("Tokenizer imported"); } catch (e: any) { toast(e.message, true); } }} />
    </div>
  );
}
