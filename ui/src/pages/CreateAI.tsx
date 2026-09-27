import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, fmt } from "../api";
import { Check, Field, Icon, Num, Seg, Sel, Spinner } from "../components/ui";
import { Route, useApp, useFetch } from "../store";

const STEPS = ["Identity", "Method", "Architecture", "Dataset", "Optimization", "Review"];
const METHODS = [
  { value: "from_scratch", label: "From scratch", desc: "Initialise every parameter randomly and train the whole model. No base model needed." },
  { value: "full", label: "Full training", desc: "Continue training all weights of an existing model (fine-tuning / continued pre-training)." },
  { value: "lora", label: "LoRA", desc: "Freeze the base (16-bit) and train small low-rank matrices. Fast, little memory." },
  { value: "qlora", label: "QLoRA", desc: "Base quantised to 4-bit NF4 (+double quantisation), LoRA on top. Least memory." },
  { value: "adapter", label: "Adapter training", desc: "Insert bottleneck adapters after attention and MLP; only they train." },
];
const TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

export default function CreateAI({ route }: { route: Route }) {
  const { settings, open, track, toast, log } = useApp();
  const opts = useFetch<any>("/api/options");
  const models = useFetch<any[]>("/api/models");
  const datasets = useFetch<any[]>("/api/datasets");
  const tokenizers = useFetch<any[]>("/api/tokenizers");
  const [step, setStep] = useState(0);
  const [id, setId] = useState({ name: "", creator_name: settings.profile.name, creator_username: settings.profile.username, description: "", version: "1.0", tags: "", icon: "", license: "" });
  const [method, setMethod] = useState<string>(route.id?.startsWith("base:") ? "lora" : "from_scratch");
  const [base, setBase] = useState<string>(route.id?.startsWith("base:") ? route.id.slice(5) : "");
  const [complexity, setComplexity] = useState(4);
  const [mode, setMode] = useState<"auto" | "custom">("auto");
  const [model, setModel] = useState<any>(null);
  const [training, setTraining] = useState<any>(null);
  const [rec, setRec] = useState<any>(null);
  const [recErr, setRecErr] = useState<string | null>(null);
  const [dsSel, setDsSel] = useState<{ id: string; weight: number }[]>([]);
  const [prep, setPrep] = useState({ val_ratio: 0.02, packing: true, min_tokens: 1, max_tokens: 0 });
  const [tokMode, setTokMode] = useState<"new" | "existing">("new");
  const [tokKind, setTokKind] = useState("bpe");
  const [tokVocab, setTokVocabRaw] = useState<number | null>(null);
  const [vocabTouched, setVocabTouched] = useState(false);
  const setTokVocab = (v: number | null) => { setVocabTouched(true); setTokVocabRaw(v); };
  const [tokId, setTokId] = useState("");
  const [ckpt, setCkpt] = useState({ save_every: 0, keep: 3, save_best: true, save_on_kill: true, checkpoint_on_pause: true });
  const [estimate, setEstimate] = useState<any>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const baseModels = (models.data || []).filter((m) => m.runnable && (m.backend || "native") === "native" && m.method !== "lora" && m.method !== "qlora" && m.method !== "adapter");
  const scratch = method === "from_scratch";
  // only a vocabulary the user chose (or an existing tokenizer's) constrains the recommendation
  const vocab = scratch ? (tokMode === "existing" ? tokenizers.data?.find((t) => t.id === tokId)?.vocab_size : vocabTouched ? tokVocab : undefined) : undefined;

  // AUTO: every relevant change re-asks the recommendation engine.
  useEffect(() => {
    if (!scratch && !base) return;
    let alive = true;
    setRecErr(null);
    api.post("/api/recommend", { complexity, method, base_model: scratch ? undefined : base, dataset_ids: dsSel.map((d) => d.id), vocab_size: vocab || undefined })
      .then((r) => {
        if (!alive) return;
        setRec(r);
        if (mode === "auto" || !model) {
          setModel(r.model);
          setTraining(r.training);
          setCkpt((c) => ({ ...c, save_every: r.training.save_every }));
          if (scratch && !vocabTouched) setTokVocabRaw(r.model.vocab_size);
        }
      }).catch((e) => alive && setRecErr(e.message));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [complexity, method, base, dsSel.map((d) => d.id).join(","), mode, vocab]);

  // live memory estimate for whatever is currently configured
  useEffect(() => {
    if (!model || !training) return;
    const t = setTimeout(() => {
      api.post("/api/estimate", { model: { ...model, vocab_size: vocab || tokVocab || model.vocab_size }, training, method }).then(setEstimate).catch(() => setEstimate(null));
    }, 250);
    return () => clearTimeout(t);
  }, [model, training, method, vocab]);

  const setM = (k: string, v: any) => { setMode("custom"); setModel((m: any) => ({ ...m, [k]: v })); };
  const setT = (k: string, v: any) => { setMode("custom"); setTraining((t: any) => ({ ...t, [k]: v })); };
  const setL = (k: string, v: any) => { setMode("custom"); setTraining((t: any) => ({ ...t, lora: { ...(t.lora || {}), [k]: v } })); };

  const valid = [
    id.name.trim() && id.creator_name.trim() && /^[a-z0-9][a-z0-9_.-]{0,38}$/.test(id.creator_username.replace(/^@/, "")) && id.version.trim(),
    scratch || !!base,
    !!model && !(estimate?.errors?.length),
    dsSel.length > 0 && (!scratch || tokMode === "new" || !!tokId),
    !!training,
    true,
  ];

  const pickIcon = (f: File) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = c.height = 96;
      const ctx = c.getContext("2d")!;
      const s = Math.min(img.width, img.height);
      ctx.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, 96, 96);
      setId((x) => ({ ...x, icon: c.toDataURL("image/png") }));
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(f);
  };

  const start = async (trainNow: boolean) => {
    setError(null);
    try {
      let tokenizer = tokId;
      if (scratch && tokMode === "new") {
        setBusy("Training tokenizer…");
        const job = await api.post("/api/tokenizers/train", { name: `${id.name} ${tokKind}`, kind: tokKind, vocab_size: tokVocab || model.vocab_size, dataset_ids: dsSel.map((d) => d.id) });
        const info = await track(job, `Train ${tokKind} tokenizer`);
        tokenizer = info.id;
        log("ui", `tokenizer ${info.id}: ${info.vocab_size} tokens`);
      }
      setBusy("Creating AI…");
      const m = await api.post("/api/models", {
        ...id, creator_username: id.creator_username.replace(/^@/, ""), tags: id.tags.split(",").map((s) => s.trim()).filter(Boolean),
        method, base_model: scratch ? undefined : base, config: scratch ? model : undefined, tokenizer: scratch ? tokenizer : undefined,
        plan: { complexity, mode, training, checkpoint: ckpt, data_prep: prep, datasets: dsSel },
      });
      toast(`${m.name} created · ${m.id}`);
      if (!trainNow) { open({ view: "ai", id: m.uid, title: m.name }); return; }
      setBusy("Starting training…");
      const r = await api.post("/api/runs", { model_uid: m.uid, datasets: dsSel, training: { ...training, save_every: ckpt.save_every }, checkpoint: ckpt, eval: { max_batches: 50 }, data_prep: prep });
      open({ view: "run", id: r.run_id, title: `Run · ${m.name}` });
    } catch (e: any) { setError(e.message); } finally { setBusy(null); }
  };

  const est = rec?.estimates;
  const memFits = estimate ? estimate.fits : est?.fits;
  return (
    <div className="page">
      <div className="head"><h1 className="title">Create AI</h1><span className="sub">Every automatic setting stays editable.</span></div>
      <div className="steps" role="tablist">
        {STEPS.map((s, i) => (
          <React.Fragment key={s}>
            {i > 0 && <span className="sep" />}
            <button role="tab" aria-selected={i === step} className={i === step ? "on" : i < step ? "done" : ""} onClick={() => (i <= step || valid.slice(0, i).every(Boolean)) && setStep(i)}>
              <span className="n">{i < step ? "✓" : i + 1}</span>{s}
            </button>
          </React.Fragment>
        ))}
      </div>

      <div className="split">
        <div className="col" style={{ gap: 12 }}>
          {step === 0 && (
            <div className="card">
              <h3>Basic information</h3>
              <div className="grid g2">
                <Field label="AI name"><input className="in" value={id.name} onChange={(e) => setId({ ...id, name: e.target.value })} placeholder="EronAI" autoFocus /></Field>
                <Field label="Version"><input className="in mono" value={id.version} onChange={(e) => setId({ ...id, version: e.target.value })} /></Field>
                <Field label="Created by"><input className="in" value={id.creator_name} onChange={(e) => setId({ ...id, creator_name: e.target.value })} /></Field>
                <Field label="Username"><input className="in mono" value={id.creator_username} onChange={(e) => setId({ ...id, creator_username: e.target.value.toLowerCase() })} /></Field>
              </div>
              <Field label="Description" style={{ marginTop: 10 }}><textarea className="in" rows={3} value={id.description} onChange={(e) => setId({ ...id, description: e.target.value })} placeholder="A general-purpose AI focused on coding and reasoning." /></Field>
              <div className="grid g2" style={{ marginTop: 10 }}>
                <Field label="Tags" hint="comma separated - coding, reasoning, general, experimental help Discover"><input className="in" value={id.tags} onChange={(e) => setId({ ...id, tags: e.target.value })} placeholder="coding, reasoning" /></Field>
                <Field label="Icon" hint="an emoji, or upload an image">
                  <div className="row">
                    <div className="ai-card"><div className="icon">{id.icon.startsWith("data:") ? <img src={id.icon} alt="" /> : id.icon || id.name.slice(0, 1).toUpperCase() || "?"}</div></div>
                    <input className="in" style={{ width: 70 }} value={id.icon.startsWith("data:") ? "" : id.icon} maxLength={4} onChange={(e) => setId({ ...id, icon: e.target.value })} placeholder="🧠" aria-label="Emoji icon" />
                    <button className="btn sm" onClick={() => fileRef.current?.click()}>Upload…</button>
                    <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && pickIcon(e.target.files[0])} />
                  </div>
                </Field>
              </div>
              <Field label="License (optional)" style={{ marginTop: 10 }}><input className="in" value={id.license} onChange={(e) => setId({ ...id, license: e.target.value })} placeholder="e.g. Apache-2.0" /></Field>
              <div className="note" style={{ marginTop: 12 }}>AI ID: <span className="mono">makeai://{id.creator_username || "user"}/{(id.name || "name").toLowerCase().replace(/[^a-z0-9._-]+/g, "-")}/{id.version}</span> - creator information is stored permanently in the AI's metadata and every exported file.</div>
            </div>
          )}

          {step === 1 && (
            <div className="card">
              <h3>Training method</h3>
              <div className="col">
                {METHODS.map((m) => (
                  <label key={m.value} className="check" style={{ alignItems: "flex-start", padding: 8, border: `1px solid ${method === m.value ? "var(--acc)" : "var(--line)"}`, borderRadius: 6 }}>
                    <input type="radio" name="method" checked={method === m.value} onChange={() => setMethod(m.value)} style={{ marginTop: 3 }} />
                    <span><b>{m.label}</b><div className="dim small">{m.desc}</div></span>
                  </label>
                ))}
              </div>
              {!scratch && (
                <Field label="Base model" style={{ marginTop: 12 }} hint={baseModels.length ? "trained or imported AIs that run on the MakeAI engine" : "import a model (Import page) or train one from scratch first"}>
                  <Sel value={base} onChange={setBase} options={[{ value: "", label: "Choose…" }, ...baseModels.map((m) => ({ value: m.uid, label: `${m.name} ${m.version} · ${fmt.params(m.param_count)} · by ${m.original_creator?.name || m.creator.name}` }))]} />
                </Field>
              )}
            </div>
          )}

          {step === 2 && (
            <>
              <div className="card">
                <h3>How capable should it be?</h3>
                <div className="cx">
                  {Object.entries(opts.data?.complexity || {}).map(([k, v]: any) => (
                    <button key={k} className={complexity === Number(k) ? "on" : ""} onClick={() => setComplexity(Number(k))} aria-pressed={complexity === Number(k)}>
                      <span className="lv">{k}</span><b>{v.label}</b><span className="dim small">{scratch ? `~${fmt.params(v.params)} · ctx ${v.ctx}` : `rank ${v.lora_rank} · ctx ${v.ctx}`}</span>
                    </button>
                  ))}
                </div>
                <div className="dim small" style={{ marginTop: 8 }}>Complexity sets the recommended model size, architecture, context, data requirement, steps, batch and duration - capped by your hardware and dataset.</div>
              </div>
              {model && mode === "auto" && (
                <div className="card">
                  <h3>Recommended model<span className="r"><button className="btn sm" onClick={() => setMode("custom")}>Customize</button></span></h3>
                  <div className="grid g3">
                    <div className="stat"><span className="k">Size</span><span className="v">{fmt.params(estimate?.params ?? rec?.estimates?.params)}</span></div>
                    <div className="stat"><span className="k">Layers × width</span><span className="v">{model.n_layers} × {model.hidden_size}</span></div>
                    <div className="stat"><span className="k">Context</span><span className="v">{fmt.int(model.context_length)}</span></div>
                  </div>
                  <div className="dim small" style={{ marginTop: 12 }}>{scratch ? "Chosen for your GPU and your data. You can change every value with Customize." : "Uses the base model's architecture."}</div>
                </div>
              )}
              {model && mode === "custom" && (
                <div className="card">
                  <h3>Architecture{!scratch ? <span className="r">from base model (fixed)</span> : <span className="r"><button className="btn sm ghost" onClick={() => setMode("auto")}>Back to automatic</button></span>}</h3>
                  <div className="grid g4">
                    <Field label="Layers"><Num value={model.n_layers} onChange={(v) => setM("n_layers", v)} min={1} disabled={!scratch} /></Field>
                    <Field label="Hidden size"><Num value={model.hidden_size} onChange={(v) => setM("hidden_size", v)} min={8} disabled={!scratch} /></Field>
                    <Field label="Intermediate size"><Num value={model.intermediate_size} onChange={(v) => setM("intermediate_size", v)} min={8} disabled={!scratch} /></Field>
                    <Field label="Attention heads"><Num value={model.n_heads} onChange={(v) => setM("n_heads", v)} min={1} disabled={!scratch} /></Field>
                    <Field label="Key/value heads" hint={model.n_kv_heads === model.n_heads ? "MHA" : model.n_kv_heads === 1 ? "MQA" : "GQA"}><Num value={model.n_kv_heads} onChange={(v) => setM("n_kv_heads", v)} min={1} disabled={!scratch} /></Field>
                    <Field label="Head dimension"><Num value={model.head_dim} onChange={(v) => setM("head_dim", v)} min={2} disabled={!scratch} /></Field>
                    <Field label="Vocabulary size" hint={scratch ? "set by the tokenizer" : ""}><Num value={vocab || tokVocab || model.vocab_size} onChange={(v) => { setM("vocab_size", v); setTokVocab(v); }} min={64} disabled={!scratch} /></Field>
                    <Field label="Context length"><Num value={model.context_length} onChange={(v) => setM("context_length", v)} min={16} disabled={!scratch} /></Field>
                    <Field label="Embedding size" hint="0 = hidden size"><Num value={model.embedding_size} onChange={(v) => setM("embedding_size", v)} min={0} disabled={!scratch} /></Field>
                    <Field label="Activation"><Sel value={model.activation} onChange={(v) => setM("activation", v)} options={opts.data?.activations || []} disabled={!scratch} /></Field>
                    <Field label="Normalization"><Sel value={model.norm} onChange={(v) => setM("norm", v)} options={opts.data?.norms || []} disabled={!scratch} /></Field>
                    <Field label="Positional encoding"><Sel value={model.pos_encoding} onChange={(v) => setM("pos_encoding", v)} options={opts.data?.positional || []} disabled={!scratch} /></Field>
                    <Field label="RoPE theta"><Num value={model.rope_theta} onChange={(v) => setM("rope_theta", v)} disabled={!scratch || model.pos_encoding !== "rope"} /></Field>
                    <Field label="RoPE scaling"><Sel value={model.rope_scaling} onChange={(v) => setM("rope_scaling", v)} options={opts.data?.rope_scaling || []} disabled={!scratch || model.pos_encoding !== "rope"} /></Field>
                    <Field label="Scaling factor"><Num value={model.rope_scaling_factor} onChange={(v) => setM("rope_scaling_factor", v)} disabled={!scratch || model.rope_scaling === "none"} /></Field>
                    <Field label="Attention type"><Sel value={model.attention_type} onChange={(v) => setM("attention_type", v)} options={opts.data?.attention_types || []} disabled={!scratch} /></Field>
                    {model.attention_type === "sliding_window" && <Field label="Window"><Num value={model.sliding_window} onChange={(v) => setM("sliding_window", v)} min={1} disabled={!scratch} /></Field>}
                  </div>
                  <div className="row" style={{ marginTop: 10, gap: 16 }}>
                    <Check checked={!!model.tie_weights} onChange={(v) => setM("tie_weights", v)} disabled={!scratch}>Weight tying</Check>
                    <Check checked={!!model.bias} onChange={(v) => setM("bias", v)} disabled={!scratch}>Linear bias</Check>
                    <Check checked={!!model.qkv_bias} onChange={(v) => setM("qkv_bias", v)} disabled={!scratch}>QKV bias</Check>
                  </div>
                  {estimate && (
                    <div className="row" style={{ marginTop: 12 }}>
                      <span className="pill acc">{fmt.params(estimate.params)} parameters</span>
                      <span className="pill">{estimate.attention_kind}</span>
                      <span className="pill">{estimate.hf_compatible ? `exports to HF/GGUF as ${estimate.hf_compatible}` : "MakeAI-only architecture (no GGUF/HF export)"}</span>
                    </div>
                  )}
                  {estimate?.errors?.map((e: string) => <div key={e} className="note bad" style={{ marginTop: 8 }}>{e}</div>)}
                </div>
              )}
            </>
          )}

          {step === 3 && (
            <>
              <div className="card">
                <h3>Datasets<span className="r"><button className="btn sm" onClick={() => open({ view: "datasets", title: "Datasets" })}>Manage datasets</button></span></h3>
                {!datasets.data?.length && <div className="empty">No datasets yet. Import JSON, JSONL, TXT, CSV, Parquet or a folder on the Datasets page.</div>}
                <table className="t"><tbody>
                  {datasets.data?.map((d) => {
                    const s = dsSel.find((x) => x.id === d.id);
                    return (
                      <tr key={d.id}>
                        <td style={{ width: 28 }}><input type="checkbox" checked={!!s} aria-label={`use ${d.name}`} onChange={(e) => setDsSel(e.target.checked ? [...dsSel, { id: d.id, weight: 1 }] : dsSel.filter((x) => x.id !== d.id))} /></td>
                        <td><b>{d.name}</b><div className="dim small">{fmt.int(d.stats?.samples)} samples · {fmt.bytes(d.stats?.bytes)}{d.stats?.tokens ? ` · ${fmt.int(d.stats.tokens)} tokens` : ""} · {d.stats?.kinds?.chat ? "chat" : "text"}</div></td>
                        <td style={{ width: 120 }}>{s && <Field label="Mix weight"><Num value={s.weight} onChange={(v) => setDsSel(dsSel.map((x) => (x.id === d.id ? { ...x, weight: v } : x)))} min={0} /></Field>}</td>
                      </tr>
                    );
                  })}
                </tbody></table>
              </div>
              <details className="more" style={{ marginTop: 4 }}><summary>More data options</summary><div>
              <div className="card">
                <h3>Preparation</h3>
                <div className="grid g4">
                  <Field label="Validation split"><Num value={prep.val_ratio} onChange={(v) => setPrep({ ...prep, val_ratio: v })} min={0} max={0.5} /></Field>
                  <Field label="Min sequence tokens"><Num value={prep.min_tokens} onChange={(v) => setPrep({ ...prep, min_tokens: v })} min={1} /></Field>
                  <Field label="Max sequence tokens" hint="0 = no limit (long docs are split)"><Num value={prep.max_tokens} onChange={(v) => setPrep({ ...prep, max_tokens: v })} min={0} /></Field>
                  <Field label="Sequence packing"><Check checked={prep.packing} onChange={(v) => setPrep({ ...prep, packing: v })}>Pack documents into full windows</Check></Field>
                </div>
                <div className="dim small" style={{ marginTop: 6 }}>Chat datasets train only on assistant replies (prompts are masked from the loss).</div>
              </div>
              </div></details>
              {scratch && (
                <div className="card">
                  <h3>Tokenizer<span className="r"><Seg value={tokMode} onChange={setTokMode} options={[{ value: "new", label: "Train new" }, { value: "existing", label: "Use existing" }]} /></span></h3>
                  {tokMode === "new" ? (
                    <div className="grid g3">
                      <Field label="Type" hint={tokKind === "wordpiece" ? "WordPiece drops whitespace detail - not for code" : tokKind === "bpe" ? "byte-level: lossless, GGUF-compatible" : ""}>
                        <Sel value={tokKind} onChange={setTokKind} options={[{ value: "bpe", label: "BPE (byte-level)" }, { value: "wordpiece", label: "WordPiece" }, { value: "unigram", label: "Unigram" }, { value: "sentencepiece", label: "SentencePiece" }]} />
                      </Field>
                      <Field label="Vocabulary size"><Num value={tokVocab} onChange={setTokVocab} min={256} /></Field>
                      <Field label="Special tokens"><div className="mono small dim">&lt;|pad|&gt; &lt;|bos|&gt; &lt;|eos|&gt; &lt;|unk|&gt; + chat roles</div></Field>
                    </div>
                  ) : (
                    <Field label="Tokenizer"><Sel value={tokId} onChange={setTokId} options={[{ value: "", label: "Choose…" }, ...(tokenizers.data || []).map((t) => ({ value: t.id, label: `${t.name} · ${t.kind} · ${t.vocab_size}` }))]} /></Field>
                  )}
                </div>
              )}
            </>
          )}

          {step === 4 && training && mode === "auto" && (
            <div className="card">
              <h3>Training plan<span className="r"><button className="btn sm" onClick={() => setMode("custom")}>Customize</button></span></h3>
              {rec?.notes?.map((n: string) => <div key={n} className="note warn" style={{ marginBottom: 10 }}>{n}</div>)}
              <div className="grid g3" style={{ rowGap: 20 }}>
                <div className="stat"><span className="k">Training steps</span><span className="v">{fmt.int(training.max_steps)}</span></div>
                <div className="stat"><span className="k">Batch</span><span className="v">{training.micro_batch_size * training.gradient_accumulation}</span></div>
                <div className="stat"><span className="k">Learning rate</span><span className="v">{training.learning_rate}</span></div>
                <div className="stat"><span className="k">Precision</span><span className="v">{String(training.precision).toUpperCase()}</span></div>
                <div className="stat"><span className="k">Memory</span><span className="v">{fmt.bytes(estimate?.memory?.total ?? rec?.estimates?.vram_bytes)}</span></div>
                <div className="stat"><span className="k">Time (est.)</span><span className="v">{fmt.dur(rec?.estimates?.duration_s)}</span></div>
              </div>
              <div className="dim small" style={{ marginTop: 14 }}>Chosen automatically for your hardware. Checkpoints are saved regularly and when you pause or stop.</div>
            </div>
          )}
          {step === 4 && training && mode === "custom" && (
            <>
              <div className="card">
                <h3>Automatic hardware optimization<span className="r"><Seg value={mode} onChange={setMode} options={[{ value: "auto", label: "AUTO" }, { value: "custom", label: "CUSTOM" }]} /></span></h3>
                {rec?.notes?.map((n: string) => <div key={n} className="note warn" style={{ marginBottom: 6 }}>{n}</div>)}
                <div className="grid g4">
                  <Field label="Precision"><Sel value={training.precision} onChange={(v) => setT("precision", v)} options={["fp32", "tf32", "fp16", "bf16"]} /></Field>
                  <Field label="Epochs"><Num value={training.epochs} onChange={(v) => setT("epochs", v)} min={1} /></Field>
                  <Field label="Training steps" hint="0 = from epochs"><Num value={training.max_steps} onChange={(v) => setT("max_steps", v)} min={0} /></Field>
                  <Field label="Context length"><Num value={training.context_length} onChange={(v) => setT("context_length", v)} min={16} /></Field>
                  <Field label="Micro batch size"><Num value={training.micro_batch_size} onChange={(v) => setT("micro_batch_size", v)} min={1} /></Field>
                  <Field label="Gradient accumulation"><Num value={training.gradient_accumulation} onChange={(v) => setT("gradient_accumulation", v)} min={1} /></Field>
                  <Field label="Batch size" hint="micro × accumulation"><input className="in mono" disabled value={training.micro_batch_size * training.gradient_accumulation} /></Field>
                  <Field label="Learning rate"><Num value={training.learning_rate} onChange={(v) => setT("learning_rate", v)} /></Field>
                  <Field label="Warmup steps"><Num value={training.warmup_steps} onChange={(v) => setT("warmup_steps", v)} min={0} /></Field>
                  <Field label="Weight decay"><Num value={training.weight_decay} onChange={(v) => setT("weight_decay", v)} min={0} /></Field>
                  <Field label="Gradient clipping"><Num value={training.grad_clip} onChange={(v) => setT("grad_clip", v)} min={0} /></Field>
                  <Field label="Dropout"><Num value={training.dropout} onChange={(v) => setT("dropout", v)} min={0} max={0.9} /></Field>
                  <Field label="Optimizer"><Sel value={training.optimizer} onChange={(v) => setT("optimizer", v)} options={[{ value: "adam", label: "Adam" }, { value: "adamw", label: "AdamW" }, { value: "adamw8bit", label: "AdamW 8-bit" }, { value: "adafactor", label: "Adafactor" }, { value: "lion", label: "Lion" }, { value: "sgd", label: "SGD" }]} /></Field>
                  <Field label="Scheduler"><Sel value={training.scheduler} onChange={(v) => setT("scheduler", v)} options={[{ value: "constant", label: "Constant" }, { value: "linear", label: "Linear" }, { value: "cosine", label: "Cosine" }, { value: "cosine_restarts", label: "Cosine with restarts" }, { value: "polynomial", label: "Polynomial" }, { value: "one_cycle", label: "One Cycle" }]} /></Field>
                  <Field label="DataLoader workers"><Num value={training.dataloader_workers} onChange={(v) => setT("dataloader_workers", v)} min={1} max={32} /></Field>
                  <Field label="Seed"><Num value={training.seed} onChange={(v) => setT("seed", v)} /></Field>
                  <Field label="Evaluate every N steps"><Num value={training.eval_every} onChange={(v) => setT("eval_every", v)} min={0} /></Field>
                  <Field label="Gradient checkpointing"><Check checked={!!training.gradient_checkpointing} onChange={(v) => setT("gradient_checkpointing", v)}>Recompute activations</Check></Field>
                </div>
              </div>
              {(method === "lora" || method === "qlora") && training.lora && (
                <div className="card">
                  <h3>{method === "qlora" ? "QLoRA" : "LoRA"}</h3>
                  <div className="grid g4">
                    <Field label="Rank"><Sel value={[1, 2, 4, 8, 16, 32, 64, 128].includes(training.lora.rank) ? training.lora.rank : "custom" as any} onChange={(v: any) => v !== "custom" && setL("rank", v)} options={[...[1, 2, 4, 8, 16, 32, 64, 128].map((r) => ({ value: r, label: String(r) })), { value: "custom" as any, label: "Custom" }]} /></Field>
                    <Field label="Custom rank"><Num value={training.lora.rank} onChange={(v) => setL("rank", v)} min={1} /></Field>
                    <Field label="Alpha"><Num value={training.lora.alpha} onChange={(v) => setL("alpha", v)} min={1} /></Field>
                    <Field label="Dropout"><Num value={training.lora.dropout} onChange={(v) => setL("dropout", v)} min={0} max={0.9} /></Field>
                    <Field label="Bias"><Sel value={training.lora.bias} onChange={(v) => setL("bias", v)} options={["none", "all", "lora_only"]} /></Field>
                    <Field label="Base quantization"><Sel value={training.base_quant || "bf16"} onChange={(v) => setT("base_quant", v)} options={method === "qlora" ? ["nf4", "fp4", "int4", "int8"] : ["bf16", "fp16", "int8", "nf4", "fp4"]} /></Field>
                    <Field label="Double quantization"><Check checked={!!training.lora.double_quant} onChange={(v) => setL("double_quant", v)}>Quantize the quantization constants</Check></Field>
                  </div>
                  <div className="lbl" style={{ marginTop: 10 }}>Target modules</div>
                  <div className="row" style={{ gap: 12 }}>{TARGETS.map((t) => <Check key={t} checked={training.lora.target_modules.includes(t)} onChange={(v) => setL("target_modules", v ? [...training.lora.target_modules, t] : training.lora.target_modules.filter((x: string) => x !== t))}><span className="mono">{t}</span></Check>)}</div>
                </div>
              )}
              {method === "adapter" && training.adapter && (
                <div className="card"><h3>Adapters</h3>
                  <Field label="Bottleneck size"><Num value={training.adapter.bottleneck} onChange={(v) => setT("adapter", { bottleneck: v })} min={4} /></Field>
                </div>
              )}
              <div className="card">
                <h3>Checkpoints</h3>
                <div className="grid g4">
                  <Field label="Save every N steps" hint="0 = only best/final"><Num value={ckpt.save_every} onChange={(v) => setCkpt({ ...ckpt, save_every: v })} min={0} /></Field>
                  <Field label="Maximum checkpoints" hint="older automatic ones are cleaned up"><Num value={ckpt.keep} onChange={(v) => setCkpt({ ...ckpt, keep: v })} min={1} /></Field>
                  <Field label="Best checkpoint"><Check checked={ckpt.save_best} onChange={(v) => setCkpt({ ...ckpt, save_best: v })}>Keep lowest val loss</Check></Field>
                  <Field label="On kill / pause"><Check checked={ckpt.save_on_kill} onChange={(v) => setCkpt({ ...ckpt, save_on_kill: v, checkpoint_on_pause: v })}>Save a safe checkpoint</Check></Field>
                </div>
              </div>
            </>
          )}

          {step === 5 && model && training && (
            <div className="card">
              <h3>Review</h3>
              <dl className="kv">
                <dt>AI</dt><dd>{id.name} {id.version} · makeai://{id.creator_username}/{id.name.toLowerCase().replace(/[^a-z0-9._-]+/g, "-")}/{id.version}</dd>
                <dt>Created by</dt><dd>{id.creator_name} (@{id.creator_username})</dd>
                <dt>Method</dt><dd>{METHODS.find((m) => m.value === method)?.label}{!scratch && ` on ${baseModels.find((m) => m.uid === base)?.name}`}</dd>
                <dt>Complexity</dt><dd>{complexity} · {opts.data?.complexity?.[complexity]?.label} ({mode.toUpperCase()})</dd>
                <dt>Architecture</dt><dd>{model.n_layers} layers · hidden {model.hidden_size} · {model.n_heads}/{model.n_kv_heads} heads · ctx {model.context_length} · {model.activation} · {model.norm} · {model.pos_encoding}</dd>
                <dt>Parameters</dt><dd>{fmt.int(estimate?.params ?? est?.params)} ({fmt.params(estimate?.params ?? est?.params)})</dd>
                <dt>Datasets</dt><dd>{dsSel.map((d) => datasets.data?.find((x) => x.id === d.id)?.name).join(", ")}</dd>
                <dt>Tokenizer</dt><dd>{scratch ? (tokMode === "new" ? `new ${tokKind}, vocab ${tokVocab}` : tokenizers.data?.find((t) => t.id === tokId)?.name) : "from base model"}</dd>
                <dt>Training</dt><dd>{training.precision.toUpperCase()} · {training.optimizer} · lr {training.learning_rate} · {training.scheduler} · batch {training.micro_batch_size}×{training.gradient_accumulation} · {training.max_steps ? `${fmt.int(training.max_steps)} steps` : `${training.epochs} epochs`}</dd>
              </dl>
              {error && <div className="note bad" style={{ marginTop: 12 }}>{error}</div>}
              <div className="row" style={{ marginTop: 16, justifyContent: "flex-end" }}>
                {busy && <span className="row dim"><Spinner />{busy}</span>}
                <button className="btn" disabled={!!busy || !valid.every(Boolean)} onClick={() => start(false)}>Create without training</button>
                <button className="btn primary lg" disabled={!!busy || !valid.every(Boolean) || memFits === false} onClick={() => start(true)}><Icon n="play" s={14} />Start training</button>
              </div>
              {memFits === false && <div className="note warn" style={{ marginTop: 8 }}>The estimated memory exceeds the free VRAM. Lower the micro batch size or context, enable gradient checkpointing, or choose QLoRA.</div>}
            </div>
          )}

          <div className="row">
            <button className="btn" disabled={step === 0} onClick={() => setStep(step - 1)}>Back</button>
            <span className="sp" />
            {step < STEPS.length - 1 && <button className="btn primary" disabled={!valid[step]} onClick={() => setStep(step + 1)}>Next</button>}
          </div>
        </div>

        <aside className="col" style={{ position: "sticky", top: 0 }}>
          <div className="card">
            <h3><Icon n="cpu" s={15} />Estimate</h3>
            {recErr && <div className="note bad">{recErr}</div>}
            {!rec && !recErr && <div className="dim small">{scratch || base ? <Spinner /> : "Choose a base model."}</div>}
            {rec && (
              <dl className="kv">
                <dt>Device</dt><dd>{rec.device.name}</dd>
                <dt>Parameters</dt><dd>{fmt.params(estimate?.params ?? est.params)}</dd>
                <dt>Trainable</dt><dd>{fmt.params(estimate?.memory?.trainable_params ?? est.trainable_params)}</dd>
                <dt>VRAM</dt><dd className={memFits === false ? "bad-t" : ""}>{fmt.bytes(estimate?.memory?.total ?? est.vram_bytes)} {estimate?.vram_free_bytes ? `/ ${fmt.bytes(estimate.vram_free_bytes)} free` : ""}</dd>
                <dt>RAM</dt><dd>{fmt.bytes(est.ram_bytes)}</dd>
                <dt>Duration</dt><dd>{mode === "auto" ? `~${fmt.dur(est.duration_s)}` : "re-estimated when training starts"}</dd>
                <dt>Tokens needed</dt><dd>{rec.dataset.recommended_tokens ? `${fmt.params(rec.dataset.recommended_tokens)} recommended` : "task-dependent"}</dd>
                <dt>Dataset</dt><dd>{rec.dataset.dataset_tokens ? `${fmt.params(rec.dataset.dataset_tokens)} tokens (est.)` : "–"}</dd>
                <dt>Speed basis</dt><dd className="small" style={{ fontFamily: "var(--sans)" }}>{est.throughput_basis}</dd>
              </dl>
            )}
            {estimate?.memory && (
              <div style={{ marginTop: 10 }}>
                <div className="lbl">VRAM breakdown</div>
                {["weights", "grads", "optimizer", "activations", "logits", "overhead"].map((k) => (
                  <div key={k} className="row small" style={{ justifyContent: "space-between" }}><span className="dim">{k}</span><span className="mono">{fmt.bytes(estimate.memory[k])}</span></div>
                ))}
              </div>
            )}
            <div className="faint small" style={{ marginTop: 10 }}>Analytic estimate. The live dashboard shows measured memory and throughput once training runs.</div>
          </div>
        </aside>
      </div>
    </div>
  );
}
