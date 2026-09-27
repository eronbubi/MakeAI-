import React, { useEffect, useRef, useState } from "react";
import { api, Job } from "../api";
import { Spinner } from "../components/ui";
import { useApp, useFetch } from "../store";

/** Optional Claude development agent. Develops MakeAI itself; not used by training or inference. */
export default function Develop() {
  const { toast, log } = useApp();
  const st = useFetch<any>("/api/dev/status");
  const [task, setTask] = useState("");
  const [phase, setPhase] = useState<"idle" | "planning" | "review" | "running" | "done">("idle");
  const [plan, setPlan] = useState<any>(null);
  const [editing, setEditing] = useState(false);
  const [planText, setPlanText] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [result, setResult] = useState<any>(null);
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => { logRef.current && (logRef.current.scrollTop = logRef.current.scrollHeight); }, [lines]);

  const poll = async (j: Job, onDone: (r: any) => void) => {
    for (;;) {
      const cur = await api.get<Job>(`/api/jobs/${j.id}`);
      setJob(cur);
      setLines(cur.log || []);
      if (cur.state === "done") { onDone(cur.result); return; }
      if (cur.state !== "running") { toast(cur.error || cur.state, true); setPhase("idle"); return; }
      await new Promise((r) => setTimeout(r, 800));
    }
  };
  const analyze = async () => {
    setPhase("planning"); setPlan(null); setResult(null); setLines([]);
    try { const j = await api.post("/api/dev/plan", { task }); await poll(j, (r) => { setPlan(r); setPlanText(r.plan); setPhase("review"); log("dev", "plan ready"); }); }
    catch (e: any) { toast(e.message, true); setPhase("idle"); }
  };
  const approve = async () => {
    setPhase("running"); setLines([]);
    try { const j = await api.post("/api/dev/execute", { plan: planText, session_id: plan?.session_id }); await poll(j, (r) => { setResult(r); setPhase("done"); }); }
    catch (e: any) { toast(e.message, true); setPhase("review"); }
  };
  const cancel = async () => { if (job && job.state === "running") await api.post(`/api/jobs/${job.id}/cancel`); setPhase("idle"); setPlan(null); };

  if (!st.data) return <div className="page"><Spinner /></div>;
  return (
    <div className="page">
      <div className="head"><h1 className="title">Development Agent</h1><span className="sub">Claude helps build and improve MakeAI. It is not part of the training or inference runtime.</span></div>
      {!st.data.available ? (
        <div className="note warn">{st.data.reason}. Everything else in MakeAI works without it: create, train, evaluate, run and share AIs locally.</div>
      ) : (
        <>
          <div className="card">
            <div className="dim small" style={{ marginBottom: 6 }}>Source: <span className="mono">{st.data.source_root}</span> · CLI: <span className="mono">{st.data.cli}</span></div>
            <textarea className="in" rows={3} value={task} onChange={(e) => setTask(e.target.value)} placeholder="e.g. Add multi-GPU data-parallel training and show per-GPU stats on the training dashboard" disabled={phase === "planning" || phase === "running"} aria-label="Change request" />
            <div className="row" style={{ marginTop: 8 }}>
              <button className="btn primary" disabled={!task.trim() || phase === "planning" || phase === "running"} onClick={analyze}>{phase === "planning" ? <><Spinner /> Analyzing project…</> : "Plan change"}</button>
              <span className="dim small">Planning is read-only. Code is only written after you approve.</span>
            </div>
          </div>
          {(phase === "review" || phase === "running" || phase === "done") && plan && (
            <div className="card" style={{ marginTop: 12 }}>
              <h3>Plan{plan.cost_usd != null && <span className="r">planning cost ${plan.cost_usd.toFixed(3)}</span>}</h3>
              {editing ? <textarea className="editor" style={{ minHeight: 260 }} value={planText} onChange={(e) => setPlanText(e.target.value)} /> : <div className="plan">{planText}</div>}
              {phase === "review" && (
                <div className="row" style={{ marginTop: 12 }}>
                  <button className="btn primary" onClick={approve}>APPROVE</button>
                  <button className="btn" onClick={() => setEditing(!editing)}>{editing ? "DONE EDITING" : "EDIT PLAN"}</button>
                  <button className="btn" onClick={cancel}>CANCEL</button>
                </div>
              )}
            </div>
          )}
          {(phase === "planning" || phase === "running" || phase === "done") && (
            <div className="card" style={{ marginTop: 12 }}>
              <h3>{phase === "running" ? <><Spinner /> Implementing…</> : phase === "done" ? "Finished" : "Agent output"}{phase === "running" && <span className="r"><button className="btn sm danger" onClick={cancel}>Stop</button></span>}</h3>
              <div ref={logRef} className="logs-body" style={{ height: 300, border: "1px solid var(--line)", borderRadius: 6, background: "var(--bg)" }}>{lines.map((l, i) => <div key={i}>{l}</div>)}</div>
              {result && <div className="plan" style={{ marginTop: 10 }}>{result.result}<div className="faint small">exit {result.exit_code}{result.cost_usd != null ? ` · $${result.cost_usd.toFixed(3)}` : ""}</div></div>}
            </div>
          )}
        </>
      )}
    </div>
  );
}
