import React, { useState } from "react";
import { api, fmt } from "../api";
import { Check, Spinner } from "../components/ui";
import { useApp, useFetch } from "../store";

export default function Requirements() {
  const { track, toast } = useApp();
  const r = useFetch<any>("/api/requirements");
  const [busy, setBusy] = useState(false);
  const [quick, setQuick] = useState(true);
  const run = async () => {
    setBusy(true);
    try { const res = await track(await api.post("/api/tests/run", { quick }), "Acceptance tests"); toast(`${res.passed} passed, ${res.failed} failed, ${res.skipped} skipped`, res.failed > 0); r.reload(); }
    catch (e: any) { toast(e.message, true); } finally { setBusy(false); }
  };
  const res = r.data?.results;
  const byTest: Record<string, any> = {};
  res?.tests?.forEach((t: any) => { byTest[t.name] = t; });
  const reqs = r.data?.requirements?.requirements || [];
  return (
    <div className="page">
      <div className="head"><h1 className="title">Requirements &amp; Tests</h1>
        <span className="sub">{res?.at ? `last run ${fmt.date(res.at)} · ${res.passed} passed · ${res.failed} failed · ${res.skipped} skipped` : "not run yet"}</span>
        <div className="actions"><Check checked={quick} onChange={setQuick}>Skip slow GPU tests</Check><button className="btn primary" disabled={busy} onClick={run}>{busy ? <Spinner /> : "Run acceptance tests"}</button></div>
      </div>
      <div className="card" style={{ padding: 0 }}>
        <table className="t"><thead><tr><th>Requirement</th><th>Acceptance test</th><th>Expected</th><th>Status</th></tr></thead><tbody>
          {reqs.map((q: any) => {
            const tests = (q.tests || []).map((n: string) => byTest[n]).filter(Boolean);
            const state = !tests.length ? "not run" : tests.some((t: any) => t.state === "failed" || t.state === "error") ? "failed" : tests.every((t: any) => t.state === "skipped") ? "skipped" : "passed";
            return (
              <tr key={q.id}>
                <td><b>{q.name}</b><div className="dim small">{q.module}</div></td>
                <td className="mono small" style={{ whiteSpace: "pre-line" }}>{(q.tests || []).join("\n")}</td>
                <td className="small">{q.expected}</td>
                <td><span className={`pill ${state === "passed" ? "good" : state === "failed" ? "bad" : ""}`}>{state}</span>
                  {tests.filter((t: any) => t.message).map((t: any) => <div key={t.name} className="faint small" style={{ maxWidth: 300 }}>{t.message}</div>)}</td>
              </tr>
            );
          })}
        </tbody></table>
      </div>
    </div>
  );
}
