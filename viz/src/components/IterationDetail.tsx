import { useEffect, useMemo, useState } from "react";
import type { Iteration, RunData } from "../lib/types";
import { infraKind } from "../lib/derive";
import { duration, int, score, timestamp } from "../lib/format";
import { Card, Empty, StatusPill } from "./ui";
import ScriptView from "./ScriptView";

const METRIC_ORDER = ["primary", "gauc", "ndcg@5", "ndcg@10", "recall@50", "raw_candidate_primary", "harness_blend_alpha"];

type Tab = "script" | "candidates" | "diagnostics" | "ensemble" | "prompts";

function Metrics({ it }: { it: Iteration }) {
  const m = it.metrics ?? {};
  const keys = [...METRIC_ORDER.filter((k) => m[k] !== undefined), ...Object.keys(m).filter((k) => !METRIC_ORDER.includes(k) && k !== "gpu_seconds")];
  if (!keys.length) return <p className="muted">No metrics recorded — the script did not finish.</p>;
  return (
    <table className="kv">
      <tbody>
        {keys.map((k) => (
          <tr key={k}>
            <td>{k}</td>
            <td className="mono">{typeof m[k] === "number" ? score(m[k] as number) : String(m[k])}</td>
          </tr>
        ))}
        <tr>
          <td>script time</td>
          <td className="mono">{duration(it.gpu_seconds)}</td>
        </tr>
        <tr>
          <td>tokens</td>
          <td className="mono">
            {int(it.tokens_in)} in / {int(it.tokens_out)} out
          </td>
        </tr>
        <tr>
          <td>finished</td>
          <td className="mono">{timestamp(it.timestamp)}</td>
        </tr>
      </tbody>
    </table>
  );
}

function Candidates({ candidates }: { candidates: Record<string, number> }) {
  const entries = Object.entries(candidates).sort((a, b) => b[1] - a[1]);
  const values = entries.map(([, v]) => v);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  // Bars start at a floor just below the worst candidate: these scores differ in the 3rd
  // decimal, and a zero-baseline bar would render them all identical.
  const floor = lo - (hi - lo || 0.001) * 0.25;

  return (
    <>
      <p className="secondary" style={{ marginTop: 0 }}>
        Variants this one script built and scored internally. Bars are drawn from {score(floor)}, not
        zero, so differences at the fourth decimal stay visible.
      </p>
      <div className="bars">
        {entries.map(([name, v], i) => (
          <div className={`bar-row${i === 0 ? " best" : ""}`} key={name}>
            <span className="name" title={name}>
              {name}
            </span>
            <span className="track">
              <span className="fill" style={{ width: `${((v - floor) / (hi - floor)) * 100}%` }} />
            </span>
            <span className="num">{score(v)}</span>
          </div>
        ))}
      </div>
    </>
  );
}

export default function IterationDetail({ data, iteration }: { data: RunData; iteration?: Iteration }) {
  const [tab, setTab] = useState<Tab>("script");

  const available = useMemo<Tab[]>(
    () =>
      iteration
        ? ([
            "script",
            iteration.candidates ? "candidates" : null,
            iteration.diagnostics ? "diagnostics" : null,
            iteration.ensemble ? "ensemble" : null,
            iteration.llmCalls.length ? "prompts" : null,
          ].filter(Boolean) as Tab[])
        : [],
    [iteration],
  );

  useEffect(() => {
    if (available.length && !available.includes(tab)) setTab(available[0]);
  }, [available, tab]);

  if (!iteration) return <Empty>Pick an iteration on the left.</Empty>;

  const parent = iteration.parent_iter_id === null ? undefined : data.byId.get(iteration.parent_iter_id);
  const ens = iteration.ensemble;

  return (
    <div>
      <Card>
        <div className="meta-row">
          <span className="pill">
            <span className="dot" />#{iteration.iter_id}
          </span>
          <StatusPill status={iteration.status} infra={iteration.infra} />
          <span className="pill">{iteration.phase}</span>
          {parent && <span className="muted">from #{parent.iter_id}</span>}
          {iteration.children.length > 0 && (
            <span className="muted">
              → {iteration.children.map((c) => `#${c}`).join(", ")}
            </span>
          )}
        </div>

        <p className="hypothesis">{iteration.hypothesis ?? "No hypothesis recorded."}</p>

        {iteration.error && (
          <div className={`error-box${iteration.infra ? " infra" : ""}`}>
            <div className="label">
              {iteration.infra
                ? `${infraKind(iteration.error)} — the LLM call failed, so this iteration never ran`
                : "error"}
            </div>
            <pre>{iteration.error}</pre>
          </div>
        )}

        <Metrics it={iteration} />
      </Card>

      <Card>
        <div className="tabs">
          {available.map((t) => (
            <button key={t} type="button" aria-current={tab === t} onClick={() => setTab(t)}>
              {t === "prompts" ? `prompts (${iteration.llmCalls.length})` : t}
            </button>
          ))}
        </div>

        {tab === "script" && (
          <ScriptView
            source={iteration.script}
            path={iteration.scriptPath}
            parentSource={parent?.script}
            parentLabel={parent ? `#${parent.iter_id}` : undefined}
          />
        )}

        {tab === "candidates" && iteration.candidates && <Candidates candidates={iteration.candidates} />}

        {tab === "diagnostics" && iteration.diagnostics && (
          <pre className="text">{iteration.diagnostics.trimEnd()}</pre>
        )}

        {tab === "ensemble" && ens && (
          <>
            <table className="kv">
              <tbody>
                <tr>
                  <td>selected alpha</td>
                  <td className="mono">{ens.selected_alpha ?? "--"}</td>
                </tr>
                <tr>
                  <td>candidate primary</td>
                  <td className="mono">{score(ens.candidate_primary)}</td>
                </tr>
                <tr>
                  <td>incumbent primary</td>
                  <td className="mono">{score(ens.incumbent_primary)}</td>
                </tr>
                <tr>
                  <td>selected primary</td>
                  <td className="mono">{score(ens.selected_primary)}</td>
                </tr>
              </tbody>
            </table>
            {ens.grid && (
              <table className="data">
                <thead>
                  <tr>
                    <th>alpha</th>
                    <th>primary</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(ens.grid)
                    .sort((a, b) => Number(a[0]) - Number(b[0]))
                    .map(([alpha, v]) => (
                      <tr key={alpha}>
                        <td>{alpha}</td>
                        <td>
                          {score(v)}
                          {ens.selected_alpha !== undefined && Number(alpha) === ens.selected_alpha ? "  ← selected" : ""}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            )}
          </>
        )}

        {tab === "prompts" && (
          <>
            <p className="secondary" style={{ marginTop: 0 }}>
              Attributed by order — <code>llm_calls.jsonl</code> carries no iteration id.
            </p>
            {iteration.llmCalls.map((idx) => {
              const call = data.llmCalls[idx];
              return (
                <details className="call" key={idx}>
                  <summary>
                    <span className="pill">{call.role}</span>
                    <span className="preview">{call.response.slice(0, 160)}</span>
                    <span className="tokens">
                      {int(call.tokens_in)} → {int(call.tokens_out)}
                    </span>
                  </summary>
                  <div className="body">
                    <div>
                      <h4>prompt</h4>
                      <pre className="text">{call.prompt}</pre>
                    </div>
                    <div>
                      <h4>response</h4>
                      <pre className="text">{call.response}</pre>
                    </div>
                  </div>
                </details>
              );
            })}
          </>
        )}
      </Card>
    </div>
  );
}
