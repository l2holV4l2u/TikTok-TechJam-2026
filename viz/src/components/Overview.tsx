import type { RunData } from "../lib/types";
import { bestIteration, primaryOf } from "../lib/derive";
import { duration, int, score, signed, truncate } from "../lib/format";
import { Card, Empty, Tile } from "./ui";
import ProgressChart from "./ProgressChart";

export default function Overview({
  data,
  onSelectIteration,
}: {
  data: RunData;
  onSelectIteration: (id: number) => void;
}) {
  const { meta, iterations } = data;

  // r39 has no run_meta.json. Everything below then comes from the ledger, and says so.
  const derived = {
    iterations: iterations.length,
    tokensIn: iterations.reduce((a, it) => a + (it.tokens_in ?? 0), 0),
    tokensOut: iterations.reduce((a, it) => a + (it.tokens_out ?? 0), 0),
    scriptSeconds: iterations.reduce((a, it) => a + (it.gpu_seconds ?? 0), 0),
    // report_run.py draws this same line: a run that lost its API key did not fail six
    // experiments, it failed to start six. run_meta.json's `failures` lumps them together.
    scriptFailures: iterations.filter((it) => it.status === "failed" && !it.infra).length,
    infraFailures: iterations.filter((it) => it.infra).length,
  };

  const best = bestIteration(iterations);
  const submission = meta?.submission;
  const submittedIter = submission ? data.byId.get(submission.iter_id) : undefined;

  const tokensTotal = meta?.tokens_total ?? derived.tokensIn + derived.tokensOut;
  const reproduced = meta?.baseline_reproduced;
  const target = meta?.baseline_target;

  return (
    <>
      {!meta && (
        <div className="warnings">
          This run has no <code>run_meta.json</code>. The figures below are recomputed from{" "}
          <code>ledger.jsonl</code>, so wall-clock time and the submitted iteration are unknown.
        </div>
      )}

      <Card title="Run">
        <table className="kv">
          <tbody>
            <tr>
              <td>dataset</td>
              <td>{meta?.dataset ?? "unknown"}</td>
            </tr>
            <tr>
              <td>model</td>
              <td className="mono">
                {meta?.model ?? "unknown"}
                {meta?.provider ? ` (${meta.provider})` : ""}
              </td>
            </tr>
            {meta?.data_contract && (
              <tr>
                <td>data contract</td>
                <td className="mono">{meta.data_contract}</td>
              </tr>
            )}
            <tr>
              <td>stopped</td>
              <td>
                {meta?.stop_reason ?? "unknown"}
                {meta?.iterations !== undefined && meta?.iteration_cap !== undefined
                  ? ` after ${meta.iterations} of ${meta.iteration_cap} iterations`
                  : ` after ${derived.iterations} iterations`}
                {meta?.strict_convergence_iteration !== undefined
                  ? ` (strict convergence at #${meta.strict_convergence_iteration})`
                  : ""}
              </td>
            </tr>
            {meta?.api_surface && (
              <tr>
                <td>API surface</td>
                <td className="mono secondary" style={{ fontSize: 12 }}>
                  {meta.api_surface.join("  ·  ")}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Card>

      <div className="tiles" style={{ marginBottom: 16 }}>
        <Tile label="Iterations" value={meta?.iterations ?? derived.iterations} note={`of ${meta?.iteration_cap ?? "?"} cap`} />
        <Tile
          label="Baseline"
          value={score(reproduced)}
          note={target !== undefined ? `target ${score(target)}` : "no target recorded"}
          tone={reproduced !== undefined && target !== undefined && reproduced >= target ? "good" : undefined}
        />
        <Tile label="Best validation" value={score(best ? primaryOf(best) : undefined)} note={best ? `iteration #${best.iter_id}` : undefined} />
        <Tile label="Wall clock" value={duration(meta?.wall_clock_s)} note={`scripts ${duration(meta?.script_seconds ?? derived.scriptSeconds)}`} />
        <Tile label="Tokens" value={int(tokensTotal)} note={`${int(meta?.tokens_in ?? derived.tokensIn)} in / ${int(meta?.tokens_out ?? derived.tokensOut)} out`} />
        <Tile
          label="Script failures"
          value={derived.scriptFailures}
          tone={derived.scriptFailures > 0 ? "bad" : "good"}
          note="the agent's own code errored"
        />
        {derived.infraFailures > 0 && (
          <Tile
            label="API outages"
            value={derived.infraFailures}
            note="iterations lost to the LLM transport before any code ran"
          />
        )}
        <Tile
          label="Manual interventions"
          value={meta?.manual_interventions ?? "--"}
          tone={meta?.manual_interventions === 0 ? "good" : undefined}
        />
        {meta?.integrity_rejections !== undefined && (
          <Tile label="Integrity rejections" value={meta.integrity_rejections} note="proposals refused by the harness" />
        )}
        {meta?.candidates_evaluated !== undefined && (
          <Tile label="Candidates evaluated" value={meta.candidates_evaluated} note="variants tried inside scripts" />
        )}
        {meta?.claims_established !== undefined && (
          <Tile label="Claims established" value={meta.claims_established} note="beliefs the agent kept" />
        )}
      </div>

      <Card title="Submission">
        {submission ? (
          <>
            <div className="tiles" style={{ marginBottom: 14 }}>
              <Tile label="From iteration" value={`#${submission.iter_id}`} note="chosen on validation only" />
              <Tile label="Validation primary" value={score(submission.valid_primary)} />
              <Tile label="Test primary" value={score(submission.test_primary)} />
              <Tile
                label="Delta vs baseline"
                value={signed(submission.test_delta)}
                tone={(submission.test_delta ?? 0) > 0 ? "good" : "bad"}
                note="on the hidden test split"
              />
              <Tile label="Test GAUC" value={score(submission.test_gauc)} />
              <Tile label="Test nDCG@5" value={score(submission["test_ndcg@5"])} />
            </div>
            {submission.hypothesis && (
              <p className="hypothesis">
                <span className="muted">Winning hypothesis — </span>
                {submission.hypothesis}
              </p>
            )}
            {submittedIter && (
              <button type="button" className="chip" onClick={() => onSelectIteration(submission.iter_id)}>
                open iteration #{submission.iter_id}
              </button>
            )}
          </>
        ) : (
          <Empty>
            This run recorded no submission.
            {best && (
              <>
                {" "}
                Its best validation iteration was{" "}
                <button type="button" className="chip" onClick={() => onSelectIteration(best.iter_id)}>
                  #{best.iter_id} · {score(primaryOf(best))}
                </button>
              </>
            )}
          </Empty>
        )}
      </Card>

      <Card>
        <ProgressChart data={data} selected={submission?.iter_id ?? null} onSelect={onSelectIteration} />
      </Card>

      {best?.hypothesis && !submission && (
        <Card title="Best iteration">
          <p className="hypothesis">{truncate(best.hypothesis, 400)}</p>
        </Card>
      )}
    </>
  );
}
