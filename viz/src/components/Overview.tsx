import type { RunData } from "../lib/types";
import { bestIteration, primaryOf } from "../lib/derive";
import { duration, int, score, signed, truncate } from "../lib/format";
import { Empty, IterLink, KeyValue, Note, Panel, Row, Stat, StatGrid } from "@/components/common";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
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
    <div className="flex flex-col gap-4">
      {!meta && (
        <Note tone="warning">
          This run has no <code>run_meta.json</code>. The figures below are recomputed from{" "}
          <code>ledger.jsonl</code>, so wall-clock time and the submitted iteration are unknown.
        </Note>
      )}

      <Panel title="Run">
        <KeyValue>
          <Row k="dataset">{meta?.dataset ?? "unknown"}</Row>
          <Row k="model">
            <span className="font-mono">
              {meta?.model ?? "unknown"}
              {meta?.provider ? ` (${meta.provider})` : ""}
            </span>
          </Row>
          {meta?.data_contract && (
            <Row k="data contract">
              <span className="font-mono">{meta.data_contract}</span>
            </Row>
          )}
          <Row k="stopped">
            {meta?.stop_reason ?? "unknown"}
            {meta?.iterations !== undefined && meta?.iteration_cap !== undefined
              ? ` after ${meta.iterations} of ${meta.iteration_cap} iterations`
              : ` after ${derived.iterations} iterations`}
            {meta?.strict_convergence_iteration !== undefined
              ? ` (strict convergence at #${meta.strict_convergence_iteration})`
              : ""}
          </Row>
          {meta?.api_surface && (
            <Row k="API surface">
              <span className="text-ink-2 font-mono text-xs">{meta.api_surface.join("  ·  ")}</span>
            </Row>
          )}
        </KeyValue>
      </Panel>

      <Panel title="Result">
        <StatGrid>
          <Stat
            label="Iterations"
            value={meta?.iterations ?? derived.iterations}
            note={`of ${meta?.iteration_cap ?? "?"} cap`}
          />
          <Stat
            label="Baseline"
            value={score(reproduced)}
            note={target !== undefined ? `target ${score(target)}` : "no target recorded"}
            tone={
              reproduced !== undefined && target !== undefined && reproduced >= target
                ? "good"
                : undefined
            }
          />
          <Stat
            label="Best validation"
            value={score(best ? primaryOf(best) : undefined)}
            note={best ? `iteration #${best.iter_id}` : undefined}
          />
          <Stat
            label="Wall clock"
            value={duration(meta?.wall_clock_s)}
            note={`scripts ${duration(meta?.script_seconds ?? derived.scriptSeconds)}`}
          />
          <Stat
            label="Tokens"
            value={int(tokensTotal)}
            note={`${int(meta?.tokens_in ?? derived.tokensIn)} in / ${int(meta?.tokens_out ?? derived.tokensOut)} out`}
          />
          <Stat
            label="Script failures"
            value={derived.scriptFailures}
            tone={derived.scriptFailures > 0 ? "bad" : "good"}
            note="the agent's own code errored"
          />
          {derived.infraFailures > 0 && (
            <Stat
              label="API outages"
              value={derived.infraFailures}
              note="iterations lost to the LLM transport before any code ran"
            />
          )}
          <Stat
            label="Manual interventions"
            value={meta?.manual_interventions ?? "--"}
            tone={meta?.manual_interventions === 0 ? "good" : undefined}
          />
          {meta?.integrity_rejections !== undefined && (
            <Stat
              label="Integrity rejections"
              value={meta.integrity_rejections}
              note="proposals refused by the harness"
            />
          )}
          {meta?.candidates_evaluated !== undefined && (
            <Stat
              label="Candidates evaluated"
              value={meta.candidates_evaluated}
              note="variants tried inside scripts"
            />
          )}
          {meta?.claims_established !== undefined && (
            <Stat
              label="Claims established"
              value={meta.claims_established}
              note="beliefs the agent kept"
            />
          )}
        </StatGrid>
      </Panel>

      <Panel title="Submission" className={submission ? "ring-primary/20 ring-1" : undefined}>
        {submission ? (
          <div className="flex flex-col gap-3.5">
            <StatGrid>
              <Stat
                label="From iteration"
                value={`#${submission.iter_id}`}
                note="chosen on validation only"
              />
              <Stat label="Validation primary" value={score(submission.valid_primary)} />
              <Stat label="Test primary" value={score(submission.test_primary)} />
              <Stat
                label="Delta vs baseline"
                value={signed(submission.test_delta)}
                tone={(submission.test_delta ?? 0) > 0 ? "good" : "bad"}
                note="on the hidden test split"
              />
              <Stat label="Test GAUC" value={score(submission.test_gauc)} />
              <Stat label="Test nDCG@5" value={score(submission["test_ndcg@5"])} />
            </StatGrid>
            {submission.hypothesis && (
              <p className="max-w-[78ch] text-[15px] leading-relaxed">
                <span className="text-muted-foreground">Winning hypothesis - </span>
                {submission.hypothesis}
              </p>
            )}
            {submittedIter && (
              <Button
                variant="outline"
                size="xs"
                className="w-fit"
                onClick={() => onSelectIteration(submission.iter_id)}
              >
                open iteration #{submission.iter_id}
                <ArrowRight />
              </Button>
            )}
          </div>
        ) : (
          <Empty>
            This run recorded no submission.
            {best && (
              <>
                {" "}
                Its best validation iteration was{" "}
                <IterLink id={best.iter_id} onClick={onSelectIteration} /> ·{" "}
                {score(primaryOf(best))}
              </>
            )}
          </Empty>
        )}
      </Panel>

      <Panel>
        <ProgressChart
          data={data}
          selected={submission?.iter_id ?? null}
          onSelect={onSelectIteration}
        />
      </Panel>

      {best?.hypothesis && !submission && (
        <Panel title="Best iteration">
          <p className="max-w-[78ch] text-[15px] leading-relaxed">
            {truncate(best.hypothesis, 400)}
          </p>
        </Panel>
      )}
    </div>
  );
}
