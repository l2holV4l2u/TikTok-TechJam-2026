import { ShieldCheck } from "lucide-react";
import type { RunIndex } from "@/lib/runIndex";
import { selectable } from "@/lib/runIndex";
import { duration, int, score, signed } from "@/lib/format";
import { Empty, KeyValue, Note, Panel, Row, Stat, StatGrid } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

/** Benchmark-wide constants: properties of KuaiRand-Pure, not of any one run. */
const RANDOM_PRIMARY = 0.4753;
const CEILING_PRIMARY = 0.8645;
const FALLBACK_BASELINE = 0.5946;
const WALLCLOCK_CAP_S = 21600; // 6 h feasibility ceiling

/** Track + tick calibration bar: random, baseline, ceiling, and the submission's own score. */
function CalibrationBar({
  baseline,
  submission,
}: {
  baseline: number;
  submission: number;
}) {
  const span = CEILING_PRIMARY - RANDOM_PRIMARY;
  const pct = (v: number) => Math.min(100, Math.max(0, ((v - RANDOM_PRIMARY) / span) * 100));
  // align: the end ticks sit at 0% and 100%, so centring them would clip half the label off.
  const ticks = [
    { label: "random", value: RANDOM_PRIMARY, align: "items-start translate-x-0" },
    { label: "official baseline", value: baseline, align: "items-center -translate-x-1/2" },
    { label: "perfect ranking", value: CEILING_PRIMARY, align: "items-end -translate-x-full" },
  ];
  const subPct = pct(submission);

  return (
    <div className="pt-6">
      <div className="relative">
        <div
          className="absolute bottom-full mb-1 flex -translate-x-1/2 flex-col items-center"
          style={{ left: `${subPct}%` }}
        >
          <span className="tnum text-foreground font-mono text-xs font-semibold">
            {score(submission)}
          </span>
          <span className="bg-foreground mt-0.5 h-2 w-0.5" />
        </div>
        <div className="bg-secondary relative h-2 rounded">
          <div
            className="bg-good absolute inset-y-0 left-0 rounded"
            style={{ width: `${subPct}%` }}
          />
          {ticks.map((t) => (
            <span
              key={t.label}
              className="border-axis absolute top-0 h-2 border-l"
              style={{ left: `${pct(t.value)}%` }}
            />
          ))}
        </div>
        <div className="relative mt-1.5 h-8">
          {ticks.map((t) => (
            <div
              key={t.label}
              className={`text-muted-foreground absolute top-0 flex flex-col text-[11px] ${t.align}`}
              style={{ left: `${pct(t.value)}%` }}
            >
              <span className="whitespace-nowrap">{t.label}</span>
              <span className="tnum font-mono">{score(t.value)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function BudgetBar({
  label,
  pct,
  usedLabel,
  capLabel,
}: {
  label: string;
  pct: number;
  usedLabel: string;
  capLabel: string;
}) {
  return (
    <div className="grid grid-cols-[92px_minmax(0,1fr)_150px] items-center gap-2.5 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="bg-secondary relative h-3.5 rounded">
        <span
          className="bg-primary absolute inset-y-0 left-0 rounded"
          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        />
      </span>
      <span className="tnum text-right font-mono">
        {usedLabel} / {capLabel}
      </span>
    </div>
  );
}

export default function Verdict({
  index,
  onOpenRun,
}: {
  index: RunIndex;
  onOpenRun: (run: string) => void;
}) {
  const submitted = index.submittedRun
    ? (index.runs.find((r) => r.run === index.submittedRun && r.eligibility === "submitted") ??
      null)
    : null;

  if (!submitted || submitted.testPrimary === null) {
    return (
      <Empty>
        No run's submission matches <code className="font-mono">submission_best.csv</code>. The
        verdict board needs a run whose <code className="font-mono">submission.csv</code> is
        byte-identical to the repo's submission file and carries hidden-test scores.
      </Empty>
    );
  }

  const runs = selectable(index);
  const scriptFailures = runs.reduce((a, r) => a + r.scriptFailures, 0);
  const infraFailures = runs.reduce((a, r) => a + r.infraFailures, 0);
  // The bar is on the hidden-test scale, so the baseline must be too. baselineTarget is the
  // VALIDATION baseline and belongs to a different scale; test always runs lower on this
  // benchmark, so plotting it here would show a winning submission as a loss.
  const baseline =
    submitted.testDelta !== null ? submitted.testPrimary - submitted.testDelta : FALLBACK_BASELINE;

  const zeroIntervention = runs.filter((r) => r.manualInterventions === 0);
  const nonZeroIntervention = runs.filter(
    (r) => r.manualInterventions !== null && r.manualInterventions !== 0,
  );

  const iterPct =
    submitted.iterations !== null && submitted.iterationCap
      ? (submitted.iterations / submitted.iterationCap) * 100
      : null;
  const wallPct = submitted.wallClockS !== null ? (submitted.wallClockS / WALLCLOCK_CAP_S) * 100 : null;

  return (
    <div className="flex flex-col gap-4">
      <Panel title={`Submitted run - ${submitted.run}`}>
        <StatGrid>
          <Stat label="Hidden test primary" value={score(submitted.testPrimary)} />
          <Stat
            label="Delta vs baseline"
            value={signed(submitted.testDelta ?? undefined)}
            tone={submitted.testDelta !== null && submitted.testDelta > 0 ? "good" : undefined}
            note="on the hidden test split"
          />
          <Stat label="Test GAUC" value={score(submitted.testGauc)} />
          <Stat label="Test nDCG@5" value={score(submitted.testNdcg5)} />
          <Stat label="Validation primary" value={score(submitted.validPrimary)} />
        </StatGrid>
        {index.submissionSha256 && (
          <div className="text-foreground flex items-center gap-2 text-xs">
            <ShieldCheck className="text-good size-4 shrink-0" aria-hidden />
            <span>
              <code className="bg-secondary rounded px-1 py-0.5 font-mono">
                submission_best.csv
              </code>{" "}
              is byte-identical to{" "}
              <code className="bg-secondary rounded px-1 py-0.5 font-mono">
                runs/{submitted.run}/submission.csv
              </code>{" "}
              - sha256{" "}
              <span className="tnum font-mono font-semibold">
                {index.submissionSha256.slice(0, 16)}
              </span>
              <span className="text-muted-foreground">…</span>
            </span>
          </div>
        )}
      </Panel>

      <Panel title="Technical execution · 35%">
        <p className="text-ink-2 text-sm">
          Everything on this bar is on the <strong className="text-foreground">hidden-test</strong>{" "}
          scale: random scoring, the official baseline, the submission, and the perfect-ranking
          ceiling.
        </p>
        <CalibrationBar baseline={baseline} submission={submitted.testPrimary} />
        <p className="text-ink-2 -mt-2 text-xs">
          Random ({score(RANDOM_PRIMARY)}) and the ceiling ({score(CEILING_PRIMARY)}) are fixed
          properties of the benchmark - 27.1% of test users have no positive label, so a perfect
          ranking still scores well under 1.0. The baseline tick is derived from this run's own
          records as test primary minus its recorded delta. Validation figures are a different
          scale and are not plotted here; test runs lower than validation on this benchmark for
          the baseline and the agent alike.
        </p>
        <Separator />
        <StatGrid>
          <Stat
            label="Script failures"
            value={int(scriptFailures)}
            tone={scriptFailures > 0 ? "bad" : "good"}
            note={`the agent's own code errored, across ${runs.length} selectable runs`}
          />
          <Stat
            label="API outages"
            value={int(infraFailures)}
            note="LLM transport error - the agent's code never ran"
          />
        </StatGrid>
      </Panel>

      <Panel title="Innovation & problem insight · 20%">
        <div className="flex flex-col gap-1.5">
          <p className="text-[15px] leading-relaxed">
            <span className="tnum font-mono font-semibold">
              {int(submitted.candidatesEvaluated)}
            </span>{" "}
            candidate solutions compared inside{" "}
            <span className="tnum font-mono font-semibold">{int(submitted.iterations)}</span>{" "}
            iterations.
          </p>
          <p className="text-[15px] leading-relaxed">
            <span className="tnum font-mono font-semibold">
              {int(submitted.claimsEstablished)}
            </span>{" "}
            beliefs established.
          </p>
        </div>
        <Button
          variant="outline"
          size="xs"
          className="w-fit"
          onClick={() => onOpenRun(submitted.run)}
        >
          read the agent's hypotheses
        </Button>
      </Panel>

      <Panel title="Autonomy · 20%">
        <div className="flex items-baseline gap-2.5">
          <span
            className={`tnum font-mono text-[32px] leading-none font-semibold ${
              submitted.manualInterventions === 0 ? "text-good-ink" : ""
            }`}
          >
            {submitted.manualInterventions ?? "--"}
          </span>
          <span className="text-ink-2 text-sm">
            manual interventions to reach the converged result
          </span>
        </div>
        <p className="text-ink-2 text-sm">
          <span className="tnum font-mono font-semibold">{zeroIntervention.length}</span> of{" "}
          <span className="tnum font-mono font-semibold">{runs.length}</span> selectable runs also
          recorded zero manual interventions.
        </p>
        {nonZeroIntervention.length > 0 && (
          <Note tone="warning">
            Not every run was hands-off:{" "}
            {nonZeroIntervention.map((r, i) => (
              <span key={r.run}>
                {i > 0 && ", "}
                <code>{r.run}</code> ({r.manualInterventions})
              </span>
            ))}
            .
          </Note>
        )}
      </Panel>

      <Panel title="Feasibility · 15%">
        <div className="flex flex-col gap-2">
          {iterPct !== null && submitted.iterationCap !== null ? (
            <BudgetBar
              label="Iterations"
              pct={iterPct}
              usedLabel={int(submitted.iterations)}
              capLabel={int(submitted.iterationCap)}
            />
          ) : (
            <KeyValue>
              <Row k="Iterations">
                {int(submitted.iterations)} <span className="text-muted-foreground">(no cap recorded)</span>
              </Row>
            </KeyValue>
          )}
          {wallPct !== null ? (
            <BudgetBar
              label="Wall clock"
              pct={wallPct}
              usedLabel={duration(submitted.wallClockS ?? undefined)}
              capLabel={duration(WALLCLOCK_CAP_S)}
            />
          ) : (
            <KeyValue>
              <Row k="Wall clock">unknown</Row>
            </KeyValue>
          )}
          <div className="mt-1 flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Tokens</span>
            <span className="tnum font-mono">
              {int(submitted.tokensTotal)}{" "}
              <span className="text-muted-foreground font-sans">uncapped</span>
            </span>
          </div>
        </div>
      </Panel>
    </div>
  );
}
