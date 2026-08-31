import { useMemo, useState } from "react";
import {
  ELIGIBILITY_FILL,
  ELIGIBILITY_LABEL,
  ELIGIBILITY_ORDER,
  type Eligibility,
  type RunIndex,
  type RunSummary,
} from "@/lib/runIndex";
import { score, signed, truncate } from "@/lib/format";
import { Panel, Note } from "@/components/common";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

function Swatch({ eligibility, className }: { eligibility: Eligibility; className?: string }) {
  return (
    <span
      className={cn("inline-block size-1.5 shrink-0 rounded-full", ELIGIBILITY_FILL[eligibility], className)}
    />
  );
}

function RunTooltip({ run }: { run: RunSummary }) {
  return (
    <TooltipContent className="max-w-[300px]">
      <div className="flex items-center gap-1.5 font-mono text-xs font-semibold">
        <Swatch eligibility={run.eligibility} />
        {run.run}
      </div>
      <div className="text-ink-2 mt-1 text-[11px] leading-relaxed">{run.reason}</div>
      <dl className="tnum mt-1.5 grid grid-cols-[max-content_minmax(0,1fr)] gap-x-2.5 gap-y-0.5 text-[11px]">
        <dt className="text-muted-foreground">dataset</dt>
        <dd className="truncate">{run.dataset ?? "--"}</dd>
        <dt className="text-muted-foreground">contract</dt>
        <dd className="truncate">{run.contract ?? "--"}</dd>
        <dt className="text-muted-foreground">iterations</dt>
        <dd>{run.iterations ?? "--"}</dd>
        <dt className="text-muted-foreground">best valid</dt>
        <dd>{score(run.bestValid)}</dd>
        <dt className="text-muted-foreground">test delta</dt>
        <dd>{signed(run.testDelta ?? undefined)}</dd>
      </dl>
      {run.eligibility === "leakage" && run.outcomeFields.length > 0 && (
        <div className="border-line-strong mt-1.5 border-t pt-1.5 text-[11px] leading-relaxed">
          <span className="text-muted-foreground">exposed: </span>
          {truncate(run.outcomeFields.join(", "), 160)}
        </div>
      )}
    </TooltipContent>
  );
}

export default function RunMap({
  index,
  activeRun,
  onOpenRun,
}: {
  index: RunIndex;
  activeRun: string;
  onOpenRun: (run: string) => void;
}) {
  const [filter, setFilter] = useState<Eligibility | "all">("all");

  const runs = useMemo(
    () => [...index.runs].sort((a, b) => a.run.localeCompare(b.run, undefined, { numeric: true })),
    [index.runs],
  );

  const counts = useMemo(() => {
    const c = new Map<Eligibility, number>();
    for (const r of runs) c.set(r.eligibility, (c.get(r.eligibility) ?? 0) + 1);
    return c;
  }, [runs]);

  const shown = filter === "all" ? runs : runs.filter((r) => r.eligibility === filter);

  const submitted = index.runs.find((r) => r.run === index.submittedRun) ?? null;
  const leakageRuns = useMemo(() => runs.filter((r) => r.eligibility === "leakage"), [runs]);
  const unverifiedCount = counts.get("unverified") ?? 0;

  return (
    <Panel title="Run provenance">
      <div className="flex flex-col gap-2.5">
        {submitted && (
          <Note>
            Submitted: <span className="font-mono">{submitted.run}</span> - submission_best.csv
            (sha256 <span className="font-mono">{index.submissionSha256?.slice(0, 16)}…</span>) is
            byte-identical to this run's submission.csv.
          </Note>
        )}
        {leakageRuns.length > 0 && (
          <Note tone="critical">
            Excluded for leakage ({leakageRuns.length}):{" "}
            {leakageRuns.map((r, i) => (
              <span key={r.run}>
                {i > 0 && ", "}
                <span className="font-mono">{r.run}</span>
              </span>
            ))}{" "}
            - each exposed full-month item outcome columns overlapping the validation/test
            windows (hover a cell for the exact fields).
          </Note>
        )}
        {unverifiedCount > 0 && (
          <Note tone="warning">
            {unverifiedCount} run{unverifiedCount > 1 ? "s are" : " is"} unverified: no
            run_meta.json, so neither contract nor API surface can be checked. They are neither
            claimed nor excluded.
          </Note>
        )}
      </div>

      <ToggleGroup
        type="single"
        value={filter}
        onValueChange={(v) => setFilter((v as Eligibility | "all") || "all")}
        className=""
      >
        <ToggleGroupItem value="all">all ({runs.length})</ToggleGroupItem>
        {ELIGIBILITY_ORDER.filter((e) => (counts.get(e) ?? 0) > 0).map((e) => (
          <ToggleGroupItem key={e} value={e}>
            <Swatch eligibility={e} />
            {ELIGIBILITY_LABEL[e]} ({counts.get(e) ?? 0})
          </ToggleGroupItem>
        ))}
      </ToggleGroup>

      <div className="grid grid-cols-[repeat(auto-fill,minmax(112px,1fr))] gap-1.5">
        {shown.map((r) => (
          <Tooltip key={r.run}>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={() => onOpenRun(r.run)}
                className={cn(
                  "bg-card hover:bg-secondary focus-visible:ring-ring min-w-0 cursor-pointer rounded-md border px-2 py-1.5 text-left outline-none transition-colors focus-visible:ring-2",
                  r.run === activeRun && "ring-primary ring-1",
                )}
              >
                <div className="flex min-w-0 items-center gap-1.5">
                  <Swatch eligibility={r.eligibility} />
                  <span className="truncate font-mono text-xs font-medium">{r.run}</span>
                </div>
                <div className="tnum text-muted-foreground mt-0.5 text-[11px]">
                  {score(r.bestValid)}
                </div>
              </button>
            </TooltipTrigger>
            <RunTooltip run={r} />
          </Tooltip>
        ))}
      </div>
    </Panel>
  );
}
