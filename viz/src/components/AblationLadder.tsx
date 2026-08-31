import { useState } from "react";
import { selectable, type RunIndex, type RunSummary } from "@/lib/runIndex";
import { int, score, signed } from "@/lib/format";
import { Panel } from "@/components/common";
import { cn } from "@/lib/utils";

/** The official baseline's own 5-seed standard deviation. Two runs inside one band width
 *  are not distinguishable, so the band is drawn rather than left to the reader. */
const NOISE = 0.0008;

interface Scale {
  min: number;
  max: number;
  /** Fraction across the track, clamped, for a value. */
  pct: (v: number) => number;
}

function makeScale(min: number, max: number): Scale {
  const span = max - min || 1;
  return { min, max, pct: (v) => Math.max(0, Math.min(100, ((v - min) / span) * 100)) };
}

/** One bar in its own track, with the axis furniture drawn behind it. */
function Bar({
  value,
  scale,
  band,
  refLine,
  emphasis,
}: {
  value: number | null;
  scale: Scale;
  band?: { from: number; to: number };
  refLine?: number | null;
  emphasis: boolean;
}) {
  if (value === null) {
    return <div className="text-muted-foreground h-5 text-[11px] leading-5">no value</div>;
  }
  const zero = scale.pct(Math.max(scale.min, 0));
  const v = scale.pct(value);
  const left = Math.min(zero, v);
  const width = Math.max(Math.abs(v - zero), 0.6);

  return (
    <div className="bg-secondary relative h-5 overflow-hidden rounded">
      {band && (
        <div
          className="absolute inset-y-0"
          style={{
            left: `${scale.pct(band.from)}%`,
            width: `${scale.pct(band.to) - scale.pct(band.from)}%`,
            background: "color-mix(in srgb, var(--axis) 22%, transparent)",
          }}
        />
      )}
      {refLine !== null && refLine !== undefined && (
        <div
          className="border-axis absolute inset-y-0 border-l border-dashed"
          style={{ left: `${scale.pct(refLine)}%` }}
        />
      )}
      <div
        className={cn("absolute inset-y-0.5 rounded-sm", emphasis ? "bg-good" : "bg-primary")}
        style={{ left: `${left}%`, width: `${width}%` }}
      />
    </div>
  );
}

export default function AblationLadder({
  index,
  onOpenRun,
}: {
  index: RunIndex;
  onOpenRun: (run: string) => void;
}) {
  const [hover, setHover] = useState<string | null>(null);

  const comparable = selectable(index).sort((a, b) =>
    a.run.localeCompare(b.run, undefined, { numeric: true }),
  );
  const excluded = index.runs.length - comparable.length;

  if (comparable.length === 0) {
    return (
      <Panel title="Cross-run comparison">
        <p className="text-muted-foreground text-sm">No run is comparable to another yet.</p>
      </Panel>
    );
  }

  const valids = comparable.map((r) => r.bestValid).filter((v): v is number => v !== null);
  const deltas = comparable.map((r) => r.testDelta).filter((v): v is number => v !== null);
  const baseline = comparable.find((r) => r.baselineTarget !== null)?.baselineTarget ?? null;

  // Validation floor sits at or below the official baseline, so "every run clears it" is visible
  // rather than asserted. It is not zero, and the axis says so.
  const vLo = Math.min(...valids, baseline ?? Infinity) - NOISE;
  const vHi = Math.max(...valids) + NOISE;
  const vScale = makeScale(vLo, vHi);
  const vMean = valids.reduce((a, b) => a + b, 0) / valids.length;

  // Delta is anchored at zero: zero means "exactly matched the official baseline", which is a
  // real origin, so truncating it would overstate the differences.
  const dScale = makeScale(Math.min(0, ...deltas), Math.max(0, ...deltas) + NOISE / 2);

  return (
    <Panel title="Cross-run comparison">
      <p className="text-ink-2 text-sm">
        {comparable.length} comparable run{comparable.length === 1 ? "" : "s"}, selected on
        validation only. {excluded} of {index.runs.length} are excluded as not comparable -
        superseded contract, leakage, bonus dataset or unverified.
      </p>
      <p className="text-muted-foreground -mt-2 text-xs">
        Validation starts at {score(vLo)}, not zero, so differences in the fourth decimal stay
        visible; the dashed line is the official baseline. The shaded band is the baseline's own
        seed noise (±{NOISE}) - runs inside one band width are not distinguishable.
      </p>

      <div className="grid grid-cols-[max-content_minmax(90px,1fr)_max-content_minmax(90px,1fr)_max-content] items-center gap-x-4 gap-y-2">
        <div />
        <div className="text-muted-foreground text-[11px] font-semibold tracking-[0.04em] uppercase">
          validation best
        </div>
        <div />
        <div className="text-muted-foreground text-[11px] font-semibold tracking-[0.04em] uppercase">
          hidden-test delta
        </div>
        <div />

        {comparable.map((r) => {
          const isSubmitted = r.eligibility === "submitted";
          const active = hover === r.run;
          return (
            <Row
              key={r.run}
              run={r}
              active={active}
              isSubmitted={isSubmitted}
              vScale={vScale}
              dScale={dScale}
              baseline={baseline}
              band={{ from: vMean - NOISE, to: vMean + NOISE }}
              onHover={setHover}
              onOpenRun={onOpenRun}
            />
          );
        })}
      </div>

      <div className="text-ink-2 flex flex-wrap items-center gap-x-5 gap-y-1.5 text-xs">
        <span className="inline-flex items-center gap-1.5">
          <span className="bg-good size-2.5 rounded-sm" /> submitted
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="bg-primary size-2.5 rounded-sm" /> eligible
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="h-2.5 w-4 rounded-sm"
            style={{ background: "color-mix(in srgb, var(--axis) 22%, transparent)" }}
          />
          seed noise ±{NOISE}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="border-axis h-2.5 border-l border-dashed" /> official baseline
          {baseline !== null ? ` ${score(baseline)}` : ""}
        </span>
      </div>
    </Panel>
  );
}

function Row({
  run,
  active,
  isSubmitted,
  vScale,
  dScale,
  baseline,
  band,
  onHover,
  onOpenRun,
}: {
  run: RunSummary;
  active: boolean;
  isSubmitted: boolean;
  vScale: Scale;
  dScale: Scale;
  baseline: number | null;
  band: { from: number; to: number };
  onHover: (run: string | null) => void;
  onOpenRun: (run: string) => void;
}) {
  const cell = cn("contents");
  const tip =
    `${run.run} · ${run.iterations ?? "?"} iterations · ` +
    `${run.tokensTotal !== null ? int(run.tokensTotal) : "?"} tokens · ` +
    `${run.manualInterventions ?? "?"} interventions`;

  return (
    <div
      className={cell}
      onMouseEnter={() => onHover(run.run)}
      onMouseLeave={() => onHover(null)}
    >
      <button
        type="button"
        onClick={() => onOpenRun(run.run)}
        title={tip}
        className={cn(
          "focus-visible:ring-ring cursor-pointer rounded px-1 py-0.5 text-left font-mono text-xs outline-none focus-visible:ring-2",
          active && "bg-secondary",
          isSubmitted ? "text-good-ink font-semibold" : "text-ink-2",
        )}
      >
        {run.run}
      </button>

      <Bar
        value={run.bestValid}
        scale={vScale}
        band={band}
        refLine={baseline}
        emphasis={isSubmitted}
      />
      <span className="tnum w-14 text-right font-mono text-xs">{score(run.bestValid)}</span>

      <Bar value={run.testDelta} scale={dScale} emphasis={isSubmitted} />
      <span
        className={cn(
          "tnum w-16 text-right font-mono text-xs",
          run.testDelta !== null && run.testDelta > 0 && "text-good-ink",
          run.testDelta !== null && run.testDelta < 0 && "text-critical",
        )}
      >
        {run.testDelta === null ? "--" : signed(run.testDelta)}
      </span>
    </div>
  );
}
