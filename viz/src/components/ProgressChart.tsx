import { useState } from "react";
import type { Iteration, RunData } from "../lib/types";
import { primaryOf } from "../lib/derive";
import { score, truncate } from "../lib/format";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

/* Geometry in viewBox units; CSS scales the svg to the card width. */
const W = 760;
const H = 320;
const PAD = { top: 16, right: 118, bottom: 52, left: 54 };
const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;
/** Iterations with no score (crashed scripts) sit on their own band under the axis. */
const FAIL_Y = H - PAD.bottom + 22;

type Point = { it: Iteration; x: number; y: number | null; value: number | undefined };

/** The five states a marker can be in. "infra" is a failed row whose script never ran. */
type MarkState = "kept" | "ok" | "reverted" | "failed" | "infra";

const stateOf = (it: Iteration): MarkState => {
  if (it.infra) return "infra";
  const s = it.status;
  return s === "kept" || s === "reverted" || s === "failed" ? s : "ok";
};

const STATE_LABEL: Record<MarkState, string> = {
  kept: "kept",
  ok: "ok",
  reverted: "reverted",
  failed: "failed",
  infra: "api outage (never ran)",
};

const STATE_COLOR: Record<MarkState, string> = {
  kept: "var(--good)",
  ok: "var(--accent-blue)",
  reverted: "var(--serious)",
  failed: "var(--critical)",
  infra: "var(--warning)",
};

/** State also carries a shape, so identity never rests on hue alone. */
function Marker({ state, cx, cy, selected }: { state: MarkState; cx: number; cy: number; selected: boolean }) {
  const color = STATE_COLOR[state];
  return (
    <g>
      {selected && <circle cx={cx} cy={cy} r={9} fill="none" stroke="var(--primary)" strokeWidth={2} />}
      {state === "failed" && (
        <g stroke={color} strokeWidth={2.4} fill="none">
          <line x1={cx - 5} y1={cy - 5} x2={cx + 5} y2={cy + 5} />
          <line x1={cx - 5} y1={cy + 5} x2={cx + 5} y2={cy - 5} />
        </g>
      )}
      {state === "infra" && (
        <path
          d={`M${cx},${cy - 6} L${cx + 6},${cy} L${cx},${cy + 6} L${cx - 6},${cy} Z`}
          fill="var(--surface-1)"
          stroke={color}
          strokeWidth={2.4}
        />
      )}
      {state === "reverted" && <circle cx={cx} cy={cy} r={5} fill="var(--surface-1)" stroke={color} strokeWidth={2.4} />}
      {(state === "kept" || state === "ok") && <circle cx={cx} cy={cy} r={5} fill={color} />}
    </g>
  );
}

export default function ProgressChart({
  data,
  selected,
  onSelect,
}: {
  data: RunData;
  selected: number | null;
  onSelect: (id: number) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const [showTable, setShowTable] = useState(false);

  const iters = data.iterations;
  if (!iters.length) return null;

  const target = data.meta?.baseline_target;
  const reproduced = data.meta?.baseline_reproduced;

  const scored = iters.map(primaryOf).filter((v): v is number => v !== undefined);
  if (!scored.length) {
    return (
      <p className="text-muted-foreground text-sm">
        No iteration in this run recorded a validation score, so there is nothing to plot.
      </p>
    );
  }

  const refs = [target, reproduced].filter((v): v is number => v !== undefined);
  const lo = Math.min(...scored, ...refs);
  const hi = Math.max(...scored, ...refs);
  const padY = Math.max((hi - lo) * 0.18, 0.0008);
  const yMin = lo - padY;
  const yMax = hi + padY;

  const xOf = (i: number) => PAD.left + (iters.length === 1 ? PLOT_W / 2 : (i * PLOT_W) / (iters.length - 1));
  const yOf = (v: number) => PAD.top + PLOT_H - ((v - yMin) / (yMax - yMin)) * PLOT_H;

  const points: Point[] = iters.map((it, i) => {
    const value = primaryOf(it);
    return { it, x: xOf(i), y: value === undefined ? null : yOf(value), value };
  });

  const path = points
    .filter((p): p is Point & { y: number } => p.y !== null)
    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
    .join(" ");

  // Five ticks is enough for a band this narrow; four decimals is the project's unit of change.
  const ticks = Array.from({ length: 5 }, (_, i) => yMin + ((yMax - yMin) * i) / 4);

  const hovered = hover === null ? null : points[hover];
  const states = [...new Set(iters.map(stateOf))];

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <h3 className="text-sm font-semibold">Validation primary by iteration</h3>
        <Button variant="outline" size="xs" className="ml-auto" onClick={() => setShowTable((s) => !s)}>
          {showTable ? "Hide table" : "Show table"}
        </Button>
      </div>

      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Validation primary score by iteration">
          <g>
            {ticks.map((t) => (
              <line key={t} className="stroke-line-strong" x1={PAD.left} x2={PAD.left + PLOT_W} y1={yOf(t)} y2={yOf(t)} />
            ))}
          </g>

          {ticks.map((t) => (
            <text key={t} className="fill-muted-foreground font-mono text-[11px]" x={PAD.left - 8} y={yOf(t) + 3.5} textAnchor="end">
              {t.toFixed(4)}
            </text>
          ))}

          {/* Reference lines: the run's own targets, never a hardcoded number. */}
          {target !== undefined && (
            <g>
              <line
                className="stroke-axis [stroke-dasharray:5_4]"
                x1={PAD.left}
                x2={PAD.left + PLOT_W}
                y1={yOf(target)}
                y2={yOf(target)}
              />
              <text className="fill-muted-foreground font-mono text-[11px]" x={PAD.left + PLOT_W + 8} y={yOf(target) + 3.5}>
                baseline {target.toFixed(4)}
              </text>
            </g>
          )}
          {reproduced !== undefined && (
            <g>
              <line
                className="stroke-chart-3 [stroke-dasharray:5_4]"
                x1={PAD.left}
                x2={PAD.left + PLOT_W}
                y1={yOf(reproduced)}
                y2={yOf(reproduced)}
              />
              <text className="fill-chart-3 font-mono text-[11px] font-medium" x={PAD.left + PLOT_W + 8} y={yOf(reproduced) + 3.5}>
                reproduced {reproduced.toFixed(4)}
              </text>
            </g>
          )}

          <g>
            <line className="stroke-axis" x1={PAD.left} x2={PAD.left + PLOT_W} y1={PAD.top + PLOT_H} y2={PAD.top + PLOT_H} />
          </g>

          <path className="fill-none stroke-chart-1 stroke-2 [stroke-linejoin:round]" d={path} />

          {hovered && (
            <line
              className="stroke-axis opacity-70 [stroke-dasharray:3_3]"
              x1={hovered.x}
              x2={hovered.x}
              y1={PAD.top}
              y2={PAD.top + PLOT_H}
            />
          )}

          {points.map((p) => (
            <g key={p.it.iter_id}>
              <Marker state={stateOf(p.it)} cx={p.x} cy={p.y ?? FAIL_Y} selected={selected === p.it.iter_id} />
              <text className="fill-muted-foreground font-mono text-[11px]" x={p.x} y={PAD.top + PLOT_H + 16} textAnchor="middle">
                #{p.it.iter_id}
              </text>
            </g>
          ))}

          {points.some((p) => p.y === null) && (
            <text className="fill-muted-foreground font-mono text-[11px]" x={PAD.left - 8} y={FAIL_Y + 3.5} textAnchor="end">
              no score
            </text>
          )}

          {/* Hit targets are wider than the marks so hovering is forgiving. */}
          {points.map((p, i) => (
            <rect
              key={p.it.iter_id}
              className="cursor-pointer fill-transparent"
              x={p.x - PLOT_W / (2 * Math.max(iters.length - 1, 1))}
              y={PAD.top}
              width={PLOT_W / Math.max(iters.length - 1, 1)}
              height={PLOT_H + 40}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelect(p.it.iter_id)}
            />
          ))}
        </svg>

        {hovered && (
          <div
            className="bg-popover text-popover-foreground pointer-events-none absolute z-50 min-w-[170px] max-w-[290px] rounded-lg border p-2.5 text-xs shadow-md"
            style={{
              left: `${((hovered.x + 12) / W) * 100}%`,
              top: `${((hovered.y ?? FAIL_Y) / H) * 100}%`,
              transform: hovered.x > W * 0.6 ? "translate(-108%, -50%)" : "translate(0, -50%)",
            }}
          >
            <div className="mb-1.5 flex items-center justify-between gap-2">
              <span className="font-mono">#{hovered.it.iter_id}</span>
              <span className="text-muted-foreground">{STATE_LABEL[stateOf(hovered.it)]}</span>
            </div>
            <dl className="tnum grid grid-cols-[max-content_1fr] gap-x-2 gap-y-0.5">
              <dt className="text-muted-foreground">primary</dt>
              <dd className="text-right font-mono">{score(hovered.value)}</dd>
              <dt className="text-muted-foreground">GAUC</dt>
              <dd className="text-right font-mono">{score(hovered.it.metrics?.gauc)}</dd>
              <dt className="text-muted-foreground">nDCG@5</dt>
              <dd className="text-right font-mono">{score(hovered.it.metrics?.["ndcg@5"])}</dd>
            </dl>
            {hovered.it.hypothesis && (
              <p className="text-ink-2 mt-1.5 leading-relaxed">{truncate(hovered.it.hypothesis, 130)}</p>
            )}
          </div>
        )}
      </div>

      <div className="text-ink-2 mt-3 flex flex-wrap gap-4 text-xs">
        {states.map((s) => (
          <span className="inline-flex items-center gap-1.5" key={s}>
            <svg width={14} height={14} viewBox="0 0 14 14">
              <Marker state={s} cx={7} cy={7} selected={false} />
            </svg>
            {STATE_LABEL[s]}
          </span>
        ))}
        <span className="text-muted-foreground inline-flex items-center gap-1.5">click a point to open that iteration</span>
      </div>

      {showTable && (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>iter</TableHead>
              <TableHead>status</TableHead>
              <TableHead className="text-right">primary</TableHead>
              <TableHead className="text-right">GAUC</TableHead>
              <TableHead className="text-right">nDCG@5</TableHead>
              <TableHead className="text-right">script s</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {iters.map((it) => (
              <TableRow key={it.iter_id}>
                <TableCell className="font-mono">#{it.iter_id}</TableCell>
                <TableCell>{STATE_LABEL[stateOf(it)]}</TableCell>
                <TableCell className="text-right font-mono">{score(primaryOf(it))}</TableCell>
                <TableCell className="text-right font-mono">{score(it.metrics?.gauc)}</TableCell>
                <TableCell className="text-right font-mono">{score(it.metrics?.["ndcg@5"])}</TableCell>
                <TableCell className="text-right font-mono">
                  {it.gpu_seconds !== undefined ? it.gpu_seconds.toFixed(1) : "--"}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
