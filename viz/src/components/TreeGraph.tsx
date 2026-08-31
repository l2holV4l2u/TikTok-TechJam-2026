import { useMemo, useState } from "react";
import type { Iteration, RunData } from "../lib/types";
import { primaryOf } from "../lib/derive";
import { score, truncate } from "../lib/format";
import { Panel, Empty } from "@/components/common";

/** Past this many nodes the layout stops being legible; the indented list is the fallback view. */
const MAX_NODES = 40;

const DX = 132; // px per search generation (depth)
/** right pad leaves room for the last column's labels, which sit outside the node. */
const PAD = { top: 24, right: 96, bottom: 24, left: 28 };
const R_MIN = 5;
const R_MAX = 12;

type StatusKey = "kept" | "ok" | "reverted" | "failed" | "infra" | "unknown";

/** Matches the project-wide status convention; anything else (blacklisted, rejected, ...)
 * is honestly "unknown" rather than guessed at. */
const STATUS_COLOR: Record<StatusKey, string> = {
  kept: "var(--good)",
  ok: "var(--accent-blue)",
  reverted: "var(--serious)",
  failed: "var(--critical)",
  infra: "var(--warning)",
  unknown: "var(--axis)",
};

const STATUS_LABEL: Record<StatusKey, string> = {
  kept: "kept",
  ok: "ok",
  reverted: "reverted",
  failed: "failed",
  infra: "api outage (never ran)",
  unknown: "unknown",
};

const STATUS_ORDER: StatusKey[] = ["kept", "ok", "reverted", "failed", "infra", "unknown"];

function statusKeyOf(it: Iteration): StatusKey {
  if (it.infra) return "infra";
  const s = (it.status ?? "").toLowerCase();
  return s === "kept" || s === "ok" || s === "reverted" || s === "failed" ? s : "unknown";
}

interface LayoutNode {
  it: Iteration;
  x: number;
  y: number;
}

interface Layout {
  nodes: LayoutNode[];
  edges: { x1: number; y1: number; x2: number; y2: number; key: number }[];
  width: number;
  height: number;
}

/**
 * Tidy-tree pass: depth -> x (generation), post-order slot assignment -> y (sibling order).
 * A leaf takes the next free slot; a parent sits at the mean of its children's slots. Because
 * children are visited in ledger order this never crosses an edge -- no layout library needed.
 */
function layoutTree(data: RunData): Layout | null {
  const { iterations, byId, roots } = data;
  if (!iterations.length) return null;

  let slot = 0;
  const slotOf = new Map<number, number>();
  const seen = new Set<number>();

  const visit = (id: number): number => {
    if (seen.has(id)) return slotOf.get(id) ?? 0;
    seen.add(id);
    const it = byId.get(id);
    if (!it) return 0;
    const kids = it.children.filter((c) => byId.has(c));
    const s = kids.length === 0 ? slot++ : kids.map(visit).reduce((a, b) => a + b, 0) / kids.length;
    slotOf.set(id, s);
    return s;
  };

  for (const r of roots) visit(r);
  // Defensive: a row whose parent link is broken in a way derive.ts didn't already root.
  for (const it of iterations) if (!seen.has(it.iter_id)) visit(it.iter_id);
  if (slot === 0) return null;

  // Compress row spacing once the tree gets tall so a flat/wide run doesn't blow past the page.
  const dy = slot > 20 ? 34 : 46; // two label lines need ~24px of vertical room
  const maxDepth = Math.max(...iterations.map((it) => it.depth));

  const nodes: LayoutNode[] = iterations
    .filter((it) => slotOf.has(it.iter_id))
    .map((it) => ({
      it,
      x: PAD.left + it.depth * DX,
      y: PAD.top + (slotOf.get(it.iter_id) ?? 0) * dy,
    }));

  const byIterId = new Map(nodes.map((n) => [n.it.iter_id, n]));
  const edges: Layout["edges"] = [];
  for (const n of nodes) {
    const pid = n.it.parent_iter_id;
    const parent = pid === null ? undefined : byIterId.get(pid);
    if (parent) edges.push({ x1: parent.x, y1: parent.y, x2: n.x, y2: n.y, key: n.it.iter_id });
  }

  return {
    nodes,
    edges,
    width: PAD.left + PAD.right + maxDepth * DX + 24,
    height: PAD.top + PAD.bottom + (slot - 1) * dy,
  };
}

/** Same shape vocabulary as ProgressChart: status never rests on hue alone. */
function NodeMark({ status, cx, cy, r, color }: { status: StatusKey; cx: number; cy: number; r: number; color: string }) {
  if (status === "failed") {
    return (
      <g stroke={color} strokeWidth={2.2} fill="none">
        <line x1={cx - r} y1={cy - r} x2={cx + r} y2={cy + r} />
        <line x1={cx - r} y1={cy + r} x2={cx + r} y2={cy - r} />
      </g>
    );
  }
  if (status === "infra") {
    return (
      <path
        d={`M${cx},${cy - r} L${cx + r},${cy} L${cx},${cy + r} L${cx - r},${cy} Z`}
        fill="var(--surface-1)"
        stroke={color}
        strokeWidth={2.2}
      />
    );
  }
  if (status === "reverted") {
    return <circle cx={cx} cy={cy} r={r} fill="var(--surface-1)" stroke={color} strokeWidth={2.2} />;
  }
  return <circle cx={cx} cy={cy} r={r} fill={color} />;
}

export default function TreeGraph({
  data,
  selected,
  onSelect,
}: {
  data: RunData;
  selected: number | null;
  onSelect: (id: number) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const layout = useMemo(() => layoutTree(data), [data]);

  if (!data.iterations.length) {
    return (
      <Panel title="Search tree">
        <Empty>This run has no ledger rows.</Empty>
      </Panel>
    );
  }
  if (data.iterations.length > MAX_NODES || !layout) {
    return (
      <Panel title="Search tree">
        <Empty>
          {data.iterations.length} iterations is too many to lay out legibly here. Use the list
          view for the full search tree.
        </Empty>
      </Panel>
    );
  }

  const { nodes, edges, width, height } = layout;
  const scored = nodes.map((n) => primaryOf(n.it)).filter((v): v is number => v !== undefined);
  const sMin = scored.length ? Math.min(...scored) : 0;
  const sMax = scored.length ? Math.max(...scored) : 0;
  const radiusOf = (it: Iteration) => {
    const p = primaryOf(it);
    if (p === undefined) return R_MIN;
    if (sMax === sMin) return (R_MIN + R_MAX) / 2;
    return R_MIN + ((R_MAX - R_MIN) * (p - sMin)) / (sMax - sMin);
  };

  const hovered = hover === null ? undefined : nodes.find((n) => n.it.iter_id === hover);
  const statesPresent = STATUS_ORDER.filter((k) => nodes.some((n) => statusKeyOf(n.it) === k));

  return (
    <Panel title="Search tree">
      <div className="flex flex-col gap-3">
        <div className="overflow-x-auto">
          <div className="relative" style={{ width }}>
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="h-auto w-full"
              role="img"
              aria-label="Search tree: iterations laid out left to right by search generation"
            >
              <g className="stroke-line-strong" fill="none" strokeWidth={1.5}>
                {edges.map((e) => {
                  const midX = (e.x1 + e.x2) / 2;
                  return (
                    <path key={e.key} d={`M${e.x1},${e.y1} C${midX},${e.y1} ${midX},${e.y2} ${e.x2},${e.y2}`} />
                  );
                })}
              </g>

              {nodes.map((n) => {
                const key = statusKeyOf(n.it);
                const color = STATUS_COLOR[key];
                const r = radiusOf(n.it);
                const isSelected = selected === n.it.iter_id;
                return (
                  <g
                    key={n.it.iter_id}
                    role="button"
                    tabIndex={0}
                    aria-pressed={isSelected}
                    aria-label={`iteration ${n.it.iter_id}, ${STATUS_LABEL[key]}, primary ${score(primaryOf(n.it))}`}
                    className="cursor-pointer outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                    onClick={() => onSelect(n.it.iter_id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onSelect(n.it.iter_id);
                      }
                    }}
                    onMouseEnter={() => setHover(n.it.iter_id)}
                    onMouseLeave={() => setHover((h) => (h === n.it.iter_id ? null : h))}
                    onFocus={() => setHover(n.it.iter_id)}
                    onBlur={() => setHover((h) => (h === n.it.iter_id ? null : h))}
                  >
                    <circle cx={n.x} cy={n.y} r={R_MAX + 7} fill="transparent" />
                    {isSelected && (
                      <circle cx={n.x} cy={n.y} r={r + 5} fill="none" className="stroke-primary" strokeWidth={2} />
                    )}
                    <NodeMark status={key} cx={n.x} cy={n.y} r={r} color={color} />
                    {/* Labels are always on: the diagram has to be readable without a pointer. */}
                    <text
                      x={n.x + R_MAX + 7}
                      y={n.y - 1}
                      className={`fill-foreground font-mono text-[11px] ${isSelected ? "font-bold" : "font-semibold"}`}
                    >
                      #{n.it.iter_id}
                    </text>
                    <text
                      x={n.x + R_MAX + 7}
                      y={n.y + 10}
                      className="fill-muted-foreground font-mono text-[10px]"
                    >
                      {score(primaryOf(n.it))}
                    </text>
                  </g>
                );
              })}
            </svg>

            {hovered && (
              <div
                className="bg-popover text-popover-foreground pointer-events-none absolute z-50 min-w-[170px] max-w-[300px] rounded-lg border p-2.5 text-xs shadow-md"
                style={{
                  left: `${(hovered.x / width) * 100}%`,
                  top: `${(hovered.y / height) * 100}%`,
                  transform: hovered.x > width * 0.6 ? "translate(-108%, -50%)" : "translate(14px, -50%)",
                }}
              >
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <span className="font-mono">#{hovered.it.iter_id}</span>
                  <span className="text-muted-foreground">{STATUS_LABEL[statusKeyOf(hovered.it)]}</span>
                </div>
                <dl className="tnum grid grid-cols-[max-content_1fr] gap-x-2 gap-y-0.5">
                  <dt className="text-muted-foreground">phase</dt>
                  <dd className="min-w-0 truncate text-right">{hovered.it.phase ?? "--"}</dd>
                  <dt className="text-muted-foreground">primary</dt>
                  <dd className="text-right font-mono">{score(primaryOf(hovered.it))}</dd>
                </dl>
                {hovered.it.hypothesis && (
                  <p className="text-ink-2 mt-1.5 leading-relaxed">{truncate(hovered.it.hypothesis, 160)}</p>
                )}
              </div>
            )}
          </div>
        </div>

        <p className="text-ink-2 text-xs leading-relaxed">
          Left to right = search generation; an edge means the child script was proposed from its
          parent. Node size scales with validation primary within this run (bigger is better);
          nodes with no score (failed/api-outage rows) are drawn at a fixed minimum size.
        </p>

        <div className="text-ink-2 flex flex-wrap gap-4 text-xs">
          {statesPresent.map((k) => (
            <span className="inline-flex items-center gap-1.5" key={k}>
              <svg width={14} height={14} viewBox="0 0 14 14">
                <NodeMark status={k} cx={7} cy={7} r={5} color={STATUS_COLOR[k]} />
              </svg>
              {STATUS_LABEL[k]}
            </span>
          ))}
          <span className="text-muted-foreground inline-flex items-center gap-1.5">
            click, or focus and press Enter/Space, to open an iteration
          </span>
        </div>
      </div>
    </Panel>
  );
}
