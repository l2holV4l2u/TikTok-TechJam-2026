import type { Iteration, RunData } from "../lib/types";
import { primaryOf } from "../lib/derive";
import { score, truncate } from "../lib/format";
import { Panel, Empty, Pre } from "@/components/common";
import { cn } from "@/lib/utils";

/** Depth-first, so children sit directly under the parent they were proposed from. */
function flatten(data: RunData): Iteration[] {
  const out: Iteration[] = [];
  const seen = new Set<number>();
  const visit = (id: number) => {
    if (seen.has(id)) return;
    seen.add(id);
    const it = data.byId.get(id);
    if (!it) return;
    out.push(it);
    for (const child of it.children) visit(child);
  };
  for (const r of data.roots) visit(r);
  // Anything orphaned by a broken parent link still deserves a row.
  for (const it of data.iterations) if (!seen.has(it.iter_id)) out.push(it);
  return out;
}

/** Left status stripe colour, keyed by infra first and then iteration status. */
const STRIPE: Record<string, string> = {
  kept: "border-l-good",
  ok: "border-l-primary",
  reverted: "border-l-serious",
  failed: "border-l-critical",
  infra: "border-l-warning",
  unknown: "border-l-axis",
};

export default function SearchTree({
  data,
  selected,
  onSelect,
}: {
  data: RunData;
  selected: number | null;
  onSelect: (id: number) => void;
}) {
  const nodes = flatten(data);

  return (
    <div>
      <Panel title={`Iterations (${nodes.length})`}>
        {nodes.length === 0 ? (
          <Empty>This run has no ledger rows.</Empty>
        ) : (
          <div className="flex max-h-[calc(100vh-260px)] min-w-0 flex-col gap-1.5 overflow-y-auto">
            {nodes.map((it) => {
              const key = it.infra ? "infra" : (it.status ?? "ok");
              const stripe = STRIPE[key] ?? STRIPE.unknown;
              const isSelected = selected === it.iter_id;
              return (
                <button
                  key={it.iter_id}
                  type="button"
                  aria-current={isSelected}
                  style={{ marginLeft: Math.min(it.depth, 6) * 14 }}
                  onClick={() => onSelect(it.iter_id)}
                  className={cn(
                    "bg-card w-full min-w-0 cursor-pointer rounded-lg border border-l-[3px] px-2.5 py-2 text-left transition-colors hover:bg-secondary",
                    stripe,
                    isSelected && "ring-primary bg-secondary ring-1",
                  )}
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="text-ink-2 font-mono text-xs font-semibold">#{it.iter_id}</span>
                    <span className="text-muted-foreground min-w-0 truncate text-xs">{it.phase}</span>
                    <span className="tnum ml-auto shrink-0 font-mono text-[13px] font-semibold">
                      {score(primaryOf(it))}
                    </span>
                  </div>
                  <div className="text-ink-2 mt-0.5 line-clamp-2 text-xs">
                    {it.status === "failed" && it.error
                      ? truncate(it.error, 120)
                      : truncate(it.hypothesis ?? "no hypothesis recorded", 140)}
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </Panel>

      {data.searchTree && (
        <Panel title="search_tree.txt" className="mt-4">
          <Pre>{data.searchTree.trimEnd()}</Pre>
        </Panel>
      )}
    </div>
  );
}
