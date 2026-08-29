import type { Iteration, RunData } from "../lib/types";
import { primaryOf } from "../lib/derive";
import { score, truncate } from "../lib/format";
import { Card, Empty } from "./ui";

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
      <Card title={`Iterations (${nodes.length})`}>
        {nodes.length === 0 ? (
          <Empty>This run has no ledger rows.</Empty>
        ) : (
          <div className="tree">
            {nodes.map((it) => (
              <button
                key={it.iter_id}
                type="button"
                className={`tree-node ${it.infra ? "infra" : (it.status ?? "ok")}`}
                aria-current={selected === it.iter_id}
                style={{ marginLeft: Math.min(it.depth, 6) * 14 }}
                onClick={() => onSelect(it.iter_id)}
              >
                <span className="top">
                  <span className="id">#{it.iter_id}</span>
                  <span className="muted phase" style={{ fontSize: 11.5 }}>
                    {it.phase}
                  </span>
                  <span className="score">{score(primaryOf(it))}</span>
                </span>
                <span className="hyp">
                  {it.status === "failed" && it.error
                    ? truncate(it.error, 120)
                    : truncate(it.hypothesis ?? "no hypothesis recorded", 140)}
                </span>
              </button>
            ))}
          </div>
        )}
      </Card>

      {data.searchTree && (
        <Card title="search_tree.txt">
          <pre className="text">{data.searchTree.trimEnd()}</pre>
        </Card>
      )}
    </div>
  );
}
