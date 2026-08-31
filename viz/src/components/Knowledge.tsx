import type { RunData } from "../lib/types";
import { Panel, Empty, Pre, StatusBadge, IterLink } from "@/components/common";
import { cn } from "@/lib/utils";

export default function Knowledge({
  data,
  onSelectIteration,
}: {
  data: RunData;
  onSelectIteration: (id: number) => void;
}) {
  if (data.beliefs.length) {
    const byStatus = new Map<string, number>();
    for (const b of data.beliefs) byStatus.set(b.status ?? "active", (byStatus.get(b.status ?? "active") ?? 0) + 1);

    return (
      <>
        <div className="mb-4 flex flex-wrap items-center gap-2">
          {[...byStatus].map(([s, n]) => (
            <span key={s} className="flex items-center gap-1.5 text-sm">
              <StatusBadge status={s} />
              <span className="text-muted-foreground">{n}</span>
            </span>
          ))}
        </div>
        <div className="flex flex-col gap-2">
          {data.beliefs.map((b, i) => {
            const retired = (b.status ?? "active") === "retired";
            return (
              <div key={i} className="bg-card rounded-xl border px-3.5 py-3">
                <div className="mb-1.5 flex flex-wrap items-center gap-2">
                  <StatusBadge status={b.status ?? "active"} />
                  {b.evidence?.length ? <span className="text-muted-foreground text-xs">from</span> : null}
                  {b.evidence?.map((e) => (
                    <IterLink key={e} id={e} onClick={onSelectIteration} />
                  ))}
                </div>
                <p className={cn("max-w-[90ch] text-sm leading-relaxed", retired && "text-muted-foreground")}>
                  {b.text}
                </p>
              </div>
            );
          })}
        </div>
      </>
    );
  }

  // r27-r30 predate knowledge.json and kept free-text reflections instead.
  if (data.reflections) {
    return (
      <Panel title="reflections.md">
        <p className="text-ink-2 mt-0 mb-3 text-sm">
          This run predates the structured belief set; it recorded free-text reflections, one block
          per iteration.
        </p>
        <Pre wrap>{data.reflections.trim()}</Pre>
      </Panel>
    );
  }

  return <Empty>This run recorded no knowledge.json and no reflections.md.</Empty>;
}
