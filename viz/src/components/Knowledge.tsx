import type { RunData } from "../lib/types";
import { Card, Empty, StatusPill } from "./ui";

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
        <div className="filters">
          {[...byStatus].map(([s, n]) => (
            <span className={`pill ${s}`} key={s}>
              <span className="dot" />
              {n} {s}
            </span>
          ))}
        </div>
        {data.beliefs.map((b, i) => (
          <div className={`belief ${b.status ?? "active"}`} key={i}>
            <div className="chips">
              <StatusPill status={b.status ?? "active"} />
              {b.evidence?.length ? <span className="label">from</span> : null}
              {b.evidence?.map((e) => (
                <button type="button" className="chip" key={e} onClick={() => onSelectIteration(e)}>
                  #{e}
                </button>
              ))}
            </div>
            <p>{b.text}</p>
          </div>
        ))}
      </>
    );
  }

  // r27-r30 predate knowledge.json and kept free-text reflections instead.
  if (data.reflections) {
    return (
      <Card title="reflections.md">
        <p className="secondary" style={{ marginTop: 0 }}>
          This run predates the structured belief set; it recorded free-text reflections, one block
          per iteration.
        </p>
        <pre className="text" style={{ whiteSpace: "pre-wrap" }}>
          {data.reflections.trim()}
        </pre>
      </Card>
    );
  }

  return <Empty>This run recorded no knowledge.json and no reflections.md.</Empty>;
}
