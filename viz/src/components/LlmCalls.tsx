import { useState } from "react";
import type { LlmCall, RunData } from "../lib/types";
import { int, timestamp, truncate } from "../lib/format";
import { Card, Empty, Tile } from "./ui";

const ROLES: { id: LlmCall["role"] | "all"; label: string }[] = [
  { id: "all", label: "all" },
  { id: "proposer", label: "proposer" },
  { id: "knowledge", label: "knowledge" },
  { id: "reflection", label: "reflection" },
  { id: "other", label: "other" },
];

export default function LlmCalls({
  data,
  onSelectIteration,
}: {
  data: RunData;
  onSelectIteration: (id: number) => void;
}) {
  const [role, setRole] = useState<LlmCall["role"] | "all">("all");

  if (!data.llmCalls.length) return <Empty>This run has no llm_calls.jsonl.</Empty>;

  const present = new Set(data.llmCalls.map((c) => c.role));
  const shown = data.llmCalls.filter((c) => role === "all" || c.role === role);
  const tokensIn = data.llmCalls.reduce((a, c) => a + (c.tokens_in ?? 0), 0);
  const tokensOut = data.llmCalls.reduce((a, c) => a + (c.tokens_out ?? 0), 0);

  return (
    <>
      <div className="tiles" style={{ marginBottom: 16 }}>
        <Tile label="Calls" value={data.llmCalls.length} />
        <Tile label="Tokens in" value={int(tokensIn)} />
        <Tile label="Tokens out" value={int(tokensOut)} />
        <Tile label="Model" value={<span style={{ fontSize: 15 }}>{data.llmCalls[0].model ?? "--"}</span>} />
      </div>

      <div className="filters">
        {ROLES.filter((r) => r.id === "all" || present.has(r.id as LlmCall["role"])).map((r) => (
          <button key={r.id} type="button" aria-pressed={role === r.id} onClick={() => setRole(r.id)}>
            {r.label}
            {r.id !== "all" && ` (${data.llmCalls.filter((c) => c.role === r.id).length})`}
          </button>
        ))}
        <span className="muted" style={{ fontSize: 12 }}>
          Iteration attribution is inferred from call order — the file records no iteration id.
        </span>
      </div>

      <Card>
        {shown.map((call) => (
          <details className="call" key={call.index}>
            <summary>
              <span className="pill">{call.role}</span>
              {call.iterId !== undefined && (
                <button
                  type="button"
                  className="chip"
                  onClick={(e) => {
                    e.preventDefault();
                    onSelectIteration(call.iterId!);
                  }}
                >
                  #{call.iterId}
                </button>
              )}
              <span className="preview">{truncate(call.response, 200)}</span>
              <span className="tokens">
                {int(call.tokens_in)} → {int(call.tokens_out)}
              </span>
            </summary>
            <div className="body">
              <div className="muted" style={{ fontSize: 12 }}>
                {timestamp(call.ts)} · {call.model}
              </div>
              <div>
                <h4>prompt</h4>
                <pre className="text">{call.prompt}</pre>
              </div>
              <div>
                <h4>response</h4>
                <pre className="text">{call.response}</pre>
              </div>
            </div>
          </details>
        ))}
      </Card>
    </>
  );
}
