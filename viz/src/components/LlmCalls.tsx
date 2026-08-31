import { useState } from "react";
import { ChevronRight } from "lucide-react";
import type { LlmCall, RunData } from "../lib/types";
import { int, timestamp, truncate } from "../lib/format";
import { Panel, Empty, StatGrid, Stat, Pre, IterLink } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

const ROLES: { id: LlmCall["role"] | "all"; label: string }[] = [
  { id: "all", label: "all" },
  { id: "proposer", label: "proposer" },
  { id: "knowledge", label: "knowledge" },
  { id: "reflection", label: "reflection" },
  { id: "other", label: "other" },
];

function CallRow({
  call,
  onSelectIteration,
}: {
  call: LlmCall;
  onSelectIteration: (id: number) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="py-2">
      {/* IterLink is a sibling of the trigger, not a child: a button cannot nest in a button. */}
      <div className="flex items-center gap-2">
        <CollapsibleTrigger className="focus-visible:ring-ring flex min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-md text-left outline-none focus-visible:ring-2">
          <ChevronRight
            className={cn("text-muted-foreground size-3.5 shrink-0 transition-transform", open && "rotate-90")}
          />
          <Badge variant="muted">{call.role}</Badge>
          <span className="text-ink-2 min-w-0 flex-1 truncate text-xs">{truncate(call.response, 200)}</span>
          <span className="tnum text-muted-foreground font-mono text-[11px]">
            {int(call.tokens_in)} → {int(call.tokens_out)}
          </span>
        </CollapsibleTrigger>
        {call.iterId !== undefined && <IterLink id={call.iterId} onClick={onSelectIteration} />}
      </div>
      <CollapsibleContent className="mt-2 flex flex-col gap-3 pl-5.5">
        <div className="text-muted-foreground text-xs">
          {timestamp(call.ts)} · {call.model}
        </div>
        <div>
          <h4 className="text-muted-foreground mb-1 text-[11px] font-semibold tracking-[0.06em] uppercase">
            prompt
          </h4>
          <Pre wrap className="max-h-[460px] overflow-auto">
            {call.prompt}
          </Pre>
        </div>
        <div>
          <h4 className="text-muted-foreground mb-1 text-[11px] font-semibold tracking-[0.06em] uppercase">
            response
          </h4>
          <Pre wrap className="max-h-[460px] overflow-auto">
            {call.response}
          </Pre>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

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
      <StatGrid className="mb-4">
        <Stat label="Calls" value={data.llmCalls.length} />
        <Stat label="Tokens in" value={int(tokensIn)} />
        <Stat label="Tokens out" value={int(tokensOut)} />
        <Stat label="Model" value={data.llmCalls[0].model ?? "--"} />
      </StatGrid>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <ToggleGroup
          type="single"
          value={role}
          onValueChange={(v) => v && setRole(v as LlmCall["role"] | "all")}
        >
          {ROLES.filter((r) => r.id === "all" || present.has(r.id as LlmCall["role"])).map((r) => (
            <ToggleGroupItem key={r.id} value={r.id}>
              {r.label}
              {r.id !== "all" && ` (${data.llmCalls.filter((c) => c.role === r.id).length})`}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        <span className="text-muted-foreground text-xs">
          Iteration attribution is inferred from call order - the file records no iteration id.
        </span>
      </div>

      <Panel>
        <div className="divide-y">
          {shown.map((call) => (
            <CallRow key={call.index} call={call} onSelectIteration={onSelectIteration} />
          ))}
        </div>
      </Panel>
    </>
  );
}
