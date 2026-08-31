import { useEffect, useMemo, useState } from "react";
import { ChevronRight } from "lucide-react";
import type { Iteration, LlmCall, RunData } from "../lib/types";
import { infraKind } from "../lib/derive";
import { duration, int, score, timestamp } from "../lib/format";
import { Panel, Empty, StatusBadge, KeyValue, Row, Pre, Note } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import ScriptView from "./ScriptView";

const METRIC_ORDER = ["primary", "gauc", "ndcg@5", "ndcg@10", "recall@50", "raw_candidate_primary", "harness_blend_alpha"];

type Tab = "script" | "candidates" | "diagnostics" | "ensemble" | "prompts";

function Metrics({ it }: { it: Iteration }) {
  const m = it.metrics ?? {};
  const keys = [...METRIC_ORDER.filter((k) => m[k] !== undefined), ...Object.keys(m).filter((k) => !METRIC_ORDER.includes(k) && k !== "gpu_seconds")];
  if (!keys.length) return <p className="text-muted-foreground text-sm">No metrics recorded - the script did not finish.</p>;
  return (
    <KeyValue>
      {keys.map((k) => (
        <Row key={k} k={k}>
          <span className="font-mono tnum">{typeof m[k] === "number" ? score(m[k] as number) : String(m[k])}</span>
        </Row>
      ))}
      <Row k="script time">
        <span className="font-mono tnum">{duration(it.gpu_seconds)}</span>
      </Row>
      <Row k="tokens">
        <span className="font-mono tnum">
          {int(it.tokens_in)} in / {int(it.tokens_out)} out
        </span>
      </Row>
      <Row k="finished">
        <span className="font-mono tnum">{timestamp(it.timestamp)}</span>
      </Row>
    </KeyValue>
  );
}

function Candidates({ candidates }: { candidates: Record<string, number> }) {
  const entries = Object.entries(candidates).sort((a, b) => b[1] - a[1]);
  const values = entries.map(([, v]) => v);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  // Bars start at a floor just below the worst candidate: these scores differ in the 3rd
  // decimal, and a zero-baseline bar would render them all identical.
  const floor = lo - (hi - lo || 0.001) * 0.25;

  return (
    <>
      <p className="text-ink-2 mb-3 text-sm">
        Variants this one script built and scored internally. Bars are drawn from {score(floor)}, not
        zero, so differences at the fourth decimal stay visible.
      </p>
      <div className="flex flex-col gap-1.5">
        {entries.map(([name, v], i) => (
          <div key={name} className="grid grid-cols-[minmax(120px,210px)_minmax(0,1fr)_74px] items-center gap-2.5 text-xs">
            <span className="truncate font-mono" title={name}>
              {name}
            </span>
            <span className="bg-secondary relative h-3.5 rounded">
              <span
                className={cn("absolute inset-y-0 left-0 rounded", i === 0 ? "bg-good" : "bg-primary")}
                style={{ width: `${((v - floor) / (hi - floor)) * 100}%` }}
              />
            </span>
            <span className="tnum text-right font-mono">{score(v)}</span>
          </div>
        ))}
      </div>
    </>
  );
}

function PromptRow({ call }: { call: LlmCall }) {
  const [open, setOpen] = useState(false);
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="py-2">
      <CollapsibleTrigger className="focus-visible:ring-ring flex w-full cursor-pointer items-center gap-2 rounded-md text-left outline-none focus-visible:ring-2">
        <ChevronRight className={cn("text-muted-foreground size-3.5 shrink-0 transition-transform", open && "rotate-90")} />
        <Badge variant="muted">{call.role}</Badge>
        <span className="text-ink-2 min-w-0 flex-1 truncate text-xs">{call.response.slice(0, 160)}</span>
        <span className="tnum text-muted-foreground font-mono text-[11px]">
          {int(call.tokens_in)} → {int(call.tokens_out)}
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 flex flex-col gap-3 pl-5.5">
        <div>
          <h4 className="text-muted-foreground mb-1 text-[11px] font-semibold tracking-[0.06em] uppercase">prompt</h4>
          <Pre wrap className="max-h-[460px] overflow-auto">
            {call.prompt}
          </Pre>
        </div>
        <div>
          <h4 className="text-muted-foreground mb-1 text-[11px] font-semibold tracking-[0.06em] uppercase">response</h4>
          <Pre wrap className="max-h-[460px] overflow-auto">
            {call.response}
          </Pre>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

export default function IterationDetail({ data, iteration }: { data: RunData; iteration?: Iteration }) {
  const [tab, setTab] = useState<Tab>("script");

  const available = useMemo<Tab[]>(
    () =>
      iteration
        ? ([
            "script",
            iteration.candidates ? "candidates" : null,
            iteration.diagnostics ? "diagnostics" : null,
            iteration.ensemble ? "ensemble" : null,
            iteration.llmCalls.length ? "prompts" : null,
          ].filter(Boolean) as Tab[])
        : [],
    [iteration],
  );

  useEffect(() => {
    if (available.length && !available.includes(tab)) setTab(available[0]);
  }, [available, tab]);

  if (!iteration) return <Empty>Pick an iteration on the left.</Empty>;

  const parent = iteration.parent_iter_id === null ? undefined : data.byId.get(iteration.parent_iter_id);
  const ens = iteration.ensemble;

  return (
    <div>
      <Panel>
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="font-mono">
              #{iteration.iter_id}
            </Badge>
            <StatusBadge status={iteration.status} infra={iteration.infra} />
            <Badge variant="outline">{iteration.phase}</Badge>
            {parent && <span className="text-muted-foreground font-mono text-xs">from #{parent.iter_id}</span>}
            {iteration.children.length > 0 && (
              <span className="text-muted-foreground font-mono text-xs">
                → {iteration.children.map((c) => `#${c}`).join(", ")}
              </span>
            )}
          </div>

          <p className="text-sm">{iteration.hypothesis ?? "No hypothesis recorded."}</p>

          {iteration.error && (
            <Note tone={iteration.infra ? "warning" : "critical"}>
              <div className="mb-1 text-[11px] font-semibold tracking-[0.06em] uppercase">
                {iteration.infra
                  ? `${infraKind(iteration.error)} - the LLM call failed, so this iteration never ran`
                  : "error"}
              </div>
              <pre className="font-mono text-xs whitespace-pre-wrap">{iteration.error}</pre>
            </Note>
          )}

          <Metrics it={iteration} />
        </div>
      </Panel>

      <Panel className="mt-4">
        <Tabs value={tab} onValueChange={(v) => setTab(v as Tab)}>
          <TabsList>
            {available.map((t) => (
              <TabsTrigger key={t} value={t}>
                {t === "prompts" ? `prompts (${iteration.llmCalls.length})` : t}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value="script">
            <ScriptView
              source={iteration.script}
              path={iteration.scriptPath}
              parentSource={parent?.script}
              parentLabel={parent ? `#${parent.iter_id}` : undefined}
            />
          </TabsContent>

          {iteration.candidates && (
            <TabsContent value="candidates">
              <Candidates candidates={iteration.candidates} />
            </TabsContent>
          )}

          {iteration.diagnostics && (
            <TabsContent value="diagnostics">
              <Pre wrap>{iteration.diagnostics.trimEnd()}</Pre>
            </TabsContent>
          )}

          {ens && (
            <TabsContent value="ensemble" className="flex flex-col gap-4">
              <KeyValue>
                <Row k="selected alpha">
                  <span className="font-mono tnum">{ens.selected_alpha ?? "--"}</span>
                </Row>
                <Row k="candidate primary">
                  <span className="font-mono tnum">{score(ens.candidate_primary)}</span>
                </Row>
                <Row k="incumbent primary">
                  <span className="font-mono tnum">{score(ens.incumbent_primary)}</span>
                </Row>
                <Row k="selected primary">
                  <span className="font-mono tnum">{score(ens.selected_primary)}</span>
                </Row>
              </KeyValue>
              {ens.grid && (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>alpha</TableHead>
                      <TableHead className="text-right">primary</TableHead>
                      <TableHead />
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.entries(ens.grid)
                      .sort((a, b) => Number(a[0]) - Number(b[0]))
                      .map(([alpha, v]) => {
                        const isSelected = ens.selected_alpha !== undefined && Number(alpha) === ens.selected_alpha;
                        return (
                          <TableRow key={alpha}>
                            <TableCell className="font-mono tnum">{alpha}</TableCell>
                            <TableCell className="text-right font-mono tnum">{score(v)}</TableCell>
                            <TableCell>{isSelected && <Badge variant="good">selected</Badge>}</TableCell>
                          </TableRow>
                        );
                      })}
                  </TableBody>
                </Table>
              )}
            </TabsContent>
          )}

          {iteration.llmCalls.length > 0 && (
            <TabsContent value="prompts">
              <p className="text-muted-foreground mb-2 text-xs">
                Attributed by order - <code className="bg-secondary rounded px-1 py-0.5 font-mono">llm_calls.jsonl</code> carries
                no iteration id.
              </p>
              <div className="divide-y">
                {iteration.llmCalls.map((idx) => (
                  <PromptRow key={idx} call={data.llmCalls[idx]} />
                ))}
              </div>
            </TabsContent>
          )}
        </Tabs>
      </Panel>
    </div>
  );
}
