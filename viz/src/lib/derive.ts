import { asNumber } from "./jsonl";
import type { EnsembleRow, Iteration, LedgerRow, LlmCall } from "./types";

/** The agent's prompts are distinguishable by their opening line; llm_calls.jsonl carries
 * no role or iter_id of its own. The reflection stage is the older runs' equivalent of the
 * knowledge pass. */
function roleOf(prompt: string): LlmCall["role"] {
  if (prompt.startsWith("You are the proposer")) return "proposer";
  if (prompt.startsWith("You maintain the information state")) return "knowledge";
  if (prompt.startsWith("You are the reflection stage")) return "reflection";
  return "other";
}

/** Errors from the LLM transport, not from the agent's code -- the same list report_run.py
 * uses. A run that lost its API key did not fail six experiments; it failed to start six, and
 * counting those as failed experiments misreports the run in both directions. */
const INFRA_MARKERS = [
  "LLMError",
  "LLMRetryExhausted",
  "LLMDailyLimit",
  "LLMKeyRejected",
  "proposer returned nothing",
];

export function isInfraFailure(error: string | null | undefined): boolean {
  return !!error && INFRA_MARKERS.some((m) => error.includes(m));
}

/** The exception name, for a one-line label. */
export function infraKind(error: string | null | undefined): string {
  const head = (error ?? "").trim().split("\n")[0].trim();
  return INFRA_MARKERS.find((m) => head.includes(m)) ?? "LLM transport";
}

/**
 * Calls are appended in execution order: the proposer writes iteration N, then the knowledge
 * pass revises beliefs from its result. So each proposer call opens a new iteration and the
 * calls that follow it belong to that iteration, counting from the ledger's first iter_id.
 */
export function deriveLlmCalls(rows: Record<string, unknown>[]): LlmCall[] {
  let iterId = -1;
  return rows.map((row, index) => {
    const prompt = typeof row.prompt === "string" ? row.prompt : "";
    const role = roleOf(prompt);
    if (role === "proposer") iterId += 1;
    return {
      index,
      ts: asNumber(row.ts),
      model: typeof row.model === "string" ? row.model : undefined,
      prompt,
      response: typeof row.response === "string" ? row.response : "",
      tokens_in: asNumber(row.tokens_in),
      tokens_out: asNumber(row.tokens_out),
      role,
      iterId: iterId >= 0 ? iterId : undefined,
    };
  });
}

interface SideData {
  scripts: Map<number, { path: string; source: string }>;
  candidates: Map<number, Record<string, number>>;
  diagnostics: Map<number, string>;
  ensembles: Map<number, EnsembleRow>;
  llmCalls: LlmCall[];
}

/** Join the ledger to everything keyed by iter_id, then link parents to children. */
export function deriveIterations(rows: LedgerRow[], side: SideData) {
  const byId = new Map<number, Iteration>();
  const iterations: Iteration[] = [];

  for (const row of rows) {
    const id = Number(row.iter_id);
    const script = side.scripts.get(id);
    const it: Iteration = {
      ...row,
      iter_id: id,
      parent_iter_id:
        row.parent_iter_id === null || row.parent_iter_id === undefined
          ? null
          : Number(row.parent_iter_id),
      script: script?.source ?? row.diff,
      scriptPath: script?.path,
      infra: row.status === "failed" && isInfraFailure(row.error),
      candidates: side.candidates.get(id),
      diagnostics: side.diagnostics.get(id),
      ensemble: side.ensembles.get(id),
      children: [],
      depth: 0,
      llmCalls: side.llmCalls.filter((c) => c.iterId === id).map((c) => c.index),
    };
    byId.set(id, it);
    iterations.push(it);
  }

  const roots: number[] = [];
  for (const it of iterations) {
    const parent = it.parent_iter_id === null ? undefined : byId.get(it.parent_iter_id);
    if (parent) parent.children.push(it.iter_id);
    else roots.push(it.iter_id);
  }

  // Depth, walking down from each root. Guarded against a cycle in a malformed ledger.
  const seen = new Set<number>();
  const walk = (id: number, depth: number) => {
    if (seen.has(id)) return;
    seen.add(id);
    const it = byId.get(id);
    if (!it) return;
    it.depth = depth;
    for (const child of it.children) walk(child, depth + 1);
  };
  for (const r of roots) walk(r, 0);

  return { iterations, byId, roots };
}

export const primaryOf = (it: Iteration): number | undefined => asNumber(it.metrics?.primary);

/** The iteration the harness would submit: best validation primary, earliest on a tie. */
export function bestIteration(iterations: Iteration[]): Iteration | undefined {
  let best: Iteration | undefined;
  for (const it of iterations) {
    const p = primaryOf(it);
    if (p === undefined) continue;
    const bp = best ? primaryOf(best) : undefined;
    if (bp === undefined || p > bp) best = it;
  }
  return best;
}
