import type { RunIndex, RunSummary } from "../../run-index";

export type { RunIndex, RunSummary };
export type Eligibility = RunSummary["eligibility"];

/** How each class reads to a judge, in the order the provenance map should stack them. */
export const ELIGIBILITY_ORDER: Eligibility[] = [
  "submitted",
  "eligible",
  "leakage",
  "legacy-contract",
  "bonus-dataset",
  "unverified",
  "scratch",
];

export const ELIGIBILITY_LABEL: Record<Eligibility, string> = {
  submitted: "submitted",
  eligible: "eligible",
  leakage: "excluded: leakage",
  "legacy-contract": "superseded contract",
  "bonus-dataset": "bonus dataset",
  unverified: "unverified",
  scratch: "scratch",
};

/** Tailwind background token per class. Static strings so the compiler can see them. */
export const ELIGIBILITY_FILL: Record<Eligibility, string> = {
  submitted: "bg-good",
  eligible: "bg-primary",
  leakage: "bg-critical",
  "legacy-contract": "bg-axis",
  "bonus-dataset": "bg-chart-2",
  unverified: "bg-warning",
  scratch: "bg-line-strong",
};

export async function loadRunIndex(): Promise<RunIndex> {
  const res = await fetch("/rundata/_index.json");
  if (!res.ok) throw new Error(`run index unavailable (${res.status})`);
  return (await res.json()) as RunIndex;
}

/** Runs that may be selected from: current contract, no future-window columns. */
export function selectable(index: RunIndex): RunSummary[] {
  return index.runs.filter(
    (r) => r.eligibility === "submitted" || r.eligibility === "eligible",
  );
}

/**
 * The run the project's own rule picks: highest validation primary among selectable runs.
 * Selection never looks at test, so ties break on the earlier run rather than the better delta.
 */
export function validationBest(index: RunIndex): RunSummary | null {
  let best: RunSummary | null = null;
  for (const r of selectable(index)) {
    if (r.bestValid === null) continue;
    if (!best || best.bestValid === null || r.bestValid > best.bestValid) best = r;
  }
  return best;
}
