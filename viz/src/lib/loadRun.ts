import { parseJsonl, asIterId, asRecord, asNumber } from "./jsonl";
import { deriveIterations, deriveLlmCalls } from "./derive";
import type {
  Belief, EnsembleRow, LedgerRow, Manifest, RunData, RunMeta,
} from "./types";

const base = (run: string) => `/rundata/${encodeURIComponent(run)}`;

async function fetchText(url: string): Promise<string | null> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) return null;
  return res.text();
}

export async function loadRun(run: string): Promise<RunData> {
  const warnings: string[] = [];

  const manifestText = await fetchText(`${base(run)}/_manifest.json`);
  if (!manifestText) {
    throw new Error(
      `Could not read runs/${run}. Check the "run" field in viz/run.config.json.`,
    );
  }
  const manifest = JSON.parse(manifestText) as Manifest;
  if (!manifest.exists) {
    throw new Error(
      `runs/${run} does not exist. Check the "run" field in viz/run.config.json.`,
    );
  }

  const has = (f: string) => manifest.files.includes(f);
  const text = async (f: string) => (has(f) ? await fetchText(`${base(run)}/${f}`) : null);

  const [
    metaText, ledgerText, llmText, candText, diagText, ensText,
    knowledgeText, knowledgeMd, reflections, eda, searchTree, consoleLog,
  ] = await Promise.all([
    text("run_meta.json"), text("ledger.jsonl"), text("llm_calls.jsonl"),
    text("candidates.jsonl"), text("diagnostics.jsonl"), text("harness_ensembles.jsonl"),
    text("knowledge.json"), text("knowledge.md"), text("reflections.md"),
    text("eda_report.txt"), text("search_tree.txt"), text("console.log"),
  ]);

  let meta: RunMeta | null = null;
  if (metaText) {
    try {
      meta = JSON.parse(metaText) as RunMeta;
    } catch (e) {
      warnings.push(`run_meta.json did not parse (${(e as Error).message})`);
    }
  }

  const ledger = parseJsonl<LedgerRow>(ledgerText ?? "");
  if (ledger.skipped) warnings.push(`ledger.jsonl: ${ledger.skipped} unparseable line(s) skipped`);
  if (!ledger.rows.length) warnings.push("ledger.jsonl is missing or empty -- no iterations to show");

  // Side files keyed by iter_id.
  const candidates = new Map<number, Record<string, number>>();
  for (const row of parseJsonl<Record<string, unknown>>(candText ?? "").rows) {
    const id = asIterId(row.iter_id);
    const rec = asRecord(row.candidates);
    if (id !== null && rec) candidates.set(id, rec);
  }

  const diagnostics = new Map<number, string>();
  for (const row of parseJsonl<Record<string, unknown>>(diagText ?? "").rows) {
    const id = asIterId(row.iter_id);
    if (id !== null && typeof row.report === "string") diagnostics.set(id, row.report);
  }

  const ensembles = new Map<number, EnsembleRow>();
  for (const row of parseJsonl<Record<string, unknown>>(ensText ?? "").rows) {
    const id = asIterId(row.iter_id);
    if (id === null) continue;
    ensembles.set(id, {
      selected_alpha: asNumber(row.selected_alpha),
      candidate_primary: asNumber(row.candidate_primary),
      incumbent_primary: asNumber(row.incumbent_primary),
      selected_primary: asNumber(row.selected_primary),
      grid: asRecord(row.grid),
    });
  }

  const llmCalls = deriveLlmCalls(parseJsonl<Record<string, unknown>>(llmText ?? "").rows);

  // Scripts. The ledger's `diff` field already holds the source, so these are fetched only
  // to prefer the on-disk file, and a failure is not fatal.
  const scripts = new Map<number, { path: string; source: string }>();
  await Promise.all(
    manifest.scripts.map(async (name) => {
      const m = /^iter_(\d+)\.py$/.exec(name);
      if (!m) return;
      const source = await fetchText(`${base(run)}/scripts/${name}`);
      if (source !== null) scripts.set(Number(m[1]), { path: `scripts/${name}`, source });
    }),
  );

  const { iterations, byId, roots } = deriveIterations(
    ledger.rows, { scripts, candidates, diagnostics, ensembles, llmCalls },
  );

  let beliefs: Belief[] = [];
  if (knowledgeText) {
    try {
      const parsed = JSON.parse(knowledgeText);
      if (Array.isArray(parsed)) beliefs = parsed as Belief[];
    } catch (e) {
      warnings.push(`knowledge.json did not parse (${(e as Error).message})`);
    }
  }

  return {
    run, manifest, meta, iterations, byId, roots, llmCalls, beliefs,
    reflections, knowledgeMd, eda, searchTree, console: consoleLog, warnings,
  };
}
