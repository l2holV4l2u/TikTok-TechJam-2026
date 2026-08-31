import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

/**
 * Builds one cross-run index from the run records alone. Nothing here is hand-curated:
 * every field is read or derived from runs/<id>/ and the repo's submission_best.csv.
 */

export interface RunSummary {
  run: string;
  /** Where this run sits in the audit trail. Derived, never hand-listed. */
  eligibility:
    | "submitted"
    | "eligible"
    | "leakage"
    | "legacy-contract"
    | "bonus-dataset"
    | "unverified"
    | "scratch";
  /** Why it landed in that class, in one phrase, for the UI to show on hover. */
  reason: string;
  dataset: string | null;
  contract: string | null;
  model: string | null;
  provider: string | null;
  stopReason: string | null;
  iterations: number | null;
  iterationCap: number | null;
  wallClockS: number | null;
  tokensTotal: number | null;
  manualInterventions: number | null;
  /** Script failures, i.e. the agent's own code erroring -- LLM outages counted separately. */
  scriptFailures: number;
  infraFailures: number;
  candidatesEvaluated: number | null;
  claimsEstablished: number | null;
  baselineReproduced: number | null;
  baselineTarget: number | null;
  /** Best validation primary over the ledger, and the iteration that reached it. */
  bestValid: number | null;
  bestValidIter: number | null;
  submissionIter: number | null;
  validPrimary: number | null;
  testPrimary: number | null;
  testGauc: number | null;
  testNdcg5: number | null;
  testDelta: number | null;
  ledgerRows: number;
  mtime: number;
  hasSubmission: boolean;
  /** The organizer-facing item outcome columns this run exposed, if any. */
  outcomeFields: string[];
}

export interface RunIndex {
  generated: number;
  /** The run whose submission.csv is byte-identical to the repo's submission_best.csv. */
  submittedRun: string | null;
  submissionSha256: string | null;
  runs: RunSummary[];
}

/**
 * video_features_statistic_pure.csv aggregates item outcomes over the whole log month, which
 * overlaps validation and test. A run that exposed any of them is not selection-eligible.
 */
function outcomeStatFields(apiSurface: string[]): string[] {
  const out: string[] = [];
  for (const entry of apiSurface) {
    const m = /^s\.num\[(.+)\]$/.exec(entry);
    if (!m) continue;
    const f = m[1];
    // user_-prefixed profile counts are the user's own attributes, not item outcomes.
    if (f.startsWith("user_")) continue;
    if (f.endsWith("_cnt") || f.startsWith("play_") || f.startsWith("show_") || f === "counts") {
      out.push(f);
    }
  }
  return out.sort();
}

function readJson<T>(file: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(file, "utf-8")) as T;
  } catch {
    return null;
  }
}

function sha256(file: string): string | null {
  try {
    return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
  } catch {
    return null;
  }
}

interface LedgerScan {
  rows: number;
  bestValid: number | null;
  bestValidIter: number | null;
  scriptFailures: number;
  infraFailures: number;
}

/** An LLM transport error means the agent's code never ran -- not a failed experiment. */
const INFRA_ERROR = /(LLMError|LLMDailyLimit|LLMUnavailable|RateLimit|proposer_unavailable)/i;

function scanLedger(file: string): LedgerScan {
  const scan: LedgerScan = {
    rows: 0,
    bestValid: null,
    bestValidIter: null,
    scriptFailures: 0,
    infraFailures: 0,
  };
  let text: string;
  try {
    text = fs.readFileSync(file, "utf-8");
  } catch {
    return scan;
  }
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    let row: Record<string, unknown>;
    try {
      row = JSON.parse(line);
    } catch {
      continue;
    }
    scan.rows++;
    const status = String(row.status ?? "");
    const error = String(row.error ?? "");
    if (status === "failed") {
      if (INFRA_ERROR.test(error)) scan.infraFailures++;
      else scan.scriptFailures++;
    }
    const metrics = row.metrics as Record<string, unknown> | null | undefined;
    const primary = metrics && typeof metrics.primary === "number" ? metrics.primary : null;
    if (primary !== null && (scan.bestValid === null || primary > scan.bestValid)) {
      scan.bestValid = primary;
      scan.bestValidIter = typeof row.iter_id === "number" ? row.iter_id : null;
    }
  }
  return scan;
}

interface RunMetaShape {
  dataset?: string;
  data_contract?: string;
  model?: string;
  provider?: string;
  stop_reason?: string;
  iterations?: number;
  iteration_cap?: number;
  wall_clock_s?: number;
  tokens_total?: number;
  manual_interventions?: number;
  candidates_evaluated?: number;
  claims_established?: number;
  baseline_reproduced?: number;
  baseline_target?: number;
  api_surface?: string[];
  submission?: {
    iter_id?: number;
    valid_primary?: number;
    test_primary?: number;
    test_gauc?: number;
    "test_ndcg@5"?: number;
    test_delta?: number;
  };
}

const SCRATCH_NAME = /^(dry|.*audit.*|.*scratch.*)/i;
const CURRENT_CONTRACT = "train-plus-valid-v2";

function classify(
  run: string,
  meta: RunMetaShape | null,
  scan: LedgerScan,
  outcome: string[],
  isSubmitted: boolean,
): { eligibility: RunSummary["eligibility"]; reason: string } {
  if (isSubmitted) {
    return { eligibility: "submitted", reason: "submission_best.csv is byte-identical to this run" };
  }
  if (SCRATCH_NAME.test(run) || scan.rows === 0) {
    return { eligibility: "scratch", reason: "dry run or no ledger rows" };
  }
  if (outcome.length) {
    return {
      eligibility: "leakage",
      reason: `exposed ${outcome.length} full-month item outcome column${outcome.length > 1 ? "s" : ""} (${outcome.slice(0, 3).join(", ")}…)`,
    };
  }
  const dataset = meta?.dataset ?? null;
  if (dataset && dataset !== "KuaiRand-Pure") {
    return { eligibility: "bonus-dataset", reason: `${dataset}: scores not comparable to Pure` };
  }
  if (!meta) {
    // Without run_meta.json neither the contract nor the API surface can be checked, so the
    // run is unverifiable rather than judged either way.
    return { eligibility: "unverified", reason: "no run_meta.json: contract and API surface unknown" };
  }
  const contract = meta.data_contract ?? null;
  if (contract !== CURRENT_CONTRACT) {
    return {
      eligibility: "legacy-contract",
      reason: contract
        ? `data contract ${contract}, superseded`
        : "predates the data-contract field",
    };
  }
  return { eligibility: "eligible", reason: "current contract, no outcome columns" };
}

export function buildRunIndex(runsDir: string, repoRoot: string): RunIndex {
  const bestPath = path.join(repoRoot, "submission_best.csv");
  const bestSha = sha256(bestPath);

  const names = fs.existsSync(runsDir)
    ? fs
        .readdirSync(runsDir, { withFileTypes: true })
        .filter((e) => e.isDirectory())
        .map((e) => e.name)
    : [];

  let submittedRun: string | null = null;
  const runs: RunSummary[] = [];

  for (const run of names) {
    const dir = path.join(runsDir, run);
    const meta = readJson<RunMetaShape>(path.join(dir, "run_meta.json"));
    const scan = scanLedger(path.join(dir, "ledger.jsonl"));
    const subPath = path.join(dir, "submission.csv");
    const hasSubmission = fs.existsSync(subPath);
    const isSubmitted = Boolean(bestSha && hasSubmission && sha256(subPath) === bestSha);
    if (isSubmitted) submittedRun = run;

    const outcome = outcomeStatFields(meta?.api_surface ?? []);
    const { eligibility, reason } = classify(run, meta, scan, outcome, isSubmitted);
    const sub = meta?.submission;

    let mtime = 0;
    try {
      mtime = fs.statSync(dir).mtimeMs;
    } catch {
      /* a folder that vanished mid-scan simply sorts last */
    }

    runs.push({
      run,
      eligibility,
      reason,
      dataset: meta?.dataset ?? null,
      contract: meta?.data_contract ?? null,
      model: meta?.model ?? null,
      provider: meta?.provider ?? null,
      stopReason: meta?.stop_reason ?? null,
      iterations: meta?.iterations ?? (scan.rows || null),
      iterationCap: meta?.iteration_cap ?? null,
      wallClockS: meta?.wall_clock_s ?? null,
      tokensTotal: meta?.tokens_total ?? null,
      manualInterventions: meta?.manual_interventions ?? null,
      scriptFailures: scan.scriptFailures,
      infraFailures: scan.infraFailures,
      candidatesEvaluated: meta?.candidates_evaluated ?? null,
      claimsEstablished: meta?.claims_established ?? null,
      baselineReproduced: meta?.baseline_reproduced ?? null,
      baselineTarget: meta?.baseline_target ?? null,
      bestValid: scan.bestValid,
      bestValidIter: scan.bestValidIter,
      submissionIter: sub?.iter_id ?? null,
      validPrimary: sub?.valid_primary ?? null,
      testPrimary: sub?.test_primary ?? null,
      testGauc: sub?.test_gauc ?? null,
      testNdcg5: sub?.["test_ndcg@5"] ?? null,
      testDelta: sub?.test_delta ?? null,
      ledgerRows: scan.rows,
      mtime,
      hasSubmission,
      outcomeFields: outcome,
    });
  }

  // Natural order on the rN numbering, so r9 sorts before r10 and suffixed ids stay together.
  runs.sort((a, b) => a.run.localeCompare(b.run, undefined, { numeric: true }));

  return { generated: Date.now(), submittedRun, submissionSha256: bestSha, runs };
}
