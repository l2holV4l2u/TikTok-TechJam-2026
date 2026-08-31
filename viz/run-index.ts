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
  /** Which file in the run folder holds the predictions. */
  submissionFile: string | null;
  /** Turns, not scripts: each turn ran its slots in parallel. */
  turns: number | null;
  /** The convergence rule this run declared, so the UI never hardcodes it. */
  epsilon: number | null;
  patience: number | null;
  /** Summed script runtime; larger than wallClockS whenever slots overlapped. */
  scriptSeconds: number | null;
  llmCalls: number | null;
  tokensIn: number | null;
  tokensOut: number | null;
  /** The organizer-facing item outcome columns this run exposed, if any. */
  outcomeFields: string[];
}

export interface RunIndex {
  generated: number;
  /** The run whose submission file is byte-identical to the repo's submission_best.csv. */
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

/** Every predictions file a run left behind, newest first. */
function submissionFiles(dir: string): string[] {
  try {
    return fs
      .readdirSync(dir)
      .filter((f) => /^submission.*\.csv$/.test(f))
      .map((f) => path.join(dir, f))
      .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  } catch {
    return [];
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
  /** Highest turn number seen. A turn runs its slots in parallel, so turns <= rows. */
  turns: number | null;
  /** First-to-last timestamp: the agent's own wall-clock, not the sum of its scripts. */
  wallClockS: number | null;
  /** Summed script runtime. Exceeds wall-clock whenever slots ran concurrently. */
  scriptSeconds: number | null;
}

/** One JSON object per line, skipping anything unparseable rather than failing the whole index. */
function readJsonl(file: string): Record<string, unknown>[] {
  let text: string;
  try {
    text = fs.readFileSync(file, "utf-8");
  } catch {
    return [];
  }
  const out: Record<string, unknown>[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      out.push(JSON.parse(line));
    } catch {
      /* a half-written trailing line while a run is live */
    }
  }
  return out;
}

/** Token spend, counted off the LLM log itself rather than the ledger's per-iteration echo. */
function scanLlmCalls(file: string) {
  const calls = readJsonl(file);
  if (!calls.length) return { llmCalls: null, tokensIn: null, tokensOut: null, model: null };
  let tokensIn = 0;
  let tokensOut = 0;
  for (const c of calls) {
    if (typeof c.tokens_in === "number") tokensIn += c.tokens_in;
    if (typeof c.tokens_out === "number") tokensOut += c.tokens_out;
  }
  return {
    llmCalls: calls.length,
    tokensIn,
    tokensOut,
    model: typeof calls[0].model === "string" ? calls[0].model : null,
  };
}

/**
 * Models compared, not scripts run. One script may build and score many candidates, and the
 * convergence rule charges per iteration -- so this is the number the design is trying to raise.
 */
function countCandidates(file: string): number | null {
  const rows = readJsonl(file);
  if (!rows.length) return null;
  let n = 0;
  // Each row maps candidate name -> validation score; older runs wrote a plain array.
  for (const r of rows) {
    const c = r.candidates;
    if (Array.isArray(c)) n += c.length;
    else if (c && typeof c === "object") n += Object.keys(c).length;
  }
  return n || null;
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
    turns: null,
    wallClockS: null,
    scriptSeconds: null,
  };
  let firstTs: number | null = null;
  let lastTs: number | null = null;
  let gpuSeconds = 0;
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

    if (typeof row.turn === "number") {
      scan.turns = scan.turns === null ? row.turn : Math.max(scan.turns, row.turn);
    }
    if (typeof row.timestamp === "number") {
      if (firstTs === null || row.timestamp < firstTs) firstTs = row.timestamp;
      if (lastTs === null || row.timestamp > lastTs) lastTs = row.timestamp;
    }
    if (typeof row.gpu_seconds === "number") gpuSeconds += row.gpu_seconds;
  }

  if (firstTs !== null && lastTs !== null && lastTs > firstTs) scan.wallClockS = lastTs - firstTs;
  if (gpuSeconds > 0) scan.scriptSeconds = gpuSeconds;
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
  turns?: number;
  claims_established?: number;
  baseline_reproduced?: number;
  baseline_target?: number;
  api_surface?: string[];
  epsilon?: number;
  patience?: number;
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
    // run_meta.json is written only when a run exits cleanly, so every figure it carries is
    // also derived from the records -- a halted run still reports what it spent.
    const llm = scanLlmCalls(path.join(dir, "llm_calls.jsonl"));
    const candidates = countCandidates(path.join(dir, "candidates.jsonl"));
    // The harness names its predictions submission.csv, submission_checkpoint.csv or
    // submission_turn<N>.csv depending on how the run ended, so match on the shape.
    const subFiles = submissionFiles(dir);
    // The one that is byte-identical to submission_best.csv wins; else the newest.
    const matched = bestSha ? subFiles.find((f) => sha256(f) === bestSha) : undefined;
    const subPath = matched ?? subFiles[0];
    const hasSubmission = subPath !== undefined;
    const isSubmitted = matched !== undefined;
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
      model: meta?.model ?? llm.model,
      provider: meta?.provider ?? null,
      stopReason: meta?.stop_reason ?? null,
      iterations: meta?.iterations ?? (scan.rows || null),
      iterationCap: meta?.iteration_cap ?? null,
      wallClockS: meta?.wall_clock_s ?? scan.wallClockS,
      tokensTotal:
        meta?.tokens_total ??
        (llm.tokensIn !== null && llm.tokensOut !== null ? llm.tokensIn + llm.tokensOut : null),
      manualInterventions: meta?.manual_interventions ?? null,
      scriptFailures: scan.scriptFailures,
      infraFailures: scan.infraFailures,
      candidatesEvaluated: meta?.candidates_evaluated ?? candidates,
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
      submissionFile: subPath ? path.basename(subPath) : null,
      turns: meta?.turns ?? scan.turns,
      epsilon: meta?.epsilon ?? null,
      patience: meta?.patience ?? null,
      scriptSeconds: scan.scriptSeconds,
      llmCalls: llm.llmCalls,
      tokensIn: llm.tokensIn,
      tokensOut: llm.tokensOut,
      outcomeFields: outcome,
    });
  }

  // Natural order on the rN numbering, so r9 sorts before r10 and suffixed ids stay together.
  runs.sort((a, b) => a.run.localeCompare(b.run, undefined, { numeric: true }));

  return { generated: Date.now(), submittedRun, submissionSha256: bestSha, runs };
}
