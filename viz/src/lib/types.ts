export type IterStatus = "ok" | "kept" | "reverted" | "failed" | string;

export interface Metrics {
  primary?: number;
  gauc?: number;
  "ndcg@5"?: number;
  "ndcg@10"?: number;
  "recall@50"?: number;
  gpu_seconds?: number;
  raw_candidate_primary?: number;
  harness_blend_alpha?: number;
  [k: string]: number | string | undefined;
}

export interface LedgerRow {
  iter_id: number;
  parent_iter_id: number | null;
  tier?: number;
  hypothesis?: string;
  /** The full source of the script this iteration ran -- despite the name, not a diff. */
  diff?: string;
  metrics?: Metrics | null;
  gpu_seconds?: number;
  tokens_in?: number;
  tokens_out?: number;
  status?: IterStatus;
  error?: string | null;
  phase?: string;
  timestamp?: number;
}

export interface Submission {
  iter_id: number;
  valid_primary?: number;
  test_primary?: number;
  test_gauc?: number;
  "test_ndcg@5"?: number;
  test_delta?: number;
  hypothesis?: string;
  /** False when the run never read test labels, which is the compliant case. */
  test_scored?: boolean;
  /** The predictions file this run wrote. */
  file?: string;
  source?: string;
}

export interface RunMeta {
  model?: string;
  dataset?: string;
  provider?: string;
  data_contract?: string;
  stop_reason?: string;
  iterations?: number;
  iteration_cap?: number;
  wall_clock_s?: number;
  script_seconds?: number;
  tokens_in?: number;
  tokens_out?: number;
  tokens_total?: number;
  proposer_tokens_in?: number;
  proposer_tokens_out?: number;
  eda_completed?: boolean;
  candidates_evaluated?: number;
  claims_established?: number;
  baseline_reproduced?: number;
  baseline_target?: number;
  manual_interventions?: number;
  failures?: number;
  integrity_rejections?: number;
  strict_convergence_iteration?: number;
  epsilon?: number;
  patience?: number;
  turns?: number;
  slots?: number;
  /** Present when run_meta.json was rebuilt from the records after a halt. */
  reconstructed_note?: string;
  api_surface?: string[];
  submission?: Submission;
  [k: string]: unknown;
}

export interface LlmCall {
  index: number;
  ts?: number;
  model?: string;
  prompt: string;
  response: string;
  tokens_in?: number;
  tokens_out?: number;
  role: "proposer" | "knowledge" | "reflection" | "other";
  /** Iteration this call belongs to, when it can be attributed. */
  iterId?: number;
}

export interface Belief {
  text: string;
  status?: string;
  evidence?: number[];
}

export interface EnsembleRow {
  selected_alpha?: number;
  candidate_primary?: number;
  incumbent_primary?: number;
  selected_primary?: number;
  grid?: Record<string, number>;
}

/** One node of the search tree: a ledger row plus everything keyed to its iter_id. */
export interface Iteration extends LedgerRow {
  script?: string;
  scriptPath?: string;
  candidates?: Record<string, number>;
  diagnostics?: string;
  ensemble?: EnsembleRow;
  /** A "failed" row that failed in the LLM transport, not in the agent's code. */
  infra: boolean;
  children: number[];
  depth: number;
  llmCalls: number[];
}

export interface Manifest {
  run: string;
  exists: boolean;
  files: string[];
  scripts: string[];
}

export interface RunData {
  run: string;
  manifest: Manifest;
  meta: RunMeta | null;
  iterations: Iteration[];
  byId: Map<number, Iteration>;
  roots: number[];
  llmCalls: LlmCall[];
  beliefs: Belief[];
  reflections: string | null;
  knowledgeMd: string | null;
  eda: string | null;
  searchTree: string | null;
  console: string | null;
  /** Files the manifest lists that failed to load, with the reason. */
  warnings: string[];
}
