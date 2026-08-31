# Run viewer

A local website for reading this repo's runs. It opens on **Evidence** - the submitted result,
what it cost, and which of the ~70 run folders count toward it - and drills down into any single
`runs/<id>`: the overview and score progression, the search tree with per-iteration scripts, the
LLM calls, and the belief set the agent ended with.

```powershell
cd viz
pnpm install
pnpm dev          # http://localhost:5173
```

## Evidence

The landing view answers the five judging criteria in order, and **every figure on it is derived
from the run records** - nothing is copied from `DEVPOST.md` or `README.md`, so the page cannot
drift away from what the runs actually recorded.

- **Verdict** - the submitted run's hidden-test delta, placed on the benchmark's attainable span
  (random 0.4753 → perfect-ranking ceiling 0.8645, official baseline 0.5946), plus robustness,
  autonomy and budget bars.
- **Cross-run comparison** - validation best and hidden-test delta per comparable run, with the
  baseline's own ±0.0008 seed-noise band drawn, so runs that are not actually distinguishable
  do not look like they are.
- **Run provenance** - every run folder, classified and filterable.

Classification is derived, never hand-listed (`run-index.ts`):

| class | how it is decided |
|---|---|
| `submitted` | the run's `submission.csv` is byte-identical (sha256) to `submission_best.csv` |
| `eligible` | current data contract, no future-window columns in `api_surface` |
| `leakage` | `api_surface` exposed full-month item outcome columns (`*_cnt`, `play_*`, `show_*`) |
| `legacy-contract` | ran under a superseded `data_contract` |
| `bonus-dataset` | not KuaiRand-Pure, so not comparable |
| `unverified` | no `run_meta.json`, so contract and API surface cannot be checked |
| `scratch` | dry run, or no ledger rows |

`unverified` is deliberately not a verdict: those runs are neither claimed nor excluded, because
the records alone cannot settle it.

## Switching runs

Use the picker at the top of the sidebar - it is grouped by the classes above. The choice lives
in the URL hash (`#/r79`), so a link opens on the same run. `run.config.json` still sets the
run the page opens on when the URL names none, and is what `pnpm build` exports.

Data is read straight off disk by a dev-server middleware (`vite-plugin-runs.ts`), so a run that
finished a moment ago shows up on refresh - nothing to rebuild or copy. The cross-run index is
served at `/rundata/_index.json` and is built once per dev server; append `?fresh=1` to rebuild
it after adding a run.

## Static export

```powershell
pnpm build        # dist/ + dist/rundata/<active run>/
pnpm preview
```

`pnpm build` copies only the run named in `run.config.json` into `dist/`, giving a self-contained
offline copy of that one run. The cross-run index is written too, so **Evidence** still works in
the export - the other runs are summarised there, they just cannot be opened.

## What it reads

| File | Where it shows up |
|---|---|
| every `runs/*/run_meta.json` + `ledger.jsonl` | Evidence: verdict, cross-run comparison, provenance |
| `submission_best.csv` (repo root) | Evidence: hashed against each run's `submission.csv` to identify the submitted run |
| `run_meta.json` | Overview header, stat tiles, submission card, chart reference lines |
| `ledger.jsonl` | Every iteration: tree, chart, hypothesis, metrics, script |
| `scripts/iter_N.py` | Script tab, with a diff against the parent iteration |
| `candidates.jsonl` | Candidates tab - variants scored inside one script |
| `diagnostics.jsonl` | Diagnostics tab |
| `harness_ensembles.jsonl` | Ensemble tab - the alpha grid and what was selected |
| `llm_calls.jsonl` | LLM calls view; also the per-iteration Prompts tab |
| `knowledge.json` / `reflections.md` | Knowledge view |
| `eda_report.txt`, `console.log`, `knowledge.md` | EDA & logs view |

Only `ledger.jsonl` and `scripts/` exist in every run - the harness gained the other files over
time. Anything missing degrades to an empty state rather than breaking the page.

Two details worth knowing, both mirroring `report_run.py`:

- The chart's baseline reference lines come from the run's own `baseline_target`. KuaiRand-1K
  (`r38_1k`) has a different reference, and hardcoding Pure's 0.6016 would misreport it.
- A `failed` iteration whose error is an LLM transport error (`LLMError`, `LLMDailyLimit`, …) is
  labelled **api outage**, not a failed experiment - the agent's code never ran. `r41` lost six
  iterations to an expired key that way.
