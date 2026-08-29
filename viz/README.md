# Run viewer

A local website for reading one `runs/<id>` folder: the overview and score progression, the
search tree with per-iteration scripts, the LLM calls, and the belief set the agent ended with.

```powershell
cd viz
pnpm install
pnpm dev          # http://localhost:5173
```

## Switching runs

One run is shown at a time. Edit `run.config.json`:

```json
{ "run": "r70" }
```

Save it and the page reloads. `run` is any folder name under `runs/`.

Data is read straight off disk by a dev-server middleware (`vite-plugin-runs.ts`), so a run that
finished a moment ago shows up on refresh — nothing to rebuild or copy.

## Static export

```powershell
pnpm build        # dist/ + dist/rundata/<active run>/
pnpm preview
```

`pnpm build` copies only the run named in `run.config.json` into `dist/`, giving a self-contained
offline copy of that one run.

## What it reads

| File | Where it shows up |
|---|---|
| `run_meta.json` | Overview header, stat tiles, submission card, chart reference lines |
| `ledger.jsonl` | Every iteration: tree, chart, hypothesis, metrics, script |
| `scripts/iter_N.py` | Script tab, with a diff against the parent iteration |
| `candidates.jsonl` | Candidates tab — variants scored inside one script |
| `diagnostics.jsonl` | Diagnostics tab |
| `harness_ensembles.jsonl` | Ensemble tab — the alpha grid and what was selected |
| `llm_calls.jsonl` | LLM calls view; also the per-iteration Prompts tab |
| `knowledge.json` / `reflections.md` | Knowledge view |
| `eda_report.txt`, `console.log`, `knowledge.md` | EDA & logs view |

Only `ledger.jsonl` and `scripts/` exist in every run — the harness gained the other files over
time. Anything missing degrades to an empty state rather than breaking the page.

Two details worth knowing, both mirroring `report_run.py`:

- The chart's baseline reference lines come from the run's own `baseline_target`. KuaiRand-1K
  (`r38_1k`) has a different reference, and hardcoding Pure's 0.6016 would misreport it.
- A `failed` iteration whose error is an LLM transport error (`LLMError`, `LLMDailyLimit`, …) is
  labelled **api outage**, not a failed experiment — the agent's code never ran. `r41` lost six
  iterations to an expired key that way.
