# Human analysis — NOT the submission, and NOT the agent's priors

Experiments we ran by hand. None of this code is on the submission path, and — since the
architecture change described in `DEVPOST.md` — **none of these findings are given to the agent
either.** The agent's brief carries the task specification, the pipeline API and the output
contract, and nothing about what works on this dataset; two tests enforce that
(`test_brief_carries_no_human_findings`, `test_knowledge_base_carries_no_measured_results`).

The submission comes from an autonomous run of `run_agent.py`, because Autonomy is a scored
criterion measured by manual intervention count, and Innovation is scored on what *the agent*
identified as worth trying.

## Current

- `ceiling_probe.py` — fits a deliberately over-powered LightGBM and scores it in-sample and on
  validation. Establishes that capacity is not the binding constraint on this benchmark and
  transfer across the date split is: 0.9245 in-sample against 0.5868 on validation. Write-up in
  `docs/generalisation-ceiling.md`. This is context for reading the agent's delta; it is stated
  as our analysis, not the agent's.

## Historical — from the earlier prior-laden design

These informed the v1 brief, which is archived at `archive/proposer_v1_human_priors.py` (and the
matching knowledge base at `archive/papers_v1_human_priors.json`). They are kept as an honest
record of what was removed and why, not as inputs to the current agent.

- `blend_model.py` — hand-built FM + LGB(binary) + LGB(lambdarank) rank blend.
  valid 0.6045 / test 0.5984 (+0.0038 vs baseline).
- `make_submission.py` — hand-built 5-seed FM. test 0.5963 (+0.0017).
- `manual_blend_submission.csv` — output of `blend_model.py`, reference only.

For the record, the findings that used to be fed to the agent and no longer are:

- single-model search is close to exhausted (~12 approaches within ±0.005 of baseline)
- rank-blending decorrelated models beats every component individually
- pairwise/listwise ranking losses lose to logloss here (33% positive rate; GAUC rewards
  calibrated pointwise scores)
- duration is a weak signal; `long_view` is already duration-normalised
- capacity does not help: ~43 rows per user

Notably, the agent has since re-derived several of these unaided — the duration point and the
"capacity does not help" point both appear in its own EDA output and belief sets — and it
contradicted two of them:

- it found a DeepFM configuration (α = 0.75 mixing weight) that beats the baseline, where our
  hand-tuned DeepFM attempts had not;
- the v1 brief asserted that per-user sequences do not exist so DIN/DIEN/BST/SASRec/GRU4Rec are
  "NOT implementable here". That was wrong. Once `Split.date` was exposed the agent constructed
  history by ordering each user's rows by date and ran a DIN-style candidate-aware pooling over
  strictly-prior positive videos (r34 #5, 0.6032). It lost to DeepFM, but it ran.
