# limits/ — ceiling probe (scratch, NOT on the submission path)

Question: if a human MLE hand-builds the model instead of the agent, how high does the
KuaiRand-Pure primary go, and where is the wall?

These scripts read **test labels on purpose** to measure the ceiling. Nothing here may be used
to build a submission, and no result here enters `runs/`, cross-run memory, or DEVPOST.md.

- `features.py` — tabular feature assembly (item / context / user groups, train-only history, affinity)
- `affinity.py` — user x item-attribute affinity fitted on train, leave-one-out on train rows
- `probe.py`    — LightGBM configs, valid-selected and oracle-selected scores
- `nn.py`       — FM/DeepFM with pointwise BCE or within-user pairwise (BPR) loss
- `blend.py`    — per-user rank blend, weights chosen on validation only
- `out/`        — per-config predictions and `results.json`

Every row reports **valid** (honest selection signal), **test** (score at the valid-selected
iteration) and **oracle_test** (best iteration by test) so the cost of selection is visible.
