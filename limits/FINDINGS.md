# How high can this benchmark actually go?

Hand-built models, no agent. Reproduce with the scripts in this folder; every number below
comes from `limits/out/results.json` or the run log of the named script.

Scratch work — **nothing here is on the submission path** and these scripts read test labels
deliberately, which a submission may never do.

## 1. Where a hand-built pipeline lands

| model | valid | test | vs baseline |
|---|---|---|---|
| official baseline (FM k=16, 5 fields) | 0.6016 | 0.5946 | - |
| my FM reproduction (`fm_bce`) | 0.6003 | 0.5934 | -0.0012 |
| LightGBM, all categorical + session (`gbdt`, 5-seed bag) | 0.6036 | 0.5967 | +0.0021 |
| legal agent run r94 | 0.6049 | 0.5974 | +0.0028 |
| legal agent run r95 | 0.6058 | 0.5991 | +0.0045 |
| **legal agent run r96** | **0.6056** | **0.5994** | **+0.0048** |
| DCNv2, wide fields, 3-seed bag | 0.6041 | 0.5991 | +0.0045 |
| DeepFM, wide fields, 6-seed bag | 0.6035 | 0.5993 | +0.0047 |
| **hand stack (DeepFM+DCN+GBDT+FM, weights on valid)** | **0.6057** | **0.5991** | **+0.0045** |
| that stack with r94 blended in | 0.6065 | 0.5989 | +0.0043 |
| **stack + r94 + r95 + r96, weights on valid** | **0.6067** | **0.6001** | **+0.0055** |

r96 versus my hand stack is a dead tie: paired bootstrap +0.0002, SE 0.0005, z = 0.4. The agent
reached, unaided, what a 4-family seed-bagged ensemble reaches by hand.

The best number on this page is the last row, and it costs no new training. Pairwise rank
correlation between r94, r95 and r96 is only 0.92-0.94, so independent runs of the same agent
are genuinely diverse. Blending the three that already exist beats the best single run by
+0.0007. Weights are chosen on validation and no component ever read test labels, so this is a
legitimate submission rather than an oracle number.

The single change that mattered was the **field set**, not the architecture: the official
baseline's 5 fields to 12 (adding video_type, upload_type, music_type, tag, hour,
user_active_degree, register_days_bucket), then seed-bagging. Every deep model on the wide
field set lands at 0.598-0.599 test; every model on the 5-field set lands at 0.593.

Note what validation does here. r94 has the **highest validation of any single model** (0.6049)
and is beaten on test by six DeepFM seeds that all score *lower* on validation (0.6026-0.6037).
Blending r94 into the stack raises validation (0.6057 -> 0.6065) and lowers test
(0.5991 -> 0.5989). At this resolution validation actively misleads.

Item-item CF with time decay was built as a fifth family and given weight **0.00** by the
validation search - no diversity value. Its best setting was no decay at all.

## 2. What the metric can even resolve

Bootstrapping over users (40 resamples, `no_hist` predictions):

| split | primary | bootstrap SD |
|---|---|---|
| validation | 0.6008 | **0.0018** |
| test | 0.5957 | **0.0017** |

The entire competitive range — baseline 0.5946 to the best submission 0.5997 — is **3 standard
deviations wide**. A single validation gain below ~0.0036 cannot be distinguished from noise on
validation at all, and the organizers' own convergence rule (ε = 0.002) sits *inside* one SD.

Across 21 configs, validation predicts test well overall (r = 0.98, slope 1.12), but in the band
that matters (valid > 0.598, n = 9) it degrades to r = 0.78 with a residual spread of 0.0014.
Session features are the clean example: **+0.0014 on validation, 0.0000 on test.**

## 3. Why the wall is where it is

`docs/generalisation-ceiling.md` attributes the wall to train→eval drift. That is only part of
it, and the smaller part. Two controls, both at an identical 136K training rows:

| training set | rows | test primary |
|---|---|---|
| test's own week, user-disjoint 5-fold CV (`limits/ceiling.py`) | 136K | 0.5883 |
| random subsample of the earlier train window (`limits/control.py`) | 136K | 0.5805 ± 0.0019 |
| full train window (`no_hist`) | 1.14M | 0.5957 |

- **Cost of the date boundary: 0.0078.** That is the whole of drift, measured with training-set
  size held fixed.
- **Value of data volume: +0.0152** for 8.4x more rows (≈ +0.017 per decade), which *outweighs*
  the drift cost over this range.

Training inside the test distribution scores *worse* (0.5883) than training on stale data from
two weeks earlier (0.5957), simply because there is 8x less of it. Drift is real but modest;
the binding constraint is that these features carry little within-user signal, and the only
cure — more rows — is not available.

**Implied ceiling: 0.5957 + 0.0078 ≈ 0.6035 test primary**, i.e. a model that perfectly
neutralised drift on the full training set. The best submission is at 0.5997. Remaining
headroom is roughly **+0.004**, and claiming it requires solving drift completely.

## 4. Levers measured and rejected

| lever | result |
|---|---|
| refit on train+valid before predicting test | 0.5939 vs 0.5934 — nothing (seed band is ±0.0006) |
| train-only item/author history rates | **−0.012**. Stale train-window popularity does not transfer |
| recency weighting over the 13 train days | −0.015 |
| high-capacity GBDT (255 leaves) | −0.033 |
| pairwise (BPR) loss in place of pointwise | −0.004, and it peaks after one epoch |
| user x attribute affinity | +0.002 |
| session context (position in stream, inter-arrival gap) | +0.0014 valid, 0.0000 test |
| randomized-exposure log | not run — see `research/random-log-verdict.md` |

The refit result is the surprising one. Adding the 125K validation rows — the week immediately
before test — changes nothing, which is independent confirmation that recency is not the lever
it looks like.

## 5. The one free gain

Averaging 3 seeds of the same FM: 0.5934 → 0.5949 on test, +0.0015 for no modelling work.
That is comparable to every feature-engineering idea in the table above combined, and the agent
does not currently do it.
