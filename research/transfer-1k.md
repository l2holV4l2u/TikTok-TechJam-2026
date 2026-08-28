# What KuaiRand-1K actually tests

Measured directly from the raw logs before running anything, because the answer changes what a
1K score would mean.

| | KuaiRand-Pure (scored task) | KuaiRand-1K (bonus set) |
|---|---|---|
| train rows | 1,141,112 | 5,055,984 |
| test rows | 170,588 | 4,132,081 |
| distinct users | 23,875 (test) | 997 (test) |
| distinct train videos | ~7,600 | 2,119,510 |
| impressions per test user | 7.1 | 4,145 |
| **test impressions on a video never seen in train** | **0.01%** | **84.94%** |

The two releases share a schema, a date range, and a label. They do not share a problem.

## The cold-item wall

Pure restricts the catalogue to roughly 7,600 videos and logs 27K users against it, so a video
id in the test window has almost always been seen thousands of times in training. Item
embeddings and item-level statistics are the dominant learnable signal, and every method that
won iterations in our runs leans on them.

1K logs 1,000 users against the **full** Kuaishou catalogue: 2.1M distinct videos across 5.06M
train impressions, an average of 2.4 impressions per video. 85% of test impressions are on a
video the model has never seen. An item embedding table is not merely less useful there — for
five rows in six it is an untrained row.

This is not a harder version of the same task. It is a cold-start task wearing the same schema.

## Consequence for the transfer claim

A Pure-tuned *solution* transferred to 1K will score badly, and that number would carry no
information: it measures the cold-item rate, not the quality of the search that produced the
solution. Reporting it as "our model generalizes / fails to generalize" would be the wrong
claim from the wrong experiment.

The deliverable of this project is the **agent**, not the model it found. So the transfer
question worth asking is:

> Given the same baseline to reproduce and the same budget, does the agent discover a solution
> that beats *that dataset's own baseline* by a comparable margin?

That is measurable without an official 1K leaderboard, which does not exist:

1. The organizers specify the baseline as a **recipe** (FM, k=16, lr=0.001, 5 fields), not as a
   number. The same recipe runs on 1K unchanged.
2. 1K covers the same calendar window, so the official date split (train ≤ 0421, valid
   0422–0428, test 0429–0508) applies unchanged.
3. The reported quantity is the agent's delta over the baseline it reproduced itself — an
   internally anchored number, not a comparison against a leaderboard we do not have.

Every 1K figure in this repo is our own reproduction under that protocol and is labelled as
such. There is no official KuaiRand-1K baseline score to compare against, and we do not imply
one.

## Budget note

Item-vocabulary caps are the one Pure-specific constant that had to change: Pure's 8,000-video
budget would have sent 99.6% of 1K's catalogue to OOV before the agent saw it. 1K uses 250,000
videos and 150,000 authors (`pipeline/data.py`). The tail still goes to OOV — with 2.4
impressions per video that tail is genuinely unlearnable — but the head is preserved, and the
decision of what to do about the remainder is left to the agent.

## Measured reference

`research/baseline_reference.py` runs the organizers' recipe (FM, k=16, lr=0.001, five fields)
on either variant. On Pure it reproduces the published numbers closely enough to trust it
elsewhere:

| | our run | published | Δ |
|---|---|---|---|
| Pure valid | 0.6022 | 0.6016 | +0.0006 |
| Pure test | 0.5957 | 0.5946 | +0.0011 |

On 1K the same recipe gives **valid 0.6422 / test 0.6355**, and the training curve is itself a
result: validation peaks at **epoch 1** and declines every epoch after. On Pure it peaks at
epoch 6. With 2.4 impressions per video the item table memorises on the first pass and each
further epoch trades generalisation for fit.

`agent/facts.py` measures the rest of the brief's dataset facts, and three of them invert:

| | Pure | 1K |
|---|---|---|
| perfect-ranking ceiling | 0.8645 | 0.9995 |
| test users with zero positives | 27.1% | 0.1% |
| random-scoring primary | 0.4732 | 0.4218 |
| item-popularity primary | 0.5709 | 0.4771 |
| **item-popularity lift over random** | **+0.098** | **+0.055** |

Item popularity is the most dependable signal in the field and it earns barely half as much on
1K — the cold-item wall measured a third way. The ceiling moves the other direction: nearly
every 1K user has a positive somewhere in their 4,145 impressions, so the metric's headroom is
almost the full unit interval rather than Pure's 0.8645.

Absolute scores are therefore **not comparable across the two datasets**, which is why the
protocol above compares deltas within a dataset and never across.

## What the agent is told

The 1K brief is generated from these measured facts, not from the Pure literals — the agent
sees 1K's own row counts, ceiling and rungs. Cross-run memory is disabled for the 1K run
(`--no-memory`): it distils prior runs relative to a baseline score, and Pure's 0.60-scale
results carry no meaning against a 0.6422 anchor.

## Columns deliberately not exposed

Not every unreached column is a missed opportunity. Measured as standalone rankers on the Pure
test split, against random at 0.4732 and item popularity at 0.5709:

| column | primary | GAUC |
|---|---|---|
| video age at impression (from `upload_dt`) | 0.4808 | 0.5065 |
| aspect ratio (`server_width`/`server_height`) | 0.4749 | 0.4958 |
| `visible_status` | 0.4760 | 0.5000 |

All three are at random. `visible_status` scores GAUC exactly 0.5000 because it is effectively
constant. Video age looks promising in principle -- the YouTube DNN paper in the knowledge base
argues for feeding item age explicitly -- but every video in KuaiRand-Pure was uploaded inside a
**three-day** window (2022-04-09 to 04-11) while impressions span thirty days, so age is almost
entirely a restatement of `s.date`, which the agent already has. The residual that varies within
a user is a median 1.25-2.12 days, and it carries nothing.

`is_rand` is 0 on every row of both standard logs, so it distinguishes nothing either.

These are recorded so the question is not reopened. The columns that were worth exposing --
`time_ms` and the 22 in `Split.num` -- earned it on the same measurement.

## A limit on how we screened features

Every feature above was screened the same way: score the validation split by that column alone
and read the primary. That works for **item-level** columns and is meaningless for **user-level**
ones, because the metric ranks within each user.

Measured on KuaiRand-Pure validation:

| score | primary |
|---|---|
| constant everywhere | 0.4837 |
| `user_id` (constant within user) | 0.4837 |
| random, constant within user | 0.4837 |
| random per row | 0.4843 |

A feature that does not vary inside a user's impression list cannot reorder it, so all three
collapse to the same tie-broken ordering. Screening the four raw user counts and the eighteen
`onehot_feat*` columns this way returned 0.4837 for all twenty-two -- not evidence they are
useless, evidence the screen cannot see them. They reach the model only through interactions
with item-side features, which is what the FM is for.

The screen was therefore used only where it is valid: `video_features_statistic` (item-level,
three columns promoted on it) and the video metadata columns (item-level, all rejected on it).

## Every raw column is now accounted for

The data-exposure avenue is exhausted. Each column in the four KuaiRand-Pure files is now either
reachable by the agent or measured and rejected:

| source | status |
|---|---|
| log: `user_id`, `video_id`, `tab`, `duration_ms`, `date`, `time_ms` | exposed |
| log: `hourmin` | hour is in `s.X`; the minute is **constant within every user's impression list** (0% of users vary), so it cannot reorder anything -- GAUC exactly 0.5000 |
| log: `is_rand` | 0 on every row of both standard logs |
| log: post-click columns | `s.aux` only -- they are outcomes of the row being scored |
| `user_features`: ranges + 18 one-hots | `s.X` |
| `user_features`: raw counts, `register_days` | `s.num` (prefixed `user_*`) |
| `video_features_basic`: author, type, upload/music type, tag, duration | `s.X` |
| `video_features_basic`: `upload_dt`, aspect, `visible_status` | measured at random (0.4749-0.4808 against a 0.4732 floor) |
| `video_features_statistic`: all 51 columns | `s.num` |

`hourmin` behaving as a session key rather than a timestamp is consistent with the tied
timestamps `s.time_ms` shows: 57.9% of users have at least two impressions sharing a millisecond,
because a feed page is logged as one batch.

Nothing raw remains to expose. Further data work would have to be derived rather than read --
and everything derivable from these columns, the agent can already compute for itself.
