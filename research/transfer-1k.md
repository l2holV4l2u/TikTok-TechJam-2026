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
