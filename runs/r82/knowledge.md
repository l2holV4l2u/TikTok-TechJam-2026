# What the agent established

- (active) A k=16 pointwise Factorization Machine using user_id, video_id, author_id, tab, and duration_bucket scored 0.6020 versus the 0.6016 official baseline, a difference inside seed noise and thus no measurable improvement. [iters 1]
- (active) A prediction/fusion comparison spanning additive memorization, expanded-field FM, and DeepFM scored 0.6035, within seed noise of the 0.6016 official baseline. [iters 2]
- (active) A prediction-stage comparison spanning additive memorization, expanded FM, DeepFM, and field-aware interactions reached 0.6040, a measurable 0.0024 gain over the 0.6016 official baseline. [iters 4]
- (active) Recency-weighted empirical-Bayes ranking and the PNN/DCN/AutoInt comparison both scored 0.6048 and produced perfectly correlated within-user rankings. [iters 6,7]
- (active) The leakage-free history-representation and temporal-selection comparison scored 0.6048. [iters 9]
- (active) The multi-task architecture comparison, objective/candidate-set prediction comparison, and randomized categorical-tree comparison each scored 0.6049 and produced perfectly correlated within-user rankings. [iters 14,15,16]
- (active) Scores of 0.6048-0.6049 measurably exceed the 0.6016 official baseline, but do not measurably exceed the prior 0.6040 field-aware result because differences of 0.0008-0.0009 are inside seed noise. [iters 4,6,7,9,14,15,16]
- (active) Iterations 11, 12, and 13 failed before producing scores, so they provide no evidence that their proposed methods help or hurt validation ranking. [iters 11,12,13]
