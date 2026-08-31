# What the agent established

- (active) A rank-16 Factorization Machine trained with Adam at learning rate 0.001 on user_id, video_id, author_id, tab, and duration_bucket reproduces the official baseline within seed noise: 0.6026 versus 0.6016, a difference of 0.0010. [iters 1]
- (active) The first experiments from all four lineages were within seed noise of the reproduced 0.6026 FM baseline: slots 0–2 scored 0.6035 and slot 3 scored 0.6044. [iters 1,2]
- (active) Across iterations 3–5, the tested DIN, RankNet, LightGBM, drift, interaction, gating, neural, non-parametric, and ensemble variants scored 0.6047–0.6048; their mutual differences of at most 0.0001 are not measurable, while they are 0.0021–0.0022 above the reproduced FM baseline. [iters 3,4,5]
- (active) In iteration 5, candidate-set-relative equal-query ListNet, an out-of-fold user-conditional LambdaRank gate, drift-robust neural/recency methods, and structurally distinct non-parametric recommenders all scored 0.6048, producing no measurable gain over the prior best. [iters 5]
- (qualified) The iteration-4 ranking identity between slots 1 and 3 was not stable across proposals: iteration-5 pairwise correlations range from 0.727 to 0.809, although the mean correlation of 0.756 still indicates substantial prediction overlap. [iters 4,5]
- (active) Iteration 5 is the third consecutive turn without a gain above 0.002, so the stated stopping condition is met. [iters 3,4,5]
