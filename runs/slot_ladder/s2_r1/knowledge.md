# What the agent established

- (active) A k=16 Factorization Machine trained with Adam at lr=0.001 on user_id, video_id, author_id, tab, and duration_bucket reproduced baseline-level validation performance: primary 0.6013 versus the official 0.6016, a 0.0003 difference within seed noise. [iters 1]
- (active) The slot 0 recency-weighted modeling and training-weighting experiment achieved primary 0.6038, which is 0.0022 above the official 0.6016 baseline. [iters 2]
- (active) The slot 1 experiment applying recency weighting across training, prediction formation, and aggregation achieved primary 0.6050, which is 0.0034 above the official baseline. [iters 3]
- (active) The slot 0 comparison of non-neural history-based prediction features scored primary 0.6050; its 0.0012 increase over slot 0's prior 0.6038 was within seed noise and showed no measurable gain. [iters 2,4]
- (active) The slot 1 comparison of target-conditioned versus target-independent representations and auxiliary training scored primary 0.6052; its 0.0002 increase over the prior 0.6050 was within seed noise and showed no measurable gain. [iters 3,5]
- (active) The interaction-architecture comparison and objective/temporal-representation comparison both scored primary 0.6054; their increases over the preceding 0.6050 and 0.6052 results were within seed noise and showed no measurable gain. [iters 4,5,6,7]
- (active) Recency-weighted pointwise boosting/training-weighting and graph-diffusion comparisons both scored primary 0.6056; they were tied and only 0.0002 above the prior 0.6054 results, so neither produced a measurable gain. [iters 6,7,8,9]
- (active) Results from iterations 2–9 span primary 0.6038 to 0.6056, a range of 0.0018; therefore none of the tested recency, history-feature, interaction-architecture, objective, temporal-representation, boosting, or graph-diffusion lineages is measurably better than another under the 0.002 threshold. [iters 2,3,4,5,6,7,8,9]
