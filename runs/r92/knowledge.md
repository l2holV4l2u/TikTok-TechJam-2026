# What the agent established

- (active) A k=16 Factorization Machine using user_id, video_id, author_id, tab, and duration_bucket, with validation-selected training duration and train+validation refitting, matches the official baseline within seed noise (0.6028 versus 0.6016). [iters 1]
- (active) Expanded pairwise/neural FM, recency-weighted gradient, hierarchical empirical-Bayes prediction, and DIN-style recent-positive attention scored 0.6045-0.6046, with no measurable advantage over the 0.6028 FM anchor under the 0.002 threshold. [iters 2,3,4,5]
- (active) Recency-weighted pointwise boosted trees scored 0.6051, measurably above the 0.6028 FM anchor and 0.6016 official baseline, but indistinguishably from the 0.6055 observed best. [iters 5,6,7]
- (active) Leakage-free hierarchical target/drift statistics, exposure-state features, and a recency-weighted wide crossed linear model each scored 0.6054; they are indistinguishable from one another and from the 0.6055 observed best. [iters 6,7]
- (active) Recency-weighted user-day LambdaMART, recency-weighted spectral user-entity models, and randomized two-hop user-video/author diffusion each scored 0.6055; none provides a measurable gain over the prior 0.6054 best. [iters 7]
Ruled out by evidence (do not revisit without a new mechanism):
- (invalidated) The earlier claim that slots 0 and 2 rank validation most similarly is invalidated; in the current turn slot 0 and slot 1 are most similar at correlation 0.539, while 0-2 and 1-2 are only 0.138 and 0.150, with mean correlation 0.276. [iters 2,3,4,5,6,7]
