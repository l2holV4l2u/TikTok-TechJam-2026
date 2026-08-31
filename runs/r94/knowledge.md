# What the agent established

- (active) A k=16 Factorization Machine trained with Adam at lr=0.001 on user_id, video_id, author_id, tab, and duration_bucket, with linear and pairwise latent effects, matches the official baseline within seed noise (0.6014 versus 0.6016) and yields no measurable gain. [iters 1]
- (active) Model-formation/temporal-weighting, non-deep temporal-drift, and pairwise/collaborative temporal-history methods each reached 0.6042, a measurable gain over the 0.6016 baseline, but are mutually tied within seed noise. [iters 2,3,4]
- (active) Behavior-sequence, click/like multi-task, and recency-weighted direct LambdaRank methods each reached 0.6047; their 0.0005 increase over 0.6042 is inside seed noise and is not a measurable incremental gain. [iters 5,6,7]
- (active) Graph representation, recency-weighted binary GBDT, list-context post-processing, holdout-selected temporal weighting, temporal stacked generalization, and decayed item-item co-visitation scored 0.6047-0.6049; none measurably improves on 0.6047, although all remain measurably above the 0.6016 baseline. [iters 8,9,10,11,12,13]
