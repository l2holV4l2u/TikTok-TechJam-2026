# What the agent established

- (active) A k=16 Factorization Machine trained with Adam at lr=0.001 on user, video, author, tab, and duration-bucket categorical interactions matches the official validation baseline within seed noise (0.6009 versus 0.6016; absolute difference 0.0007). [iters 1]
- (active) Across the tested FM, DeepFM, and boosted-tree configurations, four-day recency weighting yields no measurable improvement over the 0.6016 baseline (best primary 0.6020; difference 0.0004). [iters 2]
- (active) The tested prediction-formation package involving additive wide logits, latent user-video matrix factorization, and hierarchical components yields no measurable improvement over the 0.6016 baseline (primary 0.6023; difference 0.0007). [iters 3]
- (active) On identical categorical inputs, the best tested configuration among DCN, AutoInt, FiBiNET, and multi-task MMoE achieves primary 0.6043, a measurable 0.0027 gain over the 0.6016 baseline. [iters 4]
- (active) Across the tested history, ranking-objective, interaction-network, graph, recency/non-parametric, temporal-context, and drift-robust prediction variants, the best primary is 0.6056: measurably above the 0.6016 baseline but not measurably above 0.6043 (difference 0.0013). [iters 5,6,7,8,9,10,11,12,13]
