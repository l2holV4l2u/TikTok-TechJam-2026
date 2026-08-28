# What the agent established

- (active) A k=16 Factorization Machine trained with Adam at lr=0.001 on user, video, author, tab, and duration-bucket categorical fields reproduces the official validation baseline without a gain above the 0.002 significance threshold. [iters 1,3]
- (active) Adding a nonlinear numeric tower with alpha=1.25 over 18 robustly scaled pre-impression video statistics to the FM produces no measurable validation improvement over either the base FM or the official baseline. [iters 2]
- (active) Blending the incumbent FM with a query-grouped LambdaMART model using numeric-native splits, categorical crosses, and causal feed-position features produces no measurable improvement over the FM: the selected 150-round blend with FM weight 2.0 scores 0.6036 versus 0.6021 for the FM, a difference inside seed noise, and does not exceed the  [iters 3]
