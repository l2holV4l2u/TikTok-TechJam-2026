# What the agent established

- (active) A k=16 Factorization Machine trained with Adam at lr=0.001 on user_id, video_id, author_id, tab, and duration_bucket reproduces the official baseline within seed noise (0.6015 versus 0.6016), but provides no measurable improvement over it. [iters 1]
- (active) Applying inverse-square-root user-frequency weighting to the same Factorization Machine training objective changes nothing measurable: primary=0.6007 versus 0.6015 unweighted, a difference within seed noise, so this weighting configuration does not improve the baseline. [iters 1,2]
- (active) The jointly changed configuration using a DeepFM branch with informative content categories and 32 numeric features including log-scaled pre-impression video statistics measurably improves within-user ranking over the five-field FM and official baseline: primary=0.6053 versus 0.6015 and 0.6016, respectively. Because architecture and featu [iters 1,3]
