# What the agent established

- (active) A rank-16 categorical Factorization Machine trained with Adam at lr=0.001 on user_id, video_id, author_id, tab, and duration_bucket reproduces the official baseline within seed noise (0.6015 versus 0.6016), so it has no measurable gain. [iters 1]
- (active) Accepted bundled experiments in iterations 2-7 and 9-21 scored 0.6037-0.6057, at least 0.0022 above the implemented FM baseline, but their bundled designs do not identify the causal component. [iters 2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21]
- (active) No accepted experiment has exceeded another by more than 0.002: iterations 17-21 all scored 0.6057, only 0.0002 above the prior 0.6055 best, so no tested pipeline has demonstrated superiority under the gain criterion. [iters 2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21]
- (active) Iteration 8 was rejected and therefore provides no metric evidence about the proposed shared-bottom multitask approach. [iters 8]
Ruled out by evidence (do not revisit without a new mechanism):
- (invalidated) The identity of the distinct-ranking lineage is not stable: slot 0 was isolated in iterations 12-16, whereas in iterations 17-21 slot 3 is isolated (correlations 0.064-0.090) and slots 0-2 are substantially more aligned (0.545-0.684). [iters 12,13,14,15,16,17,18,19,20,21]
