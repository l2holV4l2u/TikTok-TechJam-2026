# What the agent established

- (qualified) The k=16 FM at lr=0.001 scored 0.6039 only with the reverted implementation containing prior feature/label leakage; with five categorical fields strictly whitelisted and validation labels reserved for the permitted test refit, it scored 0.6016. [iters 1,2]
- (active) Expanded-field factorization/gradient-boosting fusion (0.6036), drift-aware MF/NFM (0.6038), ranking-loss work (0.6039), AutoInt (0.6043), DIN/GRU sequential encoding (0.6044), structurally varied prediction/drift personalization (0.6044), and hierarchical empirical-Bayes user×content reranking (0.6045) have no measurable primary-metric d [iters 3,4,5,6,7]
- (qualified) Within-user rank agreement is strongly lineage-pair dependent: earlier observed correlations include 0.738, 0.842, and 0.924, whereas the two newest lineages correlate only 0.293 despite indistinguishable primary scores. [iters 3,4,5,6,7]
