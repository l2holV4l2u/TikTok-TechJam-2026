# What the agent established

- (active) A k=16 Factorization Machine trained with Adam at lr=0.001 on user, video, author, tab, and duration-bucket categorical fields reproduces the official baseline but provides no measurable improvement: primary 0.6025 versu [iters 1]
- (active) Increasing the Factorization Machine latent rank from k=16 to k=64 under the same validated pipeline does not improve within-user ranking and lowers primary from 0.6025 to 0.6004, so increasing FM rank alone is not a pro [iters 1,2]
- (active) A 19-field DeepFM with embedding dimension 8 and validation-selected model alpha 0.75 does not measurably outperform the k=16 FM: primary 0.6044 versus 0.6025 is a gain of only 0.0019, inside seed noise. [iters 1,3]
- (active) Leakage-free multi-task training that jointly predicts long-view, click, and like from safe categorical features does not improve over the k=16 FM in this implementation: primary falls from 0.6025 to 0.6002, and validati [iters 1,4]
