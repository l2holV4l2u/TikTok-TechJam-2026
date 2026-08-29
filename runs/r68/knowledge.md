# What the agent established

- (active) A k=16 categorical Factorization Machine trained with Adam at lr=0.001 on user_id, video_id, author_id, tab, and duration_bucket reproduces the official baseline within seed noise (primary 0.6018 versus 0.6016) but provides no measurable improvement above it. [iters 1]
- (active) Replacing the five-field categorical FM with the tested DeepFM over strong leakage-free categorical fields yields a modest measurable within-user ranking gain: primary 0.6044, improving by 0.0026 over the reproduced FM and 0.0028 over the official baseline. [iters 1,2]
- (active) For the tested DeepFM, harness blending at alpha 0.75 provides no measurable gain over the raw candidate because primary changes only from 0.6040 to 0.6044, which is inside seed noise. [iters 2]
