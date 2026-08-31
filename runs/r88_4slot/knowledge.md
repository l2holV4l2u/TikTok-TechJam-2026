# What the agent established

- (active) A 16-dimensional Factorization Machine using user_id, video_id, author_id, tab, and duration_bucket, with validation-selected training duration and train+validation refitting, reproduces the official baseline within seed noise (0.6035 versus 0.6016) but provides no measurable improvement. [iters 1]
- (active) Across all tested lineages, now including day-consensus generative Naive Bayes, conditional-utility learning, bagged tree models, and diversity-aware reranking, the best primary score is 0.6051—only 0.0016 above the 0.6035 FM baseline and therefore not a measurable gain. [iters 2,3,4,5,6,7]
Ruled out by evidence (do not revisit without a new mechanism):
- (invalidated) The earlier low-agreement characterization of the live outputs is invalidated: the latest outputs have substantial within-user rank agreement, with mean pairwise correlation 0.762 and pairwise values from 0.690 to 0.842, though they are not identical. [iters 5,6,7]
