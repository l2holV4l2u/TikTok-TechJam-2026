# What the agent established

- (qualified) The iteration-2 comparison of FM, DeepFM, categorical gradient boosting, and empirical-Bayes variants found no measurable gain over the reproduced FM baseline: its winning primary of 0.6040 was only 0.0018 above 0.6022; this conclusion is limited to those iteration-2 candidates. [iters 1,2,3]
- (active) Within-user rank blending with the incumbent has not produced a measurable gain: iterations 2 and 3 retained raw candidates, iteration 4's alpha=0.5 blend added only 0.0001, and iteration 5 selected the incumbent alone with alpha=0.0. [iters 2,3,4,5]
- (active) The iteration-3 auxiliary-MMoE winner using is_click and is_like auxiliary tasks scored 0.6043, only 0.0003 above the iteration-2 incumbent at 0.6040, so it showed no measurable gain. [iters 2,3]
- (active) In iteration 4, the DCN-based incumbent blend was selected over the tested NFM and within-user BPR-FM alternatives, but its final primary of 0.6053 was only 0.0010 above the iteration-3 incumbent, so none of those tested interaction or pairwise-objective mechanisms demonstrated a measurable gain. [iters 3,4]
- (active) The iteration-5 breadth sweep over AutoInt attention, PNN product interactions, FiBiNET field reweighting, and DCNv2 low-rank matrix crosses produced no measurable gain over the 0.6053 incumbent; validation selected the incumbent with family=None and blend alpha=0.0. [iters 4,5]
- (active) Iterations 3, 4, and 5 each failed to improve the incumbent by more than 0.002, so iteration 5 completes the third consecutive no-gain iteration and triggers the run's stopping rule. [iters 2,3,4,5]
Ruled out by evidence (do not revisit without a new mechanism):
- (invalidated) The k=16 Factorization Machine is a reproducible reference at primary 0.6022 but is not the best observed approach: the iteration-4 DCN-based result reached 0.6053, a measurable 0.0031 increase. [iters 1,3,4,5]
