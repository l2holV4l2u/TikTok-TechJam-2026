The FM successfully reproduces the baseline, but plain pairwise latent interactions are now ruled out as a meaningful improvement path.  
Its apparent lift is inside seed noise, so there is no real gain.  
Next train LightGBM LambdaRank with user-grouped queries, adding train-only smoothed video/author CTR, user–author affinity, hour/tab, and duration features; nonlinear propensity and context effects should exceed 0.002.

---

LambdaMART as a standalone replacement is dead; sparse aggregates cannot recover the FM’s collaborative signal, and querywise ranking overfits weak contexts.  
The degradation is far beyond seed noise.  
Next, train a leakage-free stacked residual model: cross-fitted FM logits plus smoothed propensity/context features into pointwise LightGBM, blended additively with the FM.  
Preserving latent personalization while learning nonlinear residual corrections offers the best chance of a >0.002 gain.

---

Residual tree stacking is dead: propensity/context features do not correct FM errors meaningfully while preserving its ranking.
The apparent change is entirely inside seed noise.
Next train a 2-layer LightGCN on train-only user–video/author positive edges with exposed negatives, then blend its score with FM logits.
Higher-order collaborative neighborhoods and author sharing can capture nonlinear preference structure absent from pairwise FM, offering the best chance of >0.002.

---

Two-layer heterogeneous LightGCN is OUT; graph propagation adds no useful affinity beyond the FM’s latent interactions.  
The apparent lift is inside seed noise, not a real gain.  
The patience budget is exhausted, so no next experiment can run; retain the FM baseline and stop.