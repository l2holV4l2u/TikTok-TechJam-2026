The leakage-free FM successfully reproduces the official baseline, so the implementation and feature pipeline are validated.
Its apparent gain is inside seed noise and is not a real improvement.
Next, run a paired 5-seed linear-only ablation with identical features and splits.
A >0.002 drop would confirm latent user–content interactions matter; otherwise rank-16 FM is dead and performance is mostly linear ID memorization.

---

All-field categorical expansion is dead: extra profile/context/content fields add no usable within-user signal for this FM and likely dilute sparse ID interactions.
The difference is entirely inside seed noise.
Next run the paired 5-seed linear-only ablation on the original 5 fields; it directly tests whether FM interactions contribute beyond linear ID memorization.

---

Positive-weighted within-user BPR is OUT: direct pairwise optimization does not improve the validated five-field FM.
The apparent gain is inside seed noise, with no nDCG benefit.
This suggests the bottleneck is representation, not pointwise-loss mismatch.
Next run the paired 5-seed linear-only ablation on the same fields and splits; it isolates whether FM interactions add signal beyond linear ID memorization.

---

Categorical LightGBM is OUT: conditional tree interactions over all leakage-free fields add no useful within-user ranking signal.
The difference from baseline is inside seed noise.
This reinforces that richer context interactions are not the bottleneck; sparse ID memorization likely dominates.
Next run the paired 5-seed linear-only ablation on the original five fields and identical splits.
It directly tests whether any latent FM interaction signal exists before trying further interaction models.