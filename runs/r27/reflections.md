The five-field FM reproduction is validated; pairwise latent feature interactions capture the baseline signal as expected.
Its gain over the official baseline is inside seed noise, so there is no evidence that checkpointing or this optimizer setup improves performance.
Treat optimizer/checkpoint tuning as a dead end for now.
Next, ablate each of the five fields across matched seeds; this directly identifies which fields and interaction pathways contribute real ranking signal and where added modeling capacity should target.

---

All-field FM expansion is OUT: indiscriminate context and metadata interactions add noise/sparsity rather than useful within-user signal.
The difference from the five-field FM and official baseline is inside seed noise.
Next, run matched-seed leave-one-field-out ablations on the validated five-field FM.
This identifies which fields and interaction pathways carry genuine ranking signal, guiding selective additions instead of another broad feature expansion.

---

`tag`-only augmentation is OUT; its label-rate variation does not translate into useful within-user FM interactions, likely due to sparse/noisy user–tag estimates.
The degradation versus the validated five-field FM is exactly at the noise threshold, so it is not convincingly real.
Stop input-stage tag expansion and run matched-seed leave-one-field-out ablations of the five baseline fields.
This isolates which existing interaction pathways carry reproducible ranking signal and identifies where targeted capacity is justified.

---

Within-user BPR is OUT; objective alignment adds no measurable ranking benefit over pointwise BCE, likely because BCE already preserves the useful ordering signal.
The difference is inside seed noise versus both the official baseline and reproduced FM.
Revert to BCE and run matched-seed leave-one-field-out ablations on the five baseline fields.
This directly identifies reproducible interaction pathways and where targeted model capacity could help.