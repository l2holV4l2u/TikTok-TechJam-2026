# NISE + AliCCP Research Notes (TikTok TechJam 2026, Track 2)

Compiled 2026-08-25. Every claim below is sourced; anything not directly confirmed from a primary source is marked **UNCONFIRMED**.

## 1. What is NISE

**High confidence identification.** NISE = **N**on-click samples **I**mproved **S**emi-sup**E**rvised method for conversion rate prediction.

- **Paper**: "Utilizing Non-click Samples via Semi-supervised Learning for Conversion Rate Prediction"
- **Authors**: Jiahui Huang, Lan Zhang (corresponding), Junhao Wang, Shanyang Jiang (USTC); Dongbo Huang, Cheng Ding, Lan Xu (Tencent)
- **Venue/Year**: RecSys '24, Bari, Italy, Oct 14–18, 2024. Pages 350–359.
- **DOI**: [10.1145/3640457.3688151](https://doi.org/10.1145/3640457.3688151) — [ACM full text](https://dl.acm.org/doi/fullHtml/10.1145/3640457.3688151)
- **DBLP**: [dblp.org/rec/conf/recsys/Huang0WJHDX24](https://dblp.org/rec/conf/recsys/Huang0WJHDX24.html)
- **No arXiv preprint found** — closed-access ACM paper only; a copy is mirrored at [github.com/tangxyw/RecSysPapers](https://github.com/tangxyw/RecSysPapers/blob/main/Multi-Task/Utilizing%20Non-click%20Samples%20via%20Semi-supervised%20Learning%20for%20Conversion%20Rate%20Prediction.pdf) (used to extract the numbers below).
- **Official code repo**: [github.com/Hjh233/NISE](https://github.com/Hjh233/NISE) (stated in the paper abstract). Built on top of the `torch-rechub` library. README describes CLI flags including `--strategy esmm`, `deepfm_esmm`, `dcn_esmm`; supports Ali-CCP and KuaiRand-Pure loaders. No license file found in README.

### Reported CVR AUC on Ali-CCP (Table 2 of the paper, MLP/DeepFM/DCNv2 backbones, embedding dim = 16, 5 runs averaged, single NVIDIA 3090)

| Backbone | Method | CVR AUC | LogLoss | KS |
|---|---|---|---|---|
| MLP | ESMM (baseline) | 0.6287 ± 0.0024 | 0.0113 ± 0.0012 | 0.1874 ± 0.0052 |
| MLP | **NISE** | **0.6498 ± 0.0038** | 0.0024 ± 0.0001 | 0.2137 ± 0.0068 |
| DeepFM | ESMM | 0.6306 ± 0.0031 | 0.0094 ± 0.0009 | 0.1874 ± 0.0060 |
| DeepFM | **NISE** | **0.6487 ± 0.0016** | 0.0023 ± 0.0001 | 0.2153 ± 0.0065 |
| DCNV2 | ESMM | 0.6280 ± 0.0035 | 0.0096 ± 0.0006 | 0.1855 ± 0.0037 |
| DCNV2 | **NISE** | **0.6520 ± 0.0044** | 0.0024 ± 0.0001 | 0.2188 ± 0.0077 |

Other baselines on Ali-CCP/MLP for context: MMoE 0.6216, ESCM²-IPS 0.6411, ESCM²-DR 0.6182, DCMT 0.6407. NISE reports a 1.11% average relative CVR-AUC gain over the best SOTA baseline on Ali-CCP, and a 3.65% gain when NISE is bolted onto other MTL architectures (shared-bottom/MMoE/PLE, see Table 5 of the paper).

### Important gap: no separate CTR AUC is reported

**UNCONFIRMED / gap**: The paper's tables (Table 2, 4, 5, 6) report **only the CVR-task AUC/LogLoss/KS**. NISE explicitly decouples CTR and CVR (Section 3.2 of the paper) and trains an auxiliary CTR head to improve embeddings, but no table in the paper lists a standalone CTR AUC number. If the TechJam grading harness requires beating *both* a CTR AUC and a CVR AUC baseline figure, that CTR AUC number does not come from this paper — it will have to be measured by whoever built the TechJam harness (or the harness authors will supply it separately). Do not assume a CTR AUC figure without checking the TechJam-provided baseline artifact/scoreboard.

Dataset stat quoted directly in the paper (Section 1): "in the Ali-CCP dataset, out of the 84 million exposed samples, only 3.4 million are clicked, making up just 4% of the total samples." Table 3 of the paper: Ali-CCP has 33 features, 84M exposures, 3.4M clicks, 18K conversions.

### Ambiguity check

I found no other model literally named "NISE" in the CTR/CVR recommendation literature. One WebSearch snippet loosely used "NISE" ablation variants (NISE-1/2/3) which are just this same paper's ablation study, not different models. Given the TechJam framing (official baseline reproduced on AliCCP, CTR+CVR AUC, torch-rechub-style ESMM comparison target) matches this paper closely, confidence this is the correct "NISE" is **high**, but the missing CTR-AUC number above is the one loose end worth confirming against whatever baseline artifact TechJam actually distributes.

## 2. AliCCP practicalities

Source page: [tianchi.aliyun.com/dataset/408](https://tianchi.aliyun.com/dataset/408) — **the page is JS-rendered and could not be scraped directly via WebFetch** (returned only header/title). Facts below are triangulated from papers/repos that cite the dataset instead; treat anything not cross-confirmed as UNCONFIRMED.

- **Row counts (confirmed, cited in the NISE paper itself, Table 3)**: 84M exposure rows total; train split ~42M exposures / 1.6M clicks / 9K conversions; test split ~42M exposures / 1.7M clicks / 9.4K conversions (per multiple secondary sources, e.g. search synthesis citing the original ESMM/Ali-CCP description — **UNCONFIRMED** exact train/test split numbers, cross-check against the file you actually download).
- **Users/items (repeated consistently across sources)**: ~400K users, ~4.3M items, ~80M+ user-item interactions.
- **Compressed size**: one search summary cited "~4.68 GB compressed test data" — **UNCONFIRMED**, could not verify against the primary Tianchi page or an official spec sheet. Budget conservatively for 10–20GB+ uncompressed CSV given 84M rows with a wide feature string per row.
- **File format**: raw distribution is `sample_skeleton_{train,test}.csv` (label + feature-id:value pairs) + `common_features_{train,test}.csv` (shared user/context features keyed by a common-feature ID), confirmed from the [torch-rechub preprocessing script](https://github.com/datawhalechina/torch-rechub/blob/main/examples/ranking/data/ali-ccp/preprocess_ali_ccp.py). This is the classic Alimama "feature-index:value" sparse format, not a flat CSV with named columns — you must join `sample_skeleton` rows to `common_features` rows by the common-feature key before you have a usable table.
- **Account/phone requirement**: Tianchi requires an Aliyun/Tianchi account login to download. Tianchi publishes a "[Registration Instruction for International Users](https://tianchi.aliyun.com/forum/post/5171)" post confirming non-Chinese users can register, but the page content itself is JS-rendered and I could not confirm from it whether a Chinese phone number is mandatory or whether email-only signup works. **UNCONFIRMED** — verify this manually before assuming international teams can download without a +86 number.
- **HuggingFace / academic mirror**: **No HuggingFace dataset mirror found.** Searches for `Ali-CCP`/`AliCCP`/`ali_ccp` on huggingface.co returned nothing (only unrelated results). [github.com/lhtlht/ali-ccp](https://github.com/lhtlht/ali-ccp) is a set of TF1/TF2 preprocessing notebooks (`data_join.ipynb`, `data_show.ipynb`, `common_utils.py`), **not a data mirror** — it still expects you to have downloaded the raw Tianchi files. Paperswithcode's Ali-CCP page now 302-redirects into the Hugging Face papers hub and no longer serves the dataset card. Bottom line: **plan on going through Tianchi directly**, no confirmed shortcut exists.
- **Schema (processed/reduced form used by most public reproductions, e.g. torch-rechub)**: 19 sparse categorical field IDs — `101, 121, 122, 124, 125, 126, 127, 128, 129, 205, 206, 207, 210, 216, 508, 509, 702, 853, 301` — plus 4 further fields treated as dense/quantized (`109_14, 110_14, 127_14, 150_14`), for 23 usable columns after preprocessing. Label columns: `click`, `purchase` (i.e. CVR label = conversion, called `purchase` in the raw schema). Source: [torch-rechub preprocess_ali_ccp.py](https://github.com/datawhalechina/torch-rechub/blob/main/examples/ranking/data/ali-ccp/preprocess_ali_ccp.py).
- **Feature-count discrepancy — flagged, not resolved**: the NISE paper's Table 3 says Ali-CCP has "33" features; other secondary write-ups (uncited primary source) claim "109 features." Neither number matches the 23-column reduced schema most repos actually train on. **UNCONFIRMED** which number is "correct" — likely explained by 33 being the paper's post-processed column count and 109 being raw multi-hot slot count (field IDs like 109_14 are literal Alimama field-index codes, not a total-feature tally), but I could not verify this from a primary spec. Do not hardcode either number without checking the file you actually receive.
- **Standard train/test split**: the dataset ships pre-split by Alimama into train/test files (one day's data held out as test per the "week 1–2 train, day-N test" scheme commonly described for this dataset in papers) — **UNCONFIRMED** exact day boundary; all reproductions surveyed (NISE/torch-rechub, ESMM, DCMT, ESCM²) simply consume the provided `_train`/`_test` CSV pair as-is rather than re-splitting.
- **NVIDIA Merlin's loader** ([merlin/datasets/ecommerce/aliccp/dataset.py](https://github.com/NVIDIA-Merlin/models/blob/main/merlin/datasets/ecommerce/aliccp/dataset.py)) documents ~24 features across user (11), item (5), user-item cross (4), context (1: position), and 2 targets (click, conversion) — roughly consistent with the 23-column reduced schema above, and confirms `position` as a context feature and separate `click`/`conversion` binary labels. This loader requires the raw Tianchi files to already be downloaded; it does not fetch or synthesize data itself.

## 3. Existing implementations with AliCCP loader + ESMM

| Library | AliCCP loader | ESMM impl | Notes |
|---|---|---|---|
| **torch-rechub** (used by NISE itself) | [examples/ranking/data/ali-ccp/preprocess_ali_ccp.py](https://github.com/datawhalechina/torch-rechub/blob/main/examples/ranking/data/ali-ccp/preprocess_ali_ccp.py) | [torch_rechub/models/multi_task/](https://github.com/datawhalechina/torch-rechub) (`ESMM` class, importable as `from torch_rechub.models.multi_task import ESMM`) | Run scripts: `examples/ranking/run_ali_ccp_ctr_ranking.py`, `examples/ranking/run_ali_ccp_multi_task.py`. This is the most direct starting point — it's literally the library the NISE authors built their repo on top of. |
| **NISE official repo** | reuses torch-rechub's Ali-CCP loader | `baselines/` dir includes ESMM via `--strategy esmm` | [github.com/Hjh233/NISE](https://github.com/Hjh233/NISE) |
| **RecBole** | Not confirmed present. RecBole's dataset URL registry (`recbole/properties/dataset/url.yaml`) did not show `ali-ccp` in the excerpts fetched; general-purpose 44-dataset library but no direct hit for Ali-CCP in this search. **UNCONFIRMED — recheck directly in-repo before relying on this.** | N/A | [github.com/RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole) |
| **FuxiCTR / BARS (reczoo)** | Not confirmed for Ali-CCP specifically in the excerpts fetched (BARS benchmark examples found were Criteo-based, e.g. `ranking/ctr/FinalNet/FinalNet_criteo_x1/`). **UNCONFIRMED** whether an Ali-CCP config exists in [reczoo/BARS](https://github.com/reczoo/BARS) or [reczoo/FuxiCTR](https://github.com/reczoo/FuxiCTR) — worth a direct repo search for `ali_ccp` / `aliccp` before ruling it out. | FuxiCTR supports ESMM as a model_zoo entry generally (multi-task CTR library) but no Ali-CCP-specific config confirmed. | |
| **DeepCTR-Torch** | Not found — this library ships CTR/multi-task model implementations (`deepctr_torch.models.multitask.esmm`) but no bundled Ali-CCP dataset loader. | [ESMM module docs](https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.multitask.esmm.html); source in [shenweichen/DeepCTR](https://github.com/shenweichen/DeepCTR/blob/master/deepctr/models/multitask/esmm.py) (TF version) / DeepCTR-Torch repo for PyTorch | You'd have to write your own Ali-CCP `DataLoader` around this model class. |
| **EasyRec (Alibaba)** | Config-driven; Alibaba's own framework, so it plausibly documents Ali-CCP examples in Chinese docs, but no specific `ali-ccp` example config file path was confirmed in this search. **UNCONFIRMED.** | [easy_rec/python/model/esmm.py](https://github.com/alibaba/EasyRec/blob/master/easy_rec/python/model/esmm.py) confirmed present. | [github.com/alibaba/EasyRec](https://github.com/alibaba/EasyRec); newer PyTorch rewrite at [github.com/alibaba/TorchEasyRec](https://github.com/alibaba/TorchEasyRec) |
| **torchrec (Meta)** | **Not found.** TorchRec is a low-level sharded-embedding-table library, not a dataset zoo — no Ali-CCP loader or ESMM model located. | Not found. | [github.com/meta-pytorch/torchrec](https://github.com/meta-pytorch/torchrec) — useful only as infra (`EmbeddingBagCollection`, etc.), you'd build everything else yourself. |
| **NVIDIA Merlin Models** | [merlin/datasets/ecommerce/aliccp/dataset.py](https://github.com/NVIDIA-Merlin/models/blob/main/merlin/datasets/ecommerce/aliccp/dataset.py) — loads/converts manually-downloaded raw Ali-CCP into parquet via `get_aliccp()`/`prepare_aliccp()`. | Merlin Models ships general multi-task ranking model builders (not confirmed to have a literal `ESMM` class name). **UNCONFIRMED.** | [nvidia-merlin.github.io/models](https://nvidia-merlin.github.io/models/stable/examples/03-Exploring-different-models.html) |

**Practical recommendation**: `torch-rechub` is the strongest starting point — it's the exact library NISE's own reference implementation depends on, has both the Ali-CCP loader and an ESMM class ready to import, and is small/readable enough to fork.

## 4. Memory feasibility on 6GB VRAM (RTX 4050 Laptop)

Known/confirmed anchors: ~400K users, ~4.3M items (multiple independent sources agree); NISE itself used **embedding_dim = 16**, MLP tower `[160, 80]`, batch size 2048, on a single NVIDIA 3090 (24GB) — i.e. the reference implementation was never validated on a 6GB card, so headroom is not guaranteed by the paper.

### Worked estimate (dominant terms only — user_id + item_id)

| embedding_dim | params (user+item ≈ 4.7M ids) | fp32 size | fp16 size |
|---|---|---|---|
| 16 | 4.7M × 16 ≈ 75.2M | ~301 MB | ~150 MB |
| 32 | 4.7M × 32 ≈ 150.4M | ~602 MB | ~301 MB |

This covers only the two highest-cardinality fields. The remaining ~17–21 categorical fields (shop, brand, category, user behavior history, position, etc.) add more, but their exact cardinalities are **UNCONFIRMED** from any source found in this research pass — they depend on the frequency-filter threshold applied during preprocessing (torch-rechub's script filters features occurring `< 10` times before indexing, which caps runaway vocab growth). A reasonable planning assumption: total embedding table stays in the **0.5–1.5 GB range at dim=16 (fp32)**, roughly **1–3 GB at dim=32 (fp32)**, before accounting for optimizer state (Adam ≈ 2× params extra for momentum/variance) or the AliCCP-scale non-click space that NISE explicitly trains over (84M rows, not just 3.4M clicked rows) — the *forward/backward activation and optimizer memory for iterating over 84M non-click samples per epoch* is likely the bigger practical VRAM/throughput constraint than the embedding table itself, not something a static parameter-count estimate captures. **Verify actual per-field cardinalities directly from your own preprocessed data before trusting this range.**

### Plain answer: does full-data training fit in 6GB?

Likely yes for the *model parameters* at dim=16, plausible but tighter at dim=32, **if** you also account for optimizer state and activation memory, which the table above does not include and which was not benchmarked by the original paper (they used a 24GB 3090). This is a qualified **UNCONFIRMED — feasible only with mitigations**, not a clean yes.

### Standard mitigations (if VRAM pressure hits)

- **Feature hashing** — hash high-cardinality fields (item_id, brand, user behavior history) into a fixed-size bucket (e.g. 1M–2M buckets) instead of a full 4.3M-slot table; trades a small amount of collision noise for a hard memory cap.
- **Reduced embedding dim** — drop from 32 → 16 → 8; NISE's own reference config already uses 16, so this is not even a deviation from the paper.
- **CPU-resident embedding + GPU compute** (`torch.nn.EmbeddingBag` on CPU, or `torchrec`'s managed-collision / UVM-style tables) — keeps the large sparse table in host RAM, moves only the looked-up rows to GPU per batch; slower but memory-safe.
- **Mixed precision (AMP/fp16 or bf16)** — roughly halves both parameter and activation memory; directly cuts the embedding-table numbers above in half.
- **Gradient accumulation** — shrink the per-step micro-batch (NISE used batch=2048) and accumulate over several steps to hit an effective batch size without holding all activations in VRAM at once.
- **Negative/non-click sampling** — NISE's own §5.7 ablation shows sampling down the non-click space (as low as 1/24) can *improve* CVR AUC up to a point while cutting compute/memory substantially — this is a paper-sanctioned way to shrink the working set, not just a hack.
