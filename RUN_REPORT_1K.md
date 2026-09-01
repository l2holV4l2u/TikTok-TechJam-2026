# Run report - r97_1k

## Loop stages executed by the agent

- **Inspect data (EDA):** completed at iteration #0 - the agent wrote and ran its own exploratory script; its findings are in `eda_report.txt` and were carried into every later prompt.
- **Reproduce official baseline (Requirement 1):** 1 attempt(s); the agent's own pipeline reached validation primary **0.6171** against the official 0.6421844312876108 (delta -0.0251, inside the baseline's 5-seed noise). That script became the root of the search tree.
- **Iterate:** 30 experiments proposed, executed and evaluated by the agent, each branching from a node of its own search tree.

## Iteration log

A turn is one hypothesis-to-score cycle and advances several lineages at once, so `turn` and `slot` say when an experiment ran and which lineage produced it. The convergence rule is measured against turns; the 50-cap counts scripts.

| # | turn | slot | phase | parent | status | secs | primary | vs baseline | hypothesis |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | - | eda | - | ok | 6 | n/a | - | INSPECT DATA stage: quantify drift, coverage, categorical and numeric  |
| 1 | 0 | - | baseline | - | ok | 30 | 0.6171 | -0.0251 | Reproduce the official baseline stage using a k=16 logistic Factorizat |
| 2 | 1 | 0 | improve | #1 | ok | 76 | 0.6547 | +0.0125 | Training-stage breadth plus drift adaptation: compare a temporally wei |
| 3 | 1 | 1 | improve | #1 | kept | 43 | 0.6547 | +0.0125 | I am testing latent collaborative filtering, learned linear item-histo |
| 4 | 1 | 2 | improve | #1 | failed | 0 | FAIL | - | Model-stage breadth: compare field-weighted factorization, explicit pr |
| 5 | 2 | 0 | improve | #2 | failed | 8 | FAIL | - | Model/training-stage breadth: compare a temporally weighted LambdaRank |
| 6 | 2 | 1 | improve | #2 | ok | 96 | 0.6763 | +0.0341 | Multi-task and metric-aligned training stage: MMoE should use click/li |
| 7 | 2 | 2 | improve | #2 | kept | 167 | 0.6767 | +0.0345 | Model-stage breadth: explicit product interactions, deep feature cross |
| 8 | 3 | 0 | improve | #7 | kept | 197 | 0.6767 | +0.0345 | Training/ranking stage: repair LambdaRank by splitting each oversized  |
| 9 | 3 | 1 | improve | #7 | ok | 263 | 0.6790 | +0.0369 | Model/training-stage breadth: compare Wide&Deep, DeepFM, NFM, FiBiNET, |
| 10 | 3 | 2 | improve | #7 | ok | 80 | 0.6824 | +0.0402 | Counterfactual/reranking stage: inverse-exposure empirical-Bayes user– |
| 11 | 4 | 0 | improve | #10 | kept | 490 | 0.6824 | +0.0402 | Rank-aggregation stage: temporally weighted boosted trees, a linear hi |
| 12 | 4 | 1 | improve | #10 | kept | 65 | 0.6828 | +0.0406 | Slate-reranking stage: creator/video deduplication and category-covera |
| 13 | 4 | 2 | improve | #10 | ok | 81 | 0.6944 | +0.0522 | Sequence/context stage: causal exposure-fatigue, session-hazard, and c |
| 14 | 5 | 0 | improve | #13 | kept | 128 | 0.6944 | +0.0522 | Preference-estimation and rank-fusion stage: temporally weighted hiera |
| 15 | 5 | 1 | improve | #13 | kept | 27 | 0.6944 | +0.0522 | Slate-composition stage: applying structurally different soft-MMR, har |
| 16 | 5 | 2 | improve | #13 | failed | 102 | FAIL | - | Sequence-model robustness and interaction stage: replicate the additiv |
| 17 | 6 | 0 | improve | #15 | ok | 298 | 0.7039 | +0.0617 | Training and score-calibration stage: recency-weighted boosted trees,  |
| 18 | 6 | 1 | improve | #15 | kept | 409 | 0.7039 | +0.0617 | Auxiliary-target distributional modeling stage: duration-normalized wa |
| 19 | 6 | 2 | improve | #15 | kept | 148 | 0.7046 | +0.0624 | Sequence/context model stage: fixing out-of-range embedding indices an |
| 20 | 7 | 0 | improve | #19 | kept | 124 | 0.7046 | +0.0624 | Prediction-formation and calibrated-fusion stage: a drift-regularized  |
| 21 | 7 | 1 | improve | #19 | kept | 93 | 0.7046 | +0.0624 | Stationarity-aware prediction stage: selecting categorical signals by  |
| 22 | 7 | 2 | improve | #19 | kept | 176 | 0.7046 | +0.0624 | Cross-boundary user-state feature stage: leakage-safe, prior-day user× |
| 23 | 8 | 0 | improve | #20 | kept | 67 | 0.7046 | +0.0624 | Model-formation breadth stage: compare a generative categorical likeli |
| 24 | 8 | 1 | improve | #20 | kept | 93 | 0.7050 | +0.0628 | Model-formation and drift-weighting stage: compare an explicit hashed- |
| 25 | 8 | 2 | improve | #20 | kept | 120 | 0.7052 | +0.0630 | Model-formation breadth stage: compare a smooth random-Fourier kernel, |
| 26 | 9 | 0 | improve | #25 | kept | 230 | 0.7052 | +0.0630 | I am targeting prediction formation under temporal drift by comparing  |
| 27 | 9 | 1 | improve | #25 | kept | 167 | 0.7052 | +0.0630 | Model-formation breadth stage: compare a neural additive spline model, |
| 28 | 9 | 2 | improve | #25 | kept | 103 | 0.7052 | +0.0630 | Prediction-fusion stage: a fixed, label-free disagreement/confidence g |
| 29 | 10 | 0 | improve | #26 | kept | 424 | 0.7052 | +0.0630 | Training-weighting and prediction-formation stage: train main models u |
| 30 | 10 | 1 | improve | #26 | kept | 394 | 0.7056 | +0.0634 | Pairwise ranking stage: training linear, low-rank quadratic, prototype |
| 31 | 10 | 2 | improve | #26 | kept | 192 | 0.7056 | +0.0634 | Temporal target-statistics stage: entity propensities that extrapolate |

## Portfolio

- Lineages advanced per turn: **3**
- Turns: **10**  ·  scripts executed: **32** of 50
- A turn is one hypothesis-to-score cycle -- Figure 1's iteration, and what the convergence rule is measured against. Scripts are reported separately so the run can be checked against the stricter reading of the 50-iteration cap as well.
- Lineages archived: **3**  ·  revived from the archive: **0**

### Did the lineages actually disagree?

Within-user rank correlation between the live slots' validation predictions. This is the acceptance test for the whole design: above ~0.95 the extra slots are three copies of one agent and bought nothing.

| turn | scripts | mean corr | max corr | alert |
|---|---|---|---|---|
| 1 | #2, #3, #4 | 0.5455 | 0.5455 | no |
| 2 | #5, #6, #7 | 0.8200 | 0.8385 | no |
| 3 | #8, #9, #10 | 0.6971 | 0.7939 | no |
| 4 | #11, #12, #13 | 0.3333 | 0.8705 | no |
| 5 | #14, #15, #16 | 0.0535 | 0.1662 | no |
| 6 | #17, #18, #19 | 0.6326 | 0.7124 | no |
| 7 | #20, #21, #22 | 1.0000 | 1.0000 | **YES** |
| 8 | #23, #24, #25 | 0.3354 | 0.8907 | no |
| 9 | #26, #27, #28 | 0.0034 | 0.3958 | no |
| 10 | #29, #30, #31 | 0.2329 | 0.8193 | no |

### Lineages retired and replaced

A slot that stops beating its own best is archived, not stopped -- the run's convergence counter is untouched by it. Refill alternates fresh drafts with a revival, and a revival is chosen for score AND for disagreeing with what is already live.

| turn | slot | archived at | choice | detail |
|---|---|---|---|---|
| 8 | 0 | 0.7046 | fresh | new draft |
| 8 | 1 | 0.7050 | fresh | new draft |
| 8 | 2 | 0.7052 | fresh | new draft |

### Controller blend of the portfolio

Every turn the controller blends the trusted incumbent, the live slots and the archive, choosing members on one half of validation and confirming on the other. A blend that wins the selection fold and loses the confirmation fold is refused: it won the split, not the task.

| turn | pool | accepted | members | fold A | fold B | reason |
|---|---|---|---|---|---|---|
| 1 | 2 | no | - | - | - | no member improved fold A |
| 2 | 3 | no | slot_1 | +0.0004 | -0.0003 | the blend won the selection fold and lost the confirmation fold |
| 3 | 3 | no | - | - | - | no member improved fold A |
| 4 | 3 | no | - | - | - | no member improved fold A |
| 5 | 3 | no | - | - | - | no member improved fold A |
| 6 | 3 | no | - | - | - | no member improved fold A |
| 7 | 3 | no | - | - | - | no member improved fold A |
| 8 | 3 | no | archive_0 | +0.0001 | -0.0005 | the blend won the selection fold and lost the confirmation fold |
| 9 | 6 | no | archive_0 | +0.0001 | -0.0005 | the blend won the selection fold and lost the confirmation fold |
| 10 | 6 | no | slot_0 | +0.0005 | -0.0004 | the blend won the selection fold and lost the confirmation fold |

## Code diff applied, per iteration

Each iteration edits a parent script. This is that edit, reconstructed from the two full sources in the ledger, truncated to the first 40 lines per iteration.


**#1 -> #2**  (+538 / -116 lines)

```diff
--- iter_1.py
+++ iter_2.py
@@ -1,149 +1,571 @@
 import os
+import gc
+import json
 import time
-import json
-import gc
+import warnings
 import numpy as np
 import torch
 import torch.nn as nn
 import torch.nn.functional as F
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
+warnings.filterwarnings("ignore")
 START = time.time()
-
-SEED = 2024
+SEED = 2025
 np.random.seed(SEED)
 torch.manual_seed(SEED)
 torch.set_num_threads(min(16, os.cpu_count() or 1))
 
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
-OFFSETS = np.cumsum([0] + CARDS[:-1]).astype(np.int64)
-TOTAL_CARDINALITY = int(sum(CARDS))
-K = 16
-BATCH_SIZE = 8192
-PRED_BATCH_SIZE = 131072
-EPOCHS = 8
-LR = 0.001
-
-
... 658 more diff lines
```

**#1 -> #3**  (+473 / -102 lines)

```diff
--- iter_1.py
+++ iter_3.py
@@ -1,6 +1,6 @@
 import os
+import gc
+import json
 import time
-import json
-import gc
 import numpy as np
 import torch
@@ -9,124 +9,458 @@
 
 from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
 START = time.time()
-
-SEED = 2024
+SEED = 2026
 np.random.seed(SEED)
 torch.manual_seed(SEED)
 torch.set_num_threads(min(16, os.cpu_count() or 1))
 
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
-OFFSETS = np.cumsum([0] + CARDS[:-1]).astype(np.int64)
-TOTAL_CARDINALITY = int(sum(CARDS))
-K = 16
-BATCH_SIZE = 8192
-PRED_BATCH_SIZE = 131072
-EPOCHS = 8
-LR = 0.001
-
-
-class FactorizationMachine(nn.Module):
-    def __init__(self, cardinality, rank, initial_bias):
+DEVICE = torch.device("cpu")
+TRAIN_BATCH = 65536
... 588 more diff lines
```

**#1 -> #4**  (+356 / -95 lines)

```diff
--- iter_1.py
+++ iter_4.py
@@ -12,121 +12,357 @@
 
 START = time.time()
-
-SEED = 2024
+SEED = 314159
 np.random.seed(SEED)
 torch.manual_seed(SEED)
 torch.set_num_threads(min(16, os.cpu_count() or 1))
 
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
+FIELDS = [
+    "user_id",
+    "video_id",
+    "author_id",
+    "tab",
+    "tag",
+    "onehot_feat3",
+    "onehot_feat8",
+    "upload_type",
+    "duration_bucket",
+    "user_active_degree",
+]
+NUM_FIELDS = [
+    "duration_ms",
+    "user_follow_user_num",
+    "user_fans_user_num",
+    "user_friend_user_num",
+    "user_register_days",
+]
+
+CARDS = [int(FEATURE_CARDINALITIES[x]) for x in FIELDS]
 OFFSETS = np.cumsum([0] + CARDS[:-1]).astype(np.int64)
-TOTAL_CARDINALITY = int(sum(CARDS))
-K = 16
-BATCH_SIZE = 8192
+TOTAL_CARD = int(sum(CARDS))
... 451 more diff lines
```

**#2 -> #5**  (+516 / -395 lines)

```diff
--- iter_2.py
+++ iter_5.py
@@ -16,5 +16,5 @@
 warnings.filterwarnings("ignore")
 START = time.time()
-SEED = 2025
+SEED = 314159
 np.random.seed(SEED)
 torch.manual_seed(SEED)
@@ -26,7 +26,4 @@
     os.makedirs(OUT, exist_ok=True)
 
-# Low-cardinality fields are passed natively to LightGBM. High-cardinality
-# identities are represented through leakage-safe train histories and
-# temporally weighted leave-one-out target statistics.
 LOW_CAT = [
     "tab", "tag", "duration_bucket", "upload_type", "hour",
@@ -34,12 +31,12 @@
     "onehot_feat8", "user_active_degree", "video_type",
 ]
+MODEL_CAT = [
+    "user_id", "video_id", "author_id", "tab", "tag",
+    "duration_bucket", "upload_type", "onehot_feat3",
+    "onehot_feat8", "music_type",
+]
 TE_FIELDS = [
     "video_id", "author_id", "tag", "tab", "duration_bucket",
     "upload_type", "onehot_feat3", "onehot_feat8",
-]
-DEEP_CAT = [
-    "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "onehot_feat3",
-    "onehot_feat8", "music_type",
 ]
 NUM_FIELDS = [
@@ -47,21 +44,23 @@
     "user_friend_user_num", "user_register_days",
 ]
-
+TE_ALPHA = {
... 1004 more diff lines
```

**#2 -> #6**  (+523 / -449 lines)

```diff
--- iter_2.py
+++ iter_6.py
@@ -8,5 +8,4 @@
 import torch.nn as nn
 import torch.nn.functional as F
-import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -16,5 +15,6 @@
 warnings.filterwarnings("ignore")
 START = time.time()
-SEED = 2025
+SEED = 7319
+rng = np.random.default_rng(SEED)
 np.random.seed(SEED)
 torch.manual_seed(SEED)
@@ -26,279 +26,294 @@
     os.makedirs(OUT, exist_ok=True)
 
-# Low-cardinality fields are passed natively to LightGBM. High-cardinality
-# identities are represented through leakage-safe train histories and
-# temporally weighted leave-one-out target statistics.
-LOW_CAT = [
-    "tab", "tag", "duration_bucket", "upload_type", "hour",
-    "music_type", "onehot_feat1", "onehot_feat3", "onehot_feat7",
-    "onehot_feat8", "user_active_degree", "video_type",
-]
-TE_FIELDS = [
-    "video_id", "author_id", "tag", "tab", "duration_bucket",
-    "upload_type", "onehot_feat3", "onehot_feat8",
-]
-DEEP_CAT = [
-    "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "onehot_feat3",
-    "onehot_feat8", "music_type",
+CAT_FIELDS = [
+    "user_id",
+    "video_id",
+    "author_id",
+    "tab",
... 1040 more diff lines
```

**#2 -> #7**  (+597 / -438 lines)

```diff
--- iter_2.py
+++ iter_7.py
@@ -8,5 +8,4 @@
 import torch.nn as nn
 import torch.nn.functional as F
-import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -15,6 +14,7 @@
 
 warnings.filterwarnings("ignore")
+
 START = time.time()
-SEED = 2025
+SEED = 2026
 np.random.seed(SEED)
 torch.manual_seed(SEED)
@@ -26,37 +26,47 @@
     os.makedirs(OUT, exist_ok=True)
 
-# Low-cardinality fields are passed natively to LightGBM. High-cardinality
-# identities are represented through leakage-safe train histories and
-# temporally weighted leave-one-out target statistics.
-LOW_CAT = [
-    "tab", "tag", "duration_bucket", "upload_type", "hour",
-    "music_type", "onehot_feat1", "onehot_feat3", "onehot_feat7",
-    "onehot_feat8", "user_active_degree", "video_type",
+CAT_FIELDS = [
+    "user_id",
+    "video_id",
+    "author_id",
+    "tab",
+    "tag",
+    "duration_bucket",
+    "upload_type",
+    "onehot_feat3",
+    "onehot_feat8",
+    "music_type",
 ]
+
... 1113 more diff lines
```

**#7 -> #8**  (+345 / -475 lines)

```diff
--- iter_7.py
+++ iter_8.py
@@ -5,7 +5,5 @@
 import warnings
 import numpy as np
-import torch
-import torch.nn as nn
-import torch.nn.functional as F
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -16,8 +14,6 @@
 
 START = time.time()
-SEED = 2026
+SEED = 20260831
 np.random.seed(SEED)
-torch.manual_seed(SEED)
-torch.set_num_threads(min(16, os.cpu_count() or 1))
 
 OUT = os.environ.get("ITER_OUT")
@@ -36,4 +32,7 @@
     "onehot_feat3",
     "onehot_feat8",
+    "user_active_degree",
+    "fans_user_num_range",
+    "register_days_bucket",
     "music_type",
 ]
@@ -47,26 +46,7 @@
 ]
 
-TE_FIELDS = [
-    "video_id",
-    "author_id",
-    "tag",
-    "tab",
-    "duration_bucket",
-    "upload_type",
-]
... 934 more diff lines
```

**#7 -> #9**  (+439 / -311 lines)

```diff
--- iter_7.py
+++ iter_9.py
@@ -16,5 +16,5 @@
 
 START = time.time()
-SEED = 2026
+SEED = 314159
 np.random.seed(SEED)
 torch.manual_seed(SEED)
@@ -67,6 +67,7 @@
 HALF_LIFE = 4.0
 RANK = 8
+EPOCHS = 2
 BATCH_SIZE = 16384
-EPOCHS = 2
+PRED_BATCH = 131072
 
 
@@ -81,6 +82,6 @@
 
 
-def choose_history_keys(d):
-    preferred_terms = (
+def select_history_keys(history):
+    preferred = (
         "long_view_rate",
         "count_log1p",
@@ -90,11 +91,12 @@
     )
     selected = []
-    for term in preferred_terms:
-        matches = [k for k in sorted(d.keys()) if term in k.lower()]
-        if matches:
+    keys = sorted(history.keys())
+    for term in preferred:
+        matches = [k for k in keys if term in k.lower()]
+        if matches and matches[0] not in selected:
             selected.append(matches[0])
-    for k in sorted(d.keys()):
-        if k not in selected:
... 951 more diff lines
```

**#7 -> #10**  (+338 / -572 lines)

```diff
--- iter_7.py
+++ iter_10.py
@@ -5,10 +5,6 @@
 import warnings
 import numpy as np
-import torch
-import torch.nn as nn
-import torch.nn.functional as F
 
 from pipeline.data import load, FEATURE_CARDINALITIES
-from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
@@ -16,9 +12,4 @@
 
 START = time.time()
-SEED = 2026
-np.random.seed(SEED)
-torch.manual_seed(SEED)
-torch.set_num_threads(min(16, os.cpu_count() or 1))
-
 OUT = os.environ.get("ITER_OUT")
 SHARED = os.environ.get("SHARED_ARTIFACTS")
@@ -26,26 +17,5 @@
     os.makedirs(OUT, exist_ok=True)
 
-CAT_FIELDS = [
-    "user_id",
-    "video_id",
-    "author_id",
-    "tab",
-    "tag",
-    "duration_bucket",
-    "upload_type",
-    "onehot_feat3",
-    "onehot_feat8",
-    "music_type",
-]
-
-NUM_FIELDS = [
... 987 more diff lines
```

**#10 -> #11**  (+502 / -359 lines)

```diff
--- iter_10.py
+++ iter_11.py
@@ -5,6 +5,8 @@
 import warnings
 import numpy as np
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
@@ -14,9 +16,10 @@
 OUT = os.environ.get("ITER_OUT")
 SHARED = os.environ.get("SHARED_ARTIFACTS")
+
 if OUT:
     os.makedirs(OUT, exist_ok=True)
 
-FIELDS = [
-    "video_id",
+CAT_FIELDS = [
+    "user_id",
     "author_id",
     "tag",
@@ -26,47 +29,59 @@
     "onehot_feat3",
     "onehot_feat8",
+    "onehot_feat1",
+    "onehot_feat7",
 ]
 
-EXPOSURE_FIELDS = [
+NUM_FIELDS = [
+    "duration_ms",
+    "user_fans_user_num",
+    "user_follow_user_num",
+    "user_friend_user_num",
+    "user_register_days",
+]
+
... 920 more diff lines
```

**#10 -> #12**  (+394 / -415 lines)

```diff
--- iter_10.py
+++ iter_12.py
@@ -5,6 +5,7 @@
 import warnings
 import numpy as np
-
-from pipeline.data import load, FEATURE_CARDINALITIES
+import lightgbm as lgb
+
+from pipeline.data import load
 from pipeline.evaluate import evaluate
 
@@ -14,8 +15,10 @@
 OUT = os.environ.get("ITER_OUT")
 SHARED = os.environ.get("SHARED_ARTIFACTS")
+
 if OUT:
     os.makedirs(OUT, exist_ok=True)
 
-FIELDS = [
+CAT_FIELDS = [
+    "user_id",
     "video_id",
     "author_id",
@@ -26,331 +29,333 @@
     "onehot_feat3",
     "onehot_feat8",
+    "onehot_feat1",
+    "onehot_feat7",
+    "music_type",
+    "user_active_degree",
+    "fans_user_num_range",
+    "follow_user_num_range",
+    "friend_user_num_range",
+    "register_days_bucket",
 ]
 
-EXPOSURE_FIELDS = [
-    "tag",
-    "tab",
... 834 more diff lines
```

**#10 -> #13**  (+406 / -397 lines)

```diff
--- iter_10.py
+++ iter_13.py
@@ -6,5 +6,5 @@
 import numpy as np
 
-from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.data import load
 from pipeline.evaluate import evaluate
 
@@ -14,59 +14,7 @@
 OUT = os.environ.get("ITER_OUT")
 SHARED = os.environ.get("SHARED_ARTIFACTS")
+
 if OUT:
     os.makedirs(OUT, exist_ok=True)
-
-FIELDS = [
-    "video_id",
-    "author_id",
-    "tag",
-    "tab",
-    "duration_bucket",
-    "upload_type",
-    "onehot_feat3",
-    "onehot_feat8",
-]
-
-EXPOSURE_FIELDS = [
-    "tag",
-    "tab",
-    "duration_bucket",
-    "upload_type",
-]
-
-FIELD_WEIGHT = {
-    "video_id": 0.70,
-    "author_id": 1.00,
-    "tag": 0.65,
-    "tab": 0.90,
-    "duration_bucket": 0.35,
... 844 more diff lines
```

**#13 -> #14**  (+356 / -316 lines)

```diff
--- iter_13.py
+++ iter_14.py
@@ -6,5 +6,5 @@
 import numpy as np
 
-from pipeline.data import load
+from pipeline.data import load, FEATURE_CARDINALITIES
 from pipeline.evaluate import evaluate
 
@@ -31,229 +31,47 @@
 
     order = np.lexsort((row, scores, user_ids))
-    su = user_ids[order]
+    sorted_users = user_ids[order]
 
     starts = np.empty(n, dtype=bool)
     starts[0] = True
-    starts[1:] = su[1:] != su[:-1]
+    starts[1:] = sorted_users[1:] != sorted_users[:-1]
 
     start_pos = np.maximum.accumulate(
         np.where(starts, np.arange(n, dtype=np.int64), 0)
     )
-    local = np.arange(n, dtype=np.float32) - start_pos.astype(np.float32)
+    local_rank = (
+        np.arange(n, dtype=np.float32) - start_pos.astype(np.float32)
+    )
 
     ends = np.empty(n, dtype=bool)
     ends[-1] = True
-    ends[:-1] = su[:-1] != su[1:]
+    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
     end_pos = np.flatnonzero(ends)
-    sizes = np.diff(np.r_[-1, end_pos]).astype(np.float32)
-
-    group = np.cumsum(starts, dtype=np.int64) - 1
-    denom = np.maximum(sizes[group] - 1.0, 1.0)
+    group_sizes = np.diff(np.r_[-1, end_pos]).astype(np.float32)
+
+    group_index = np.cumsum(starts, dtype=np.int64) - 1
... 751 more diff lines
```

**#13 -> #15**  (+255 / -399 lines)

```diff
--- iter_13.py
+++ iter_15.py
@@ -19,9 +19,4 @@
 
 
-def safe_logit(p):
-    p = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1.0 - 1e-4)
-    return (np.log(p) - np.log1p(-p)).astype(np.float32)
-
-
 def within_user_rank(user_ids, scores):
     user_ids = np.asarray(user_ids, dtype=np.int64)
@@ -31,349 +26,207 @@
 
     order = np.lexsort((row, scores, user_ids))
-    su = user_ids[order]
+    sorted_users = user_ids[order]
 
     starts = np.empty(n, dtype=bool)
     starts[0] = True
-    starts[1:] = su[1:] != su[:-1]
-
-    start_pos = np.maximum.accumulate(
+    starts[1:] = sorted_users[1:] != sorted_users[:-1]
+
+    start_positions = np.maximum.accumulate(
         np.where(starts, np.arange(n, dtype=np.int64), 0)
     )
-    local = np.arange(n, dtype=np.float32) - start_pos.astype(np.float32)
+    local_positions = (
+        np.arange(n, dtype=np.float32)
+        - start_positions.astype(np.float32)
+    )
 
     ends = np.empty(n, dtype=bool)
     ends[-1] = True
-    ends[:-1] = su[:-1] != su[1:]
-    end_pos = np.flatnonzero(ends)
-    sizes = np.diff(np.r_[-1, end_pos]).astype(np.float32)
-
... 681 more diff lines
```

**#13 -> #16**  (+417 / -274 lines)

```diff
--- iter_13.py
+++ iter_16.py
@@ -5,4 +5,8 @@
 import warnings
 import numpy as np
+import lightgbm as lgb
+import torch
+import torch.nn as nn
+from torch.utils.data import DataLoader, TensorDataset
 
 from pipeline.data import load
@@ -17,4 +21,8 @@
 if OUT:
     os.makedirs(OUT, exist_ok=True)
+
+np.random.seed(2026)
+torch.manual_seed(2026)
+torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
 
 
@@ -31,26 +39,29 @@
 
     order = np.lexsort((row, scores, user_ids))
-    su = user_ids[order]
+    sorted_users = user_ids[order]
 
     starts = np.empty(n, dtype=bool)
     starts[0] = True
-    starts[1:] = su[1:] != su[:-1]
-
-    start_pos = np.maximum.accumulate(
+    starts[1:] = sorted_users[1:] != sorted_users[:-1]
+
+    start_positions = np.maximum.accumulate(
         np.where(starts, np.arange(n, dtype=np.int64), 0)
     )
-    local = np.arange(n, dtype=np.float32) - start_pos.astype(np.float32)
+    local_rank = (
+        np.arange(n, dtype=np.float32) -
+        start_positions.astype(np.float32)
... 831 more diff lines
```

**#15 -> #17**  (+486 / -279 lines)

```diff
--- iter_15.py
+++ iter_17.py
@@ -5,6 +5,8 @@
 import warnings
 import numpy as np
-
-from pipeline.data import load
+import lightgbm as lgb
+
+from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
@@ -18,8 +20,59 @@
     os.makedirs(OUT, exist_ok=True)
 
+if not SHARED:
+    raise RuntimeError("SHARED_ARTIFACTS is required")
+
+INC_VALID_PATH = os.path.join(SHARED, "incumbent_valid_scores.npy")
+INC_TEST_PATH = os.path.join(SHARED, "incumbent_test_scores.npy")
+
+if not os.path.exists(INC_VALID_PATH) or not os.path.exists(INC_TEST_PATH):
+    raise RuntimeError("Trusted incumbent predictions are unavailable")
+
+
+CAT_FIELDS = [
+    "user_id",
+    "tab",
+    "tag",
+    "upload_type",
+    "duration_bucket",
+    "onehot_feat3",
+    "onehot_feat8",
+    "onehot_feat1",
+    "onehot_feat7",
+    "music_type",
+    "user_active_degree",
+    "fans_user_num_range",
+    "follow_user_num_range",
... 786 more diff lines
```

**#15 -> #18**  (+424 / -279 lines)

```diff
--- iter_15.py
+++ iter_18.py
@@ -5,6 +5,8 @@
 import warnings
 import numpy as np
+import lightgbm as lgb
 
 from pipeline.data import load
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
@@ -18,8 +20,17 @@
     os.makedirs(OUT, exist_ok=True)
 
+if not SHARED:
+    raise RuntimeError("SHARED_ARTIFACTS is required")
+
+INC_VALID_PATH = os.path.join(SHARED, "incumbent_valid_scores.npy")
+INC_TEST_PATH = os.path.join(SHARED, "incumbent_test_scores.npy")
+
+if not os.path.exists(INC_VALID_PATH) or not os.path.exists(INC_TEST_PATH):
+    raise RuntimeError("Trusted incumbent predictions are missing")
+
 
 def within_user_rank(user_ids, scores):
     user_ids = np.asarray(user_ids, dtype=np.int64)
-    scores = np.asarray(scores)
+    scores = np.asarray(scores, dtype=np.float64)
     n = len(scores)
     row = np.arange(n, dtype=np.int64)
@@ -32,284 +43,375 @@
     starts[1:] = sorted_users[1:] != sorted_users[:-1]
 
-    start_positions = np.maximum.accumulate(
+    starts_at = np.maximum.accumulate(
         np.where(starts, np.arange(n, dtype=np.int64), 0)
     )
-    local_positions = (
-        np.arange(n, dtype=np.float32)
-        - start_positions.astype(np.float32)
... 728 more diff lines
```

**#15 -> #19**  (+554 / -255 lines)

```diff
--- iter_15.py
+++ iter_19.py
@@ -3,8 +3,11 @@
 import json
 import time
+import math
 import warnings
 import numpy as np
-
-from pipeline.data import load
+import torch
+import torch.nn as nn
+
+from pipeline.data import load, FEATURE_CARDINALITIES
 from pipeline.evaluate import evaluate
 
@@ -17,4 +20,23 @@
 if OUT:
     os.makedirs(OUT, exist_ok=True)
+
+torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
+torch.manual_seed(1729)
+np.random.seed(1729)
+
+FIELDS = [
+    "user_id",
+    "video_id",
+    "author_id",
+    "tag",
+    "tab",
+    "duration_bucket",
+    "upload_type",
+]
+CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
+FIELD_INDEX = {f: i for i, f in enumerate(FIELDS)}
+HISTORY_LENGTH = 4
+BATCH_SIZE = 32768
+EPOCHS = 2
 
 
... 836 more diff lines
```

**#19 -> #20**  (+386 / -465 lines)

```diff
--- iter_19.py
+++ iter_20.py
@@ -3,11 +3,12 @@
 import json
 import time
-import math
 import warnings
 import numpy as np
 import torch
 import torch.nn as nn
+from scipy.special import ndtri
 
 from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
@@ -22,6 +23,6 @@
 
 torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
-torch.manual_seed(1729)
-np.random.seed(1729)
+torch.manual_seed(4103)
+np.random.seed(4103)
 
 FIELDS = [
@@ -33,10 +34,86 @@
     "duration_bucket",
     "upload_type",
+    "onehot_feat3",
+    "onehot_feat8",
+    "user_active_degree",
 ]
-CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
-FIELD_INDEX = {f: i for i, f in enumerate(FIELDS)}
-HISTORY_LENGTH = 4
-BATCH_SIZE = 32768
+CARDS = np.asarray(
+    [int(FEATURE_CARDINALITIES[f]) for f in FIELDS],
+    dtype=np.int64,
+)
... 961 more diff lines
```

**#19 -> #21**  (+435 / -510 lines)

```diff
--- iter_19.py
+++ iter_21.py
@@ -3,11 +3,10 @@
 import json
 import time
-import math
 import warnings
 import numpy as np
-import torch
-import torch.nn as nn
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
@@ -21,31 +20,18 @@
     os.makedirs(OUT, exist_ok=True)
 
-torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
-torch.manual_seed(1729)
-np.random.seed(1729)
-
-FIELDS = [
-    "user_id",
-    "video_id",
-    "author_id",
-    "tag",
-    "tab",
-    "duration_bucket",
-    "upload_type",
-]
-CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
-FIELD_INDEX = {f: i for i, f in enumerate(FIELDS)}
-HISTORY_LENGTH = 4
-BATCH_SIZE = 32768
-EPOCHS = 2
+np.random.seed(314159)
+
+MAX_SELECTED_FIELDS = 12
... 1018 more diff lines
```

**#19 -> #22**  (+536 / -529 lines)

```diff
--- iter_19.py
+++ iter_22.py
@@ -3,11 +3,12 @@
 import json
 import time
-import math
 import warnings
 import numpy as np
 import torch
 import torch.nn as nn
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
@@ -21,9 +22,18 @@
     os.makedirs(OUT, exist_ok=True)
 
+np.random.seed(2718)
+torch.manual_seed(2718)
 torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
-torch.manual_seed(1729)
-np.random.seed(1729)
-
-FIELDS = [
+
+CROSS_FIELDS = [
+    "tag",
+    "tab",
+    "duration_bucket",
+    "upload_type",
+    "onehot_feat3",
+    "onehot_feat8",
+]
+
+RAW_CATEGORICALS = [
     "user_id",
     "video_id",
@@ -33,405 +43,356 @@
... 1131 more diff lines
```

**#20 -> #23**  (+468 / -368 lines)

```diff
--- iter_20.py
+++ iter_23.py
@@ -10,5 +10,4 @@
 
 from pipeline.data import load, FEATURE_CARDINALITIES
-from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
@@ -22,7 +21,7 @@
     os.makedirs(OUT, exist_ok=True)
 
+np.random.seed(7319)
+torch.manual_seed(7319)
 torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
-torch.manual_seed(4103)
-np.random.seed(4103)
 
 FIELDS = [
@@ -34,91 +33,43 @@
     "duration_bucket",
     "upload_type",
-    "onehot_feat3",
-    "onehot_feat8",
-    "user_active_degree",
 ]
+
 CARDS = np.asarray(
     [int(FEATURE_CARDINALITIES[f]) for f in FIELDS],
     dtype=np.int64,
 )
-OFFSETS = np.r_[0, np.cumsum(CARDS[:-1])].astype(np.int64)
-TOTAL_CARD = int(np.sum(CARDS))
-
-BATCH_SIZE = 16384
-EPOCHS = 2
+
+FIELD_INDEX = {name: i for i, name in enumerate(FIELDS)}
+
+PAIR_SPECS = [
+    ("user_id", "tag"),
... 955 more diff lines
```

**#20 -> #24**  (+535 / -264 lines)

```diff
--- iter_20.py
+++ iter_24.py
@@ -22,7 +22,8 @@
     os.makedirs(OUT, exist_ok=True)
 
-torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
-torch.manual_seed(4103)
-np.random.seed(4103)
+THREADS = max(1, min(12, os.cpu_count() or 1))
+torch.set_num_threads(THREADS)
+torch.manual_seed(7319)
+np.random.seed(7319)
 
 FIELDS = [
@@ -38,4 +39,5 @@
     "user_active_degree",
 ]
+
 CARDS = np.asarray(
     [int(FEATURE_CARDINALITIES[f]) for f in FIELDS],
@@ -45,17 +47,72 @@
 TOTAL_CARD = int(np.sum(CARDS))
 
-BATCH_SIZE = 16384
+FIELD_INDEX = {name: i for i, name in enumerate(FIELDS)}
+
+CROSS_PAIRS = [
+    ("user_id", "tag"),
+    ("user_id", "tab"),
+    ("user_id", "duration_bucket"),
+    ("user_id", "upload_type"),
+    ("user_id", "onehot_feat3"),
+    ("user_id", "onehot_feat8"),
+    ("author_id", "tag"),
+    ("author_id", "tab"),
+    ("video_id", "tab"),
+    ("tag", "duration_bucket"),
+]
+
+HASH_BITS = 21
... 1003 more diff lines
```

**#20 -> #25**  (+538 / -343 lines)

```diff
--- iter_20.py
+++ iter_25.py
@@ -7,4 +7,5 @@
 import torch
 import torch.nn as nn
+import torch.nn.functional as F
 from scipy.special import ndtri
 
@@ -23,11 +24,12 @@
 
 torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
-torch.manual_seed(4103)
-np.random.seed(4103)
-
-FIELDS = [
+torch.manual_seed(8317)
+np.random.seed(8317)
+
+BATCH_SIZE = 32768
+HALF_LIFE_DAYS = 4.0
+
+TE_FIELDS = [
     "user_id",
-    "video_id",
-    "author_id",
     "tag",
     "tab",
@@ -37,72 +39,193 @@
     "onehot_feat8",
     "user_active_degree",
+    "music_type",
 ]
-CARDS = np.asarray(
-    [int(FEATURE_CARDINALITIES[f]) for f in FIELDS],
-    dtype=np.int64,
-)
-OFFSETS = np.r_[0, np.cumsum(CARDS[:-1])].astype(np.int64)
-TOTAL_CARD = int(np.sum(CARDS))
-
-BATCH_SIZE = 16384
... 1025 more diff lines
```

**#25 -> #26**  (+550 / -401 lines)

```diff
--- iter_25.py
+++ iter_26.py
@@ -5,8 +5,5 @@
 import warnings
 import numpy as np
-import torch
-import torch.nn as nn
-import torch.nn.functional as F
-from scipy.special import ndtri
+from scipy.special import ndtri, logsumexp
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -23,10 +20,12 @@
     os.makedirs(OUT, exist_ok=True)
 
-torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
-torch.manual_seed(8317)
-np.random.seed(8317)
-
-BATCH_SIZE = 32768
+np.random.seed(27419)
+
 HALF_LIFE_DAYS = 4.0
+N_HIST_BINS = 24
+GMM_COMPONENTS = 6
+GMM_SAMPLE_PER_CLASS = 220000
+GMM_EM_ITERATIONS = 4
+PRED_BATCH = 131072
 
 TE_FIELDS = [
@@ -96,7 +95,5 @@
     ends[:-1] = sorted_users[:-1] != sorted_users[1:]
 
-    sizes = np.diff(
-        np.r_[-1, np.flatnonzero(ends)]
-    ).astype(np.float64)
+    sizes = np.diff(np.r_[-1, np.flatnonzero(ends)]).astype(np.float64)
     group_index = np.cumsum(starts, dtype=np.int64) - 1
     denom = np.maximum(sizes[group_index] - 1.0, 1.0)
@@ -117,16 +114,14 @@
... 1106 more diff lines
```

**#25 -> #27**  (+270 / -316 lines)

```diff
--- iter_25.py
+++ iter_27.py
@@ -24,6 +24,6 @@
 
 torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
-torch.manual_seed(8317)
-np.random.seed(8317)
+torch.manual_seed(27183)
+np.random.seed(27183)
 
 BATCH_SIZE = 32768
@@ -42,21 +42,5 @@
 ]
 
-RAW_NUMERIC = [
-    "duration_ms",
-    "user_fans_user_num",
-    "user_follow_user_num",
-    "user_friend_user_num",
-    "user_register_days",
-]
-
-HISTORY_SUFFIXES = (
-    "train_count_log1p",
-    "long_view_rate",
-    "is_click_rate",
-    "play_time_ms_logmean",
-    "comment_stay_time_logmean",
-)
-
-PRIOR_STRENGTHS = {
+TE_PRIORS = {
     "user_id": 150.0,
     "tag": 1000.0,
@@ -70,4 +54,20 @@
 }
 
+RAW_NUMERIC = [
+    "duration_ms",
+    "user_fans_user_num",
... 809 more diff lines
```

**#25 -> #28**  (+353 / -553 lines)

```diff
--- iter_25.py
+++ iter_28.py
@@ -5,7 +5,5 @@
 import warnings
 import numpy as np
-import torch
-import torch.nn as nn
-import torch.nn.functional as F
+import lightgbm as lgb
 from scipy.special import ndtri
 
@@ -23,10 +21,5 @@
     os.makedirs(OUT, exist_ok=True)
 
-torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
-torch.manual_seed(8317)
-np.random.seed(8317)
-
-BATCH_SIZE = 32768
-HALF_LIFE_DAYS = 4.0
+np.random.seed(27183)
 
 TE_FIELDS = [
@@ -34,4 +27,6 @@
     "tag",
     "tab",
+    "author_id",
+    "video_id",
     "duration_bucket",
     "upload_type",
@@ -42,4 +37,18 @@
 ]
 
+TE_STRENGTH = {
+    "user_id": 120.0,
+    "tag": 700.0,
+    "tab": 900.0,
+    "author_id": 80.0,
+    "video_id": 45.0,
+    "duration_bucket": 800.0,
... 1026 more diff lines
```

**#26 -> #29**  (+550 / -659 lines)

```diff
--- iter_26.py
+++ iter_29.py
@@ -5,5 +5,6 @@
 import warnings
 import numpy as np
-from scipy.special import ndtri, logsumexp
+import lightgbm as lgb
+from scipy.special import ndtri
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -20,12 +21,6 @@
     os.makedirs(OUT, exist_ok=True)
 
-np.random.seed(27419)
-
-HALF_LIFE_DAYS = 4.0
-N_HIST_BINS = 24
-GMM_COMPONENTS = 6
-GMM_SAMPLE_PER_CLASS = 220000
-GMM_EM_ITERATIONS = 4
-PRED_BATCH = 131072
+SEED = 73129
+rng = np.random.default_rng(SEED)
 
 TE_FIELDS = [
@@ -41,4 +36,19 @@
 ]
 
+RAW_CATEGORICAL = [
+    "user_id",
+    "author_id",
+    "tag",
+    "tab",
+    "duration_bucket",
+    "upload_type",
+    "onehot_feat3",
+    "onehot_feat8",
+    "user_active_degree",
+    "music_type",
+    "fans_user_num_range",
... 1344 more diff lines
```

**#26 -> #30**  (+425 / -633 lines)

```diff
--- iter_26.py
+++ iter_30.py
@@ -5,5 +5,8 @@
 import warnings
 import numpy as np
-from scipy.special import ndtri, logsumexp
+import torch
+import torch.nn as nn
+import torch.nn.functional as F
+from scipy.special import ndtri
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -20,15 +23,17 @@
     os.makedirs(OUT, exist_ok=True)
 
-np.random.seed(27419)
-
-HALF_LIFE_DAYS = 4.0
-N_HIST_BINS = 24
-GMM_COMPONENTS = 6
-GMM_SAMPLE_PER_CLASS = 220000
-GMM_EM_ITERATIONS = 4
-PRED_BATCH = 131072
+torch.set_num_threads(min(12, os.cpu_count() or 8))
+np.random.seed(73129)
+torch.manual_seed(73129)
+
+HISTORY_SUFFIXES = (
+    "train_count_log1p",
+    "long_view_rate",
+    "is_click_rate",
+    "play_time_ms_logmean",
+    "comment_stay_time_logmean",
+)
 
 TE_FIELDS = [
-    "user_id",
     "tag",
     "tab",
@@ -39,5 +44,20 @@
... 1212 more diff lines
```

**#26 -> #31**  (+391 / -758 lines)

```diff
--- iter_26.py
+++ iter_31.py
@@ -5,8 +5,7 @@
 import warnings
 import numpy as np
-from scipy.special import ndtri, logsumexp
+from scipy.special import ndtri
 
 from pipeline.data import load, FEATURE_CARDINALITIES
-from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
@@ -20,52 +19,74 @@
     os.makedirs(OUT, exist_ok=True)
 
-np.random.seed(27419)
-
-HALF_LIFE_DAYS = 4.0
-N_HIST_BINS = 24
-GMM_COMPONENTS = 6
-GMM_SAMPLE_PER_CLASS = 220000
-GMM_EM_ITERATIONS = 4
-PRED_BATCH = 131072
-
-TE_FIELDS = [
+FIELDS = [
     "user_id",
+    "author_id",
+    "video_id",
     "tag",
     "tab",
+    "onehot_feat3",
+    "onehot_feat8",
     "duration_bucket",
     "upload_type",
-    "onehot_feat3",
-    "onehot_feat8",
-    "user_active_degree",
-    "music_type",
 ]
... 1216 more diff lines
```

## Search tree

Each line is an executed script; indentation is the edit it was derived from. A node marked `[retired]` produced three children that failed to improve on it, so the search abandoned that branch and backtracked.

```
#1 0.6171  Reproduce the official baseline stage using a k=16 logistic Factorizat
  #2 0.6547  Training-stage breadth plus drift adaptation: compare a temporally wei
    #6 0.6763  Multi-task and metric-aligned training stage: MMoE should use click/li
    #7 0.6767  Model-stage breadth: explicit product interactions, deep feature cross
      #8 0.6767  Training/ranking stage: repair LambdaRank by splitting each oversized 
      #9 0.6790  Model/training-stage breadth: compare Wide&Deep, DeepFM, NFM, FiBiNET,
      #10 0.6824 [retired]  Counterfactual/reranking stage: inverse-exposure empirical-Bayes user–
        #11 0.6824  Rank-aggregation stage: temporally weighted boosted trees, a linear hi
        #12 0.6828  Slate-reranking stage: creator/video deduplication and category-covera
        #13 0.6944 [retired]  Sequence/context stage: causal exposure-fatigue, session-hazard, and c
          #14 0.6944  Preference-estimation and rank-fusion stage: temporally weighted hiera
          #15 0.6944  Slate-composition stage: applying structurally different soft-MMR, har
            #17 0.7039  Training and score-calibration stage: recency-weighted boosted trees, 
            #18 0.7039  Auxiliary-target distributional modeling stage: duration-normalized wa
            #19 0.7046 [retired]  Sequence/context model stage: fixing out-of-range embedding indices an
              #20 0.7046 [retired]  Prediction-formation and calibrated-fusion stage: a drift-regularized 
                #23 0.7046  Model-formation breadth stage: compare a generative categorical likeli
                #24 0.7050  Model-formation and drift-weighting stage: compare an explicit hashed-
                #25 0.7052 [retired]  Model-formation breadth stage: compare a smooth random-Fourier kernel,
                  #26 0.7052 [retired]  I am targeting prediction formation under temporal drift by comparing 
                    #29 0.7052  Training-weighting and prediction-formation stage: train main models u
                    #30 0.7056  Pairwise ranking stage: training linear, low-rank quadratic, prototype
                    #31 0.7056  Temporal target-statistics stage: entity propensities that extrapolate
                  #27 0.7052  Model-formation breadth stage: compare a neural additive spline model,
                  #28 0.7052  Prediction-fusion stage: a fixed, label-free disagreement/confidence g
              #21 0.7046  Stationarity-aware prediction stage: selecting categorical signals by 
              #22 0.7046  Cross-boundary user-state feature stage: leakage-safe, prior-day user×
  #3 0.6547  I am testing latent collaborative filtering, learned linear item-histo
```

## What the agent established

The agent's belief set, revised after every scored iteration rather than appended to, so later evidence can demote an earlier conclusion instead of piling up beside it. A claim marked `(invalidated)` was contradicted by a later result. Machine-readable form with per-claim evidence in `knowledge.json`.

```
- (qualified) The training-stage lineage improved from 0.6790 to 0.7039, and newer recipes reached 0.7056, but this additional 0.0017 is inside seed noise; independent-seed stability and component contributions of the leading recipes remain unidentified. [iters 2,3,4,5,8,10,11,12]
- (qualified) No scored prediction-level fusion has measurably exceeded the leading 0.7039–0.7056 band: the latest label-free disagreement/confidence gate scored 0.7052, while earlier rank-only fusion did not improve on 0.6944. [iters 2,3,7,8,9,10,11,12]
- (qualified) Within-user ranking similarity is strongly method-dependent: earlier correlations ranged from 0.535 to 0.712 or were identical, iteration 11 included negative correlations, and the latest pairs are -0.044, -0.077, and 0.819 despite scores differing by at most 0.0004. [iters 4,5,6,7,8,9,10,11,12]
- (qualified) MMoE metric-aligned training at 0.6763 and interaction/cross/self-attention modeling at 0.6767 differ by only 0.0004, so neither measurably outperforms the other; both beat the older 0.6547 runs in their single evaluations. [iters 4]
- (qualified) Time-ordered splitting of oversized user queries repaired LambdaRank but scored 0.6767, measurably below the later 0.7039–0.7056 band. [iters 5,6,7,8,9,10,11,12]
- (qualified) Recent recency-weighted, auxiliary-target, repaired sequence/context, model-formation, fusion, pairwise-ranking, a
```

## Alternatives compared inside iterations

1104 candidate solutions were built and scored across 27 iteration(s). The convergence rule charges one iteration per experiment, so searching inside an iteration buys comparisons the iteration budget cannot.

| iteration | candidates (validation primary) |
|---|---|
| #2 | empirical_bayes 0.5762, empirical_bayes_best_blend 0.6208, temporal_deepfm 0.6403, temporal_deepfm_best_blend 0.6547, temporal_lgbm 0.3837, temporal_lgbm_best_blend 0.5974, trusted_incumbent 0.6171 |
| #3 | collaborative_mf 0.5861, collaborative_mf_blend_0.15 0.6200, collaborative_mf_blend_0.25 0.6229, collaborative_mf_blend_0.35 0.6268, collaborative_mf_blend_0.50 0.6316, empirical_bayes 0.5993, empirical_bayes_blend_0.15 0.6182, empirical_bayes_blend_0.25 0.6187 |
| #6 | multitask_mmoe 0.6744, multitask_mmoe_best_blend 0.6763, trusted_incumbent 0.6547, within_user_pairwise_nn 0.6333, within_user_pairwise_nn_best_blend 0.6519 |
| #7 | autoint 0.6270, autoint_best_blend 0.6542, dcnv2 0.6282, dcnv2_best_blend 0.6554, pnn 0.5790, pnn_best_blend 0.6532, trusted_incumbent 0.6547 |
| #8 | binary_hl2 0.5765, binary_hl2_best_blend 0.6700, binary_hl4 0.5933, binary_hl4_best_blend 0.6694, binary_hl8 0.5873, binary_hl8_best_blend 0.6699, empirical_bayes 0.6117, empirical_bayes_best_blend 0.6751 |
| #9 | deepfm 0.6501, deepfm_best_blend 0.6770, fibinet 0.6360, fibinet_best_blend 0.6764, nfm 0.6285, nfm_best_blend 0.6778, ple 0.6537, ple_best_blend 0.6790 |
| #10 | marginal_hl4 0.6105, marginal_hl4_best_blend 0.6715, personal_hl4 0.6304, personal_hl4_best_blend 0.6813, personal_ips4 0.6270, personal_ips4_best_blend 0.6809, temporal_trend 0.6227, temporal_trend_best_blend 0.6815 |
| #11 | best_joint_or_overall 0.6806, gbdt_binary 0.6549, gbdt_binary_best_inc_blend 0.6802, incumbent 0.6806, linear_history 0.5645, linear_history_best_inc_blend 0.6759, marginal_te 0.6113, marginal_te_best_inc_blend 0.6754 |
| #12 | author_mmr 0.6818, creator_quota 0.6809, hierarchical_coverage 0.6808, identity 0.6824, tag_xquad 0.6796, video_dedup 0.6828 |
| #13 | content_markov_best_adjusted 0.6727, content_markov_standalone 0.6030, exposure_fatigue_best_adjusted 0.6766, exposure_fatigue_standalone 0.4632, sequence_mixture_best_adjusted 0.6907, sequence_mixture_standalone 0.6186, session_hazard_best_adjusted 0.6944, session_hazard_standalone 0.5207 |
| #14 | confidence_gated_expert_best_blend 0.6885, confidence_gated_expert_standalone 0.6168, demographic_cohort_best_blend 0.6867, demographic_cohort_standalone 0.5999, global_content_best_blend 0.6885, global_content_standalone 0.6168, hierarchical_additive_best_blend 0.6930, hierarchical_additive_standalone 0.6151 |
| #15 | balanced_submodular_slate 0.6925, content_source_coverage 0.6910, delayed_creator_quota 0.6941, hard_creator_quota_top40 0.6933, hard_video_quota_top40 0.6944, soft_creator_mmr_003 0.6939, soft_creator_mmr_007 0.6934, soft_video_creator_mmr 0.6936 |
| #17 | gbdt_rf_bagboost 0.6680, gbdt_rf_bagboost__global__a0.05 0.6993, gbdt_rf_bagboost__global__a0.10 0.6999, gbdt_rf_bagboost__global__a0.16 0.6988, gbdt_rf_bagboost__global__a0.24 0.6997, gbdt_rf_bagboost__global__a0.34 0.6982, gbdt_rf_bagboost__head_union__a0.05 0.6990, gbdt_rf_bagboost__head_union__a0.10 0.6994 |
| #18 | auxiliary_survival_consensus 0.6419, auxiliary_survival_consensus_incumbent_0.70 0.6968, auxiliary_survival_consensus_incumbent_0.82 0.6976, auxiliary_survival_consensus_incumbent_0.90 0.6972, auxiliary_survival_consensus_incumbent_0.95 0.6952, click_hurdle_esmm 0.6554, click_hurdle_esmm_incumbent_0.70 0.6940, click_hurdle_esmm_incumbent_0.82 0.6962 |
| #19 | din_attention_blend_0.10 0.6925, din_attention_blend_0.25 0.6901, din_attention_blend_0.50 0.6884, din_attention_blend_0.75 0.6848, din_attention_blend_1.00 0.6554, din_attention_standalone 0.6554, transition_mlp_blend_0.10 0.6963, transition_mlp_blend_0.25 0.6949 |
| #20 | additive_wide_copula_blend_0.03 0.7042, additive_wide_copula_blend_0.06 0.7039, additive_wide_copula_blend_0.10 0.7036, additive_wide_copula_blend_0.16 0.7006, additive_wide_copula_blend_0.25 0.6984, additive_wide_rank_blend_0.03 0.6997, additive_wide_rank_blend_0.06 0.6962, additive_wide_rank_blend_0.10 0.6930 |
| #21 | generative_nb_blend_0.03 0.6951, generative_nb_blend_0.06 0.6926, generative_nb_blend_0.10 0.6929, generative_nb_blend_0.15 0.6891, generative_nb_blend_0.25 0.6854, generative_nb_blend_0.40 0.6761, generative_nb_blend_1.00 0.6054, generative_nb_standalone 0.6054 |
| #22 | additive_logistic_state_blend_0.025 0.7030, additive_logistic_state_blend_0.050 0.7027, additive_logistic_state_blend_0.100 0.6972, additive_logistic_state_blend_0.200 0.6925, additive_logistic_state_blend_0.350 0.6872, additive_logistic_state_blend_0.500 0.6810, additive_logistic_state_blend_0.750 0.6694, additive_logistic_state_blend_1.000 0.6358 |
| #23 | categorical_likelihood_ratio_copula_blend_0.02 0.7045, categorical_likelihood_ratio_copula_blend_0.04 0.7041, categorical_likelihood_ratio_copula_blend_0.07 0.7036, categorical_likelihood_ratio_copula_blend_0.10 0.7025, categorical_likelihood_ratio_copula_blend_0.15 0.7010, categorical_likelihood_ratio_copula_blend_0.22 0.6972, categorical_likelihood_ratio_rank_blend_0.02 0.6975, categorical_likelihood_ratio_rank_blend_0.04 0.6961 |
| #24 | factorization_machine_copula_blend_0.02 0.7045, factorization_machine_copula_blend_0.04 0.7044, factorization_machine_copula_blend_0.07 0.7038, factorization_machine_copula_blend_0.10 0.7031, factorization_machine_copula_blend_0.15 0.7022, factorization_machine_copula_blend_0.22 0.7009, factorization_machine_copula_blend_0.32 0.6957, factorization_machine_rank_blend_0.02 0.7050 |
| #25 | interaction_mlp_copula_blend_0.02 0.7044, interaction_mlp_copula_blend_0.04 0.7044, interaction_mlp_copula_blend_0.07 0.7040, interaction_mlp_copula_blend_0.10 0.7045, interaction_mlp_copula_blend_0.15 0.7036, interaction_mlp_copula_blend_0.22 0.7016, interaction_mlp_copula_blend_0.32 0.7001, interaction_mlp_rank_blend_0.02 0.7024 |
| #26 | full_covariance_qda_copula_gamma1.0_alpha0.02 0.7048, full_covariance_qda_copula_gamma1.0_alpha0.04 0.7048, full_covariance_qda_copula_gamma1.0_alpha0.07 0.7048, full_covariance_qda_copula_gamma1.0_alpha0.10 0.7039, full_covariance_qda_copula_gamma1.0_alpha0.15 0.7016, full_covariance_qda_copula_gamma1.0_alpha0.22 0.7013, full_covariance_qda_copula_gamma1.0_alpha0.32 0.6957, full_covariance_qda_copula_gamma2.0_alpha0.02 0.7049 |
| #27 | attentive_tabular_copula_blend_0.01 0.7050, attentive_tabular_copula_blend_0.02 0.7050, attentive_tabular_copula_blend_0.04 0.7045, attentive_tabular_copula_blend_0.07 0.7034, attentive_tabular_copula_blend_0.10 0.7033, attentive_tabular_copula_blend_0.15 0.7025, attentive_tabular_copula_blend_0.22 0.6997, attentive_tabular_copula_blend_0.30 0.6972 |
| #28 | boosted_candidate_standalone 0.3557, candidate_demotions_a0.02 0.7049, candidate_demotions_a0.04 0.7044, candidate_demotions_a0.07 0.7033, candidate_demotions_a0.10 0.7030, candidate_demotions_a0.15 0.7031, candidate_demotions_a0.22 0.7013, candidate_demotions_a0.32 0.6982 |
| #29 | cross_family_consensus 0.4014, cross_family_consensus_blend_copula_g1_a0.03 0.7048, cross_family_consensus_blend_copula_g1_a0.06 0.7045, cross_family_consensus_blend_copula_g1_a0.10 0.7039, cross_family_consensus_blend_copula_g1_a0.15 0.7018, cross_family_consensus_blend_copula_g1_a0.22 0.6998, cross_family_consensus_blend_copula_g1_a0.32 0.6933, cross_family_consensus_blend_copula_g1_a0.45 0.6751 |
| #30 | pairwise_linear_best_incumbent_blend 0.7050, pairwise_linear_standalone 0.6231, pairwise_mlp_seed1_best_incumbent_blend 0.7050, pairwise_mlp_seed1_standalone 0.6487, pairwise_mlp_seed2_best_incumbent_blend 0.7051, pairwise_mlp_seed2_standalone 0.6460, pairwise_mlp_seed_ensemble_best_incumbent_blend 0.7051, pairwise_mlp_seed_ensemble_standalone 0.6466 |
| #31 | change_point_best_blend 0.7049, change_point_standalone 0.6061, linear_extrapolation_best_blend 0.7051, linear_extrapolation_standalone 0.6091, recency_level_best_blend 0.7051, recency_level_standalone 0.6135, temporal_consensus_best_blend 0.7051, temporal_consensus_standalone 0.6109 |

## Resource usage (Feasibility & Practicality)

- **Agent wall-clock to converged result: 1.70 h (102 min)**
- Total LLM tokens: **659,573** (421,721 in / 237,852 out), including the knowledge-revision stage
- Iterations used: **32 of 50** (28 accepted scores, 3 failed, 0 rejected)
- Compute inside generated scripts: **1.36 h (82 min)** on CPU.
- Mean tokens per iteration: 20,612
- Stop reason: `converged`

## Autonomy (Impact & Relevance)

- **Manual interventions: 0.** No human edited code, restarted the loop, chose a hypothesis, or selected a result during the run.
- The agent inspected the data, reproduced the baseline, and chose every subsequent experiment itself. The prompt it starts from contains the task specification, the pipeline API and the output contract - no findings about what works on this dataset.
- Failures recovered by in-loop retry: 3
- Ideas retired after repeated failure or underperformance: 0

## Robustness (Technical Execution)

- SyntaxError: 1
- lightgbm.basic.LightGBMError: 1
- IndexError: 1

The loop never stalled, crashed, or escalated to a human. Guards in place: retry-with-source on crash, distinct handling for timeouts (which are not bugs to fix), method-keyed retirement so a reworded idea cannot evade the blacklist, process-tree kill so a timed-out script leaves no orphan burning CPU, tolerance of LLM outages and rate limits, node retirement with backtracking so the search cannot grind on one exhausted branch, and a circuit breaker that halts on repeated instant failures (a broken environment rather than broken code).

## Result

- Best validation primary: **0.7056** (baseline 0.6421844312876108, delta +0.0634)
- From iteration #30: Pairwise ranking stage: training linear, low-rank quadratic, prototype-distance, and independently seeded nonlinear rankers on within-user impression pairs should suppress irrelevant user-level propensity and learn structurally different decision surfaces whose standalone or rank-blended predictions improve GAUC and top-5 ordering.
- Selection is on validation only, per the scoring rules; the hidden test set was never used to choose between iterations.
- Submission: written to `runs/r97_1k/submission.csv`

### Results table

| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| validation (best iteration) | 0.7042 | 0.7070 | 0.7056 |
| KuaiRand-1K reference (research/baseline_reference.py) (validation) | 0.6725 | 0.6118 | 0.6422 |
| hidden test (this submission) | unscored | unscored | unscored |

Test predictions were written without reading labels; final test metrics are computed once by the organizers.
