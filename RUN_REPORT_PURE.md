# Run report - r96

## Loop stages executed by the agent

- **Inspect data (EDA):** completed at iteration #0 - the agent wrote and ran its own exploratory script; its findings are in `eda_report.txt` and were carried into every later prompt.
- **Reproduce official baseline (Requirement 1):** 1 attempt(s); the agent's own pipeline reached validation primary **0.6020** against the official 0.6016 (delta +0.0004, inside the baseline's 5-seed noise). That script became the root of the search tree.
- **Iterate:** 12 experiments proposed, executed and evaluated by the agent, each branching from a node of its own search tree.

## Iteration log

A turn is one hypothesis-to-score cycle and advances several lineages at once, so `turn` and `slot` say when an experiment ran and which lineage produced it. The convergence rule is measured against turns; the 50-cap counts scripts.

| # | turn | slot | phase | parent | status | secs | primary | vs baseline | hypothesis |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | - | eda | - | ok | 28 | n/a | - | In the data-inspection stage, quantify temporal label drift, user spar |
| 1 | 0 | - | baseline | - | ok | 144 | 0.6020 | +0.0004 | Reproduce the official baseline at the modeling stage using a k=16 Fac |
| 2 | 1 | 0 | improve | #1 | ok | 783 | 0.6042 | +0.0026 | At the model/training stage, compare recency-weighted DeepFM, recency- |
| 3 | 1 | 1 | improve | #1 | kept | 826 | 0.6047 | +0.0031 | At the model and objective stages, compare a rich-feature DCN, a withi |
| 4 | 1 | 2 | improve | #1 | kept | 1042 | 0.6050 | +0.0034 | At the model and drift-robust training stages, compare recency-weighte |
| 5 | 2 | 0 | improve | #4 | kept | 161 | 0.6056 | +0.0040 | At the prediction-forming and objective stages, compare a chronologica |
| 6 | 2 | 1 | improve | #4 | kept | 74 | 0.6056 | +0.0040 | At the objective and prediction-forming stages, compare a recency-weig |
| 7 | 2 | 2 | improve | #4 | kept | 23 | 0.6056 | +0.0040 | At the prediction-forming and drift-handling stages, compare temporall |
| 8 | 3 | 0 | improve | #5 | kept | 309 | 0.6056 | +0.0040 | At the prediction-forming and drift-robust training stages, compare ex |
| 9 | 3 | 1 | improve | #5 | kept | 353 | 0.6056 | +0.0040 | At the feature and prediction-forming stages, leakage-safe date-cross- |
| 10 | 3 | 2 | improve | #5 | kept | 25 | 0.6056 | +0.0040 | At the drift-aware feature and prediction-forming stages, compare time |
| 11 | 4 | 0 | improve | #10 | kept | 340 | 0.6058 | +0.0042 | At the objective and sequence-feature stages, compare recency-weighted |
| 12 | 4 | 1 | improve | #10 | kept | 374 | 0.6058 | +0.0042 | At the model and drift-weighting stages, compare recency-weighted fiel |
| 13 | 4 | 2 | improve | #10 | kept | 35 | 0.6058 | +0.0042 | At the prediction-forming stage, compare graph diffusion over the trai |

## Portfolio

- Lineages advanced per turn: **3**
- Turns: **4**  ·  scripts executed: **14** of 50
- A turn is one hypothesis-to-score cycle -- Figure 1's iteration, and what the convergence rule is measured against. Scripts are reported separately so the run can be checked against the stricter reading of the 50-iteration cap as well.
- Lineages archived: **3**  ·  revived from the archive: **0**

### Did the lineages actually disagree?

Within-user rank correlation between the live slots' validation predictions. This is the acceptance test for the whole design: above ~0.95 the extra slots are three copies of one agent and bought nothing.

| turn | scripts | mean corr | max corr | alert |
|---|---|---|---|---|
| 1 | #2, #3, #4 | 0.9017 | 0.9188 | no |
| 2 | #5, #6, #7 | 0.6509 | 0.7327 | no |
| 3 | #8, #9, #10 | 0.7397 | 0.9111 | no |
| 4 | #11, #12, #13 | 0.5676 | 0.7545 | no |

### Lineages retired and replaced

A slot that stops beating its own best is archived, not stopped -- the run's convergence counter is untouched by it. Refill alternates fresh drafts with a revival, and a revival is chosen for score AND for disagreeing with what is already live.

| turn | slot | archived at | choice | detail |
|---|---|---|---|---|
| 3 | 0 | 0.6056 | fresh | new draft |
| 3 | 1 | 0.6056 | fresh | new draft |
| 3 | 2 | 0.6056 | fresh | new draft |

### Controller blend of the portfolio

Every turn the controller blends the trusted incumbent, the live slots and the archive, choosing members on one half of validation and confirming on the other. A blend that wins the selection fold and loses the confirmation fold is refused: it won the split, not the task.

| turn | pool | accepted | members | fold A | fold B | reason |
|---|---|---|---|---|---|---|
| 1 | 3 | no | - | - | - | no member improved fold A |
| 2 | 3 | no | - | - | - | no member improved fold A |
| 3 | 3 | no | - | - | - | no member improved fold A |
| 4 | 6 | no | - | - | - | no member improved fold A |

## Code diff applied, per iteration

Each iteration edits a parent script. This is that edit, reconstructed from the two full sources in the ledger, truncated to the first 40 lines per iteration.


**#1 -> #2**  (+363 / -100 lines)

```diff
--- iter_1.py
+++ iter_2.py
@@ -5,124 +5,389 @@
 import torch
 import torch.nn as nn
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
 
 START_TIME = time.time()
-SEED = 2024
-
+SEED = 2025
 np.random.seed(SEED)
 torch.manual_seed(SEED)
 torch.set_num_threads(min(16, os.cpu_count() or 1))
 
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-EMBED_DIM = 16
-LEARNING_RATE = 0.001
-BATCH_SIZE = 4096
-EPOCHS = 5
-
-
-def make_matrix(split):
-    offsets = np.cumsum(
-        np.asarray([0] + [FEATURE_CARDINALITIES[f] for f in FIELDS[:-1]],
-                   dtype=np.int64)
-    )
-    columns = [
-        np.asarray(split.X[field], dtype=np.int64) + offsets[j]
-        for j, field in enumerate(FIELDS)
+OUT_DIR = os.environ.get("ITER_OUT")
+SHARED = os.environ.get("SHARED_ARTIFACTS")
+
+FIELDS = [
+    "author_id", "duration_bucket", "fans_user_num_range",
... 462 more diff lines
```

**#1 -> #3**  (+454 / -103 lines)

```diff
--- iter_1.py
+++ iter_3.py
@@ -5,4 +5,6 @@
 import torch
 import torch.nn as nn
+import scipy.sparse as sp
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -10,103 +12,414 @@
 
 
-START_TIME = time.time()
-SEED = 2024
-
+START = time.time()
+SEED = 2025
 np.random.seed(SEED)
 torch.manual_seed(SEED)
 torch.set_num_threads(min(16, os.cpu_count() or 1))
 
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-EMBED_DIM = 16
-LEARNING_RATE = 0.001
-BATCH_SIZE = 4096
-EPOCHS = 5
-
-
-def make_matrix(split):
-    offsets = np.cumsum(
-        np.asarray([0] + [FEATURE_CARDINALITIES[f] for f in FIELDS[:-1]],
-                   dtype=np.int64)
+FIELDS = list(FEATURE_CARDINALITIES.keys())
+NUM_FIELDS = [
+    "duration_ms",
+    "user_fans_user_num",
+    "user_follow_user_num",
+    "user_friend_user_num",
+    "user_register_days",
+]
... 548 more diff lines
```

**#1 -> #4**  (+387 / -79 lines)

```diff
--- iter_1.py
+++ iter_4.py
@@ -11,58 +11,277 @@
 
 START_TIME = time.time()
-SEED = 2024
-
+SEED = 2025
 np.random.seed(SEED)
 torch.manual_seed(SEED)
 torch.set_num_threads(min(16, os.cpu_count() or 1))
 
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-EMBED_DIM = 16
-LEARNING_RATE = 0.001
+FIELDS = [
+    "user_id",
+    "video_id",
+    "author_id",
+    "tab",
+    "duration_bucket",
+    "hour",
+    "tag",
+    "upload_type",
+    "music_type",
+    "user_active_degree",
+    "is_live_streamer",
+    "is_video_author",
+    "follow_user_num_range",
+    "fans_user_num_range",
+    "friend_user_num_range",
+    "register_days_range",
+    "onehot_feat3",
+    "onehot_feat7",
+    "onehot_feat8",
+    "video_type",
+]
+
+EMBED_DIM = 8
 BATCH_SIZE = 4096
... 472 more diff lines
```

**#4 -> #5**  (+413 / -371 lines)

```diff
--- iter_4.py
+++ iter_5.py
@@ -2,14 +2,17 @@
 import time
 import json
+import gc
 import numpy as np
 import torch
 import torch.nn as nn
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
 
-START_TIME = time.time()
-SEED = 2025
+START = time.time()
+SEED = 7319
 np.random.seed(SEED)
 torch.manual_seed(SEED)
@@ -17,55 +20,20 @@
 
 FIELDS = [
-    "user_id",
-    "video_id",
-    "author_id",
-    "tab",
-    "duration_bucket",
-    "hour",
-    "tag",
-    "upload_type",
-    "music_type",
-    "user_active_degree",
-    "is_live_streamer",
-    "is_video_author",
-    "follow_user_num_range",
-    "fans_user_num_range",
-    "friend_user_num_range",
... 819 more diff lines
```

**#4 -> #6**  (+433 / -305 lines)

```diff
--- iter_4.py
+++ iter_6.py
@@ -5,11 +5,14 @@
 import torch
 import torch.nn as nn
+from scipy import sparse
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
 
-START_TIME = time.time()
-SEED = 2025
+START = time.time()
+SEED = 7319
 np.random.seed(SEED)
 torch.manual_seed(SEED)
@@ -25,47 +28,39 @@
     "tag",
     "upload_type",
-    "music_type",
     "user_active_degree",
     "is_live_streamer",
     "is_video_author",
+    "fans_user_num_range",
     "follow_user_num_range",
-    "fans_user_num_range",
-    "friend_user_num_range",
     "register_days_range",
-    "onehot_feat3",
-    "onehot_feat7",
-    "onehot_feat8",
-    "video_type",
 ]
-
-EMBED_DIM = 8
-BATCH_SIZE = 4096
-PRED_BATCH_SIZE = 16384
... 813 more diff lines
```

**#4 -> #7**  (+401 / -386 lines)

```diff
--- iter_4.py
+++ iter_7.py
@@ -3,6 +3,6 @@
 import json
 import numpy as np
-import torch
-import torch.nn as nn
+from scipy import sparse
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -10,62 +10,47 @@
 
 
-START_TIME = time.time()
-SEED = 2025
-np.random.seed(SEED)
-torch.manual_seed(SEED)
-torch.set_num_threads(min(16, os.cpu_count() or 1))
-
-FIELDS = [
-    "user_id",
+START = time.time()
+np.random.seed(2026)
+
+GLOBAL_FIELDS = [
     "video_id",
     "author_id",
-    "tab",
+    "tag",
     "duration_bucket",
-    "hour",
-    "tag",
+    "onehot_feat3",
+    "onehot_feat8",
     "upload_type",
     "music_type",
-    "user_active_degree",
-    "is_live_streamer",
-    "is_video_author",
... 808 more diff lines
```

**#5 -> #8**  (+427 / -346 lines)

```diff
--- iter_5.py
+++ iter_8.py
@@ -6,5 +6,6 @@
 import torch
 import torch.nn as nn
-import lightgbm as lgb
+import scipy.sparse as sp
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -14,5 +15,5 @@
 
 START = time.time()
-SEED = 7319
+SEED = 19427
 np.random.seed(SEED)
 torch.manual_seed(SEED)
@@ -30,10 +31,9 @@
     "user_friend_user_num", "user_register_days",
 ]
-SEQ_LEN = 12
 EMBED_DIM = 8
-BATCH_SIZE = 4096
+TRAIN_BATCH = 4096
 PRED_BATCH = 16384
-DIN_EPOCHS = 2
-DIN_HALF_LIFE = 5.0
+EPOCHS = 2
+HALF_LIFE = 4.0
 
 
@@ -43,30 +43,30 @@
     n = len(scores)
     order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
-    su = user_ids[order]
-
-    starts = np.empty(n, dtype=bool)
-    starts[0] = True
-    starts[1:] = su[1:] != su[:-1]
-    start_pos = np.maximum.accumulate(
... 834 more diff lines
```

**#5 -> #9**  (+480 / -341 lines)

```diff
--- iter_5.py
+++ iter_9.py
@@ -9,20 +9,23 @@
 
 from pipeline.data import load, FEATURE_CARDINALITIES
-from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
 
 START = time.time()
-SEED = 7319
+SEED = 18437
 np.random.seed(SEED)
 torch.manual_seed(SEED)
 torch.set_num_threads(min(16, os.cpu_count() or 1))
 
-FIELDS = [
+CAT_FIELDS = [
     "user_id", "video_id", "author_id", "tab", "duration_bucket",
     "hour", "tag", "upload_type", "music_type", "user_active_degree",
-    "is_live_streamer", "is_video_author", "follow_user_num_range",
-    "fans_user_num_range", "friend_user_num_range", "register_days_range",
-    "onehot_feat3", "onehot_feat7", "onehot_feat8", "video_type",
+    "is_live_streamer", "is_video_author", "onehot_feat3",
+    "onehot_feat8", "video_type",
+]
+PNN_FIELDS = [
+    "user_id", "video_id", "author_id", "tab", "duration_bucket",
+    "hour", "tag", "upload_type", "music_type", "user_active_degree",
+    "onehot_feat3", "onehot_feat8",
 ]
 NUM_FIELDS = [
@@ -30,10 +33,18 @@
     "user_friend_user_num", "user_register_days",
 ]
-SEQ_LEN = 12
+N_FOLDS = 4
+SMOOTH = 12.0
+HALF_LIFE = 4.0
 EMBED_DIM = 8
... 890 more diff lines
```

**#5 -> #10**  (+411 / -404 lines)

```diff
--- iter_5.py
+++ iter_10.py
@@ -4,36 +4,34 @@
 import gc
 import numpy as np
-import torch
-import torch.nn as nn
-import lightgbm as lgb
+from scipy import sparse
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
-from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
 
 START = time.time()
-SEED = 7319
+SEED = 19427
 np.random.seed(SEED)
-torch.manual_seed(SEED)
-torch.set_num_threads(min(16, os.cpu_count() or 1))
-
-FIELDS = [
-    "user_id", "video_id", "author_id", "tab", "duration_bucket",
-    "hour", "tag", "upload_type", "music_type", "user_active_degree",
-    "is_live_streamer", "is_video_author", "follow_user_num_range",
-    "fans_user_num_range", "friend_user_num_range", "register_days_range",
-    "onehot_feat3", "onehot_feat7", "onehot_feat8", "video_type",
-]
-NUM_FIELDS = [
-    "duration_ms", "user_fans_user_num", "user_follow_user_num",
-    "user_friend_user_num", "user_register_days",
-]
-SEQ_LEN = 12
-EMBED_DIM = 8
-BATCH_SIZE = 4096
-PRED_BATCH = 16384
-DIN_EPOCHS = 2
-DIN_HALF_LIFE = 5.0
... 838 more diff lines
```

**#10 -> #11**  (+337 / -388 lines)

```diff
--- iter_10.py
+++ iter_11.py
@@ -4,33 +4,41 @@
 import gc
 import numpy as np
-from scipy import sparse
-from scipy.sparse.linalg import svds
-
-from pipeline.data import load, FEATURE_CARDINALITIES
+import lightgbm as lgb
+
+from pipeline.data import load
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
 
 START = time.time()
-SEED = 19427
+SEED = 73191
 np.random.seed(SEED)
 
-PAIR_FIELDS = ["video_id", "author_id", "tag", "duration_bucket"]
-PAIR_WEIGHTS = {
-    "video_id": 0.38,
-    "author_id": 0.30,
-    "tag": 0.20,
-    "duration_bucket": 0.12,
-}
-ENTITY_SMOOTH = {
-    "video_id": 18.0,
-    "author_id": 22.0,
-    "tag": 45.0,
-    "duration_bucket": 90.0,
-}
-PAIR_SMOOTH = {
-    "video_id": 3.0,
-    "author_id": 5.0,
-    "tag": 8.0,
-    "duration_bucket": 12.0,
+CAT_FIELDS = [
... 748 more diff lines
```

**#10 -> #12**  (+416 / -373 lines)

```diff
--- iter_10.py
+++ iter_12.py
@@ -4,6 +4,7 @@
 import gc
 import numpy as np
-from scipy import sparse
-from scipy.sparse.linalg import svds
+import torch
+import torch.nn as nn
+import torch.nn.functional as F
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -12,26 +13,34 @@
 
 START = time.time()
-SEED = 19427
+SEED = 27183
 np.random.seed(SEED)
-
-PAIR_FIELDS = ["video_id", "author_id", "tag", "duration_bucket"]
-PAIR_WEIGHTS = {
-    "video_id": 0.38,
-    "author_id": 0.30,
-    "tag": 0.20,
-    "duration_bucket": 0.12,
-}
-ENTITY_SMOOTH = {
-    "video_id": 18.0,
-    "author_id": 22.0,
-    "tag": 45.0,
-    "duration_bucket": 90.0,
-}
-PAIR_SMOOTH = {
-    "video_id": 3.0,
-    "author_id": 5.0,
-    "tag": 8.0,
-    "duration_bucket": 12.0,
-}
+torch.manual_seed(SEED)
+torch.set_num_threads(min(16, os.cpu_count() or 8))
... 825 more diff lines
```

**#10 -> #13**  (+506 / -368 lines)

```diff
--- iter_10.py
+++ iter_13.py
@@ -5,5 +5,4 @@
 import numpy as np
 from scipy import sparse
-from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -12,51 +11,58 @@
 
 START = time.time()
-SEED = 19427
+SEED = 27183
 np.random.seed(SEED)
 
-PAIR_FIELDS = ["video_id", "author_id", "tag", "duration_bucket"]
-PAIR_WEIGHTS = {
-    "video_id": 0.38,
-    "author_id": 0.30,
-    "tag": 0.20,
-    "duration_bucket": 0.12,
-}
-ENTITY_SMOOTH = {
-    "video_id": 18.0,
-    "author_id": 22.0,
-    "tag": 45.0,
-    "duration_bucket": 90.0,
-}
-PAIR_SMOOTH = {
-    "video_id": 3.0,
-    "author_id": 5.0,
-    "tag": 8.0,
-    "duration_bucket": 12.0,
-}
-
 
 def rank_percentile(user_ids, scores):
+    """Tie-aware percentile ranks independently within each user."""
     user_ids = np.asarray(user_ids, dtype=np.int64)
     scores = np.asarray(scores, dtype=np.float64)
... 915 more diff lines
```

## Alternatives compared inside iterations

316 candidate solutions were built and scored across 12 iteration(s). The convergence rule charges one iteration per experiment, so searching inside an iteration buys comparisons the iteration budget cannot.

| iteration | candidates (validation primary) |
|---|---|
| #2 | deepfm_recency 0.6032, deepfm_recency_incumbent_blend 0.6041, empirical_bayes_history 0.5802, empirical_bayes_history_incumbent_blend 0.6020, lightgbm_recency_history 0.5784, lightgbm_recency_history_incumbent_blend 0.6021 |
| #3 | dcn 0.6045, deep_pair_wide 0.6027, incumbent 0.6020, incumbent+dcn@0.10 0.6027, incumbent+dcn@0.20 0.6030, incumbent+dcn@0.30 0.6033, incumbent+dcn@0.40 0.6037, incumbent+dcn@0.50 0.6040 |
| #4 | autoint_blend_0.25 0.6025, autoint_blend_0.50 0.6034, autoint_blend_0.75 0.6034, autoint_standalone 0.6030, mmoe_blend_0.25 0.6025, mmoe_blend_0.50 0.6037, mmoe_blend_0.75 0.6043, mmoe_standalone 0.6041 |
| #5 | din_blend_0.20 0.6051, din_blend_0.40 0.6056, din_blend_0.60 0.6049, din_blend_0.80 0.6050, din_standalone 0.6047, empirical_bayes_blend_0.20 0.6048, empirical_bayes_blend_0.40 0.6000, empirical_bayes_blend_0.60 0.5912 |
| #6 | empirical_bayes 0.5801, empirical_bayes_incblend_0.25 0.6040, empirical_bayes_incblend_0.50 0.5961, empirical_bayes_incblend_0.75 0.5848, latent_svd 0.5193, latent_svd_incblend_0.25 0.6016, latent_svd_incblend_0.50 0.5755, latent_svd_incblend_0.75 0.5360 |
| #7 | affinity_recent3 0.5649, affinity_recent3_incblend_0.10 0.6051, affinity_recent3_incblend_0.20 0.6043, affinity_recent3_incblend_0.35 0.6003, affinity_recent3_incblend_0.50 0.5925, affinity_recent7 0.5688, affinity_recent7_incblend_0.10 0.6051, affinity_recent7_incblend_0.20 0.6044 |
| #8 | fibinet_blend_0.15 0.6056, fibinet_blend_0.30 0.6055, fibinet_blend_0.50 0.6053, fibinet_blend_0.70 0.6043, fibinet_standalone 0.6039, incumbent_reference 0.6056, signed_svd_blend_0.15 0.6050, signed_svd_blend_0.30 0.6004 |
| #9 | crossfit_binary_lgb_blend_0.20 0.6053, crossfit_binary_lgb_blend_0.40 0.6037, crossfit_binary_lgb_blend_0.60 0.5986, crossfit_binary_lgb_blend_0.80 0.5940, crossfit_binary_lgb_standalone 0.5925, crossfit_family_ensemble_blend_0.20 0.6055, crossfit_family_ensemble_blend_0.40 0.6050, crossfit_family_ensemble_blend_0.60 0.6023 |
| #10 | entity_pair_svd_incblend_0.10 0.6056, entity_pair_svd_incblend_0.20 0.6054, entity_pair_svd_incblend_0.35 0.6032, entity_pair_svd_incblend_0.50 0.5982, entity_pair_svd_standalone 0.5772, incumbent 0.6056, personalized_plus_residual_incblend_0.10 0.6055, personalized_plus_residual_incblend_0.20 0.6051 |
| #11 | chronological_transition_incblend_0.10 0.6054, chronological_transition_incblend_0.20 0.6044, chronological_transition_incblend_0.35 0.5991, chronological_transition_incblend_0.50 0.5888, chronological_transition_incblend_0.65 0.5776, chronological_transition_standalone 0.5568, incumbent 0.6056, recency_lambdamart_incblend_0.10 0.6056 |
| #12 | ffm_pnn_equal_incblend_0.10 0.6055, ffm_pnn_equal_incblend_0.20 0.6055, ffm_pnn_equal_incblend_0.35 0.6057, ffm_pnn_equal_incblend_0.50 0.6052, ffm_pnn_equal_incblend_0.75 0.6042, ffm_pnn_equal_standalone 0.6018, incumbent 0.6056, pnn_fibinet_equal_incblend_0.10 0.6055 |
| #13 | all_graph_sequence_incblend_0.10 0.6057, all_graph_sequence_incblend_0.20 0.6046, all_graph_sequence_incblend_0.35 0.5981, all_graph_sequence_incblend_0.50 0.5887, all_graph_sequence_incblend_0.70 0.5707, all_graph_sequence_standalone 0.5568, chronological_markov_incblend_0.10 0.6056, chronological_markov_incblend_0.20 0.6052 |

## Resource usage (Feasibility & Practicality)

- **Agent wall-clock to converged result: 0.88 h (53 min)**
- Total LLM tokens: **260,967** (167,466 in / 93,501 out), including the knowledge-revision stage
- Iterations used: **14 of 50** (13 accepted scores, 0 failed, 0 rejected)
- Compute inside generated scripts: **1.25 h (75 min)** on CPU.
- Mean tokens per iteration: 18,640
- Stop reason: `converged`

## Autonomy (Impact & Relevance)

- **Manual interventions: 0.** No human edited code, restarted the loop, chose a hypothesis, or selected a result during the run.
- The agent inspected the data, reproduced the baseline, and chose every subsequent experiment itself. The prompt it starts from contains the task specification, the pipeline API and the output contract - no findings about what works on this dataset.
- Failures recovered by in-loop retry: 0
- Ideas retired after repeated failure or underperformance: 0

## Robustness (Technical Execution)

- **No failures occurred in this run.** The recovery machinery was therefore never exercised here; the evidence that it works is below.

The loop never stalled, crashed, or escalated to a human. Guards in place: retry-with-source on crash, distinct handling for timeouts (which are not bugs to fix), method-keyed retirement so a reworded idea cannot evade the blacklist, process-tree kill so a timed-out script leaves no orphan burning CPU, tolerance of LLM outages and rate limits, node retirement with backtracking so the search cannot grind on one exhausted branch, and a circuit breaker that halts on repeated instant failures (a broken environment rather than broken code).

### Evidence from development runs

Across 71 development runs of this agent, 703 iterations were executed and 199 failed. Every one was handled in-loop; none was escalated to a human. Failure taxonomy:

- `(no output)`: 93
- `IndexError`: 30
- `RuntimeError`: 23
- `TypeError`: 15
- `TIMEOUT`: 8
- `}`: 7
- `ValueError`: 6
- `AttributeError`: 6

Each recovery path, with a concrete instance:

- **Retry with source.** `r11` #1 crashed with `RuntimeError: The size of tensor a (8192) must match the size of tenso`. The traceback *and the failing script* went back to the proposer, which fixed it: #2 scored 0.6001.
- **Timeout, handled as distinct from a bug.** `r11` #8 was killed at the limit after 420s. The feedback says the approach is too slow rather than wrong, because re-running a slow script unchanged just times out again.
- **Idea retirement.** `r2` #16: an idea was retired after repeated failure and never proposed again. Retirement keys on the named method, so restating it in different words does not evade the blacklist.
- **Circuit breaker.** `r7` hit 32 consecutive instant, output-less failures: the interpreter could not spawn children at all. That is a broken machine, not broken code, and grinding on would shred the budget for nothing. The loop now halts with `environment_broken` after five such failures — this incident is what the guard was written for.

## Result

- Best validation primary: **0.6058** (baseline 0.6016, delta +0.0042)
- From iteration #11: At the objective and sequence-feature stages, compare recency-weighted LambdaMART, setwise boosted rank_xendcg, and a non-parametric chronological transition model; ranking losses directly emphasize within-user ordering while transition rates exploit short-term feed context that static identity models miss.
- Selection is on validation only, per the scoring rules; the hidden test set was never used to choose between iterations.
- Submission: written to `runs\r96\submission.csv`

### Results table

| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| validation (best iteration) | 0.6729 | 0.5386 | 0.6058 |
| official baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
| hidden test (this submission) | 0.6668 | 0.5315 | **0.5991** |

**Absolute delta over the official baseline on hidden test: GAUC +0.0058, nDCG@5 +0.0033, mean +0.0045** (primary +0.0045).

Per the scoring formula, delta(m) = score_agent(m) - score_baseline(m), averaged over metrics.

Validation delta was +0.0042 and test delta is +0.0045, a
validation-test gap of 0.0066. Some of the validation gain did not
transfer, which is the expected direction: the iteration was chosen for its validation score, so it
carries the winner's-curse inflation that a held-out split removes.

How these test numbers were produced, and what they are not. The run never read a test label:
`pipeline.data` hides them unless `AGENT_ALLOW_TEST_LABELS=1` is set, the harness does not set it,
and `report_run.py` no longer scores test at all. Iteration #11 was selected on
validation alone, and `submission_best.csv` was fixed before this table was filled in. The figures
above were computed once, after the fact, by scoring that committed CSV with the flag set
explicitly -- a local diagnostic, not an input to any choice the agent or the operator made. FAQ
2.9.3 forbids test labels influencing selection in any way; reporting a completed run's score does
not, but using it to pick between runs would.

Unlike the KuaiRand-1K report, the baseline here is the organizers' published reference -- GAUC
0.6610 / nDCG@5 0.5282 / primary 0.5946, mean of 5 seeds, std 0.0008 -- so these deltas are
measured against an external anchor rather than an internal one.
