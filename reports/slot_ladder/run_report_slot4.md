# Run report - r88_4slot

> 1 scored iteration(s) were rejected by the integrity critic. They are excluded from the search tree, cross-run memory, and submission.

## Loop stages executed by the agent

- **Inspect data (EDA):** completed at iteration #0 - the agent wrote and ran its own exploratory script; its findings are in `eda_report.txt` and were carried into every later prompt.
- **Reproduce official baseline (Requirement 1):** 1 attempt(s); the agent's own pipeline reached validation primary **0.6035** against the official 0.6016 (delta +0.0019, inside the baseline's 5-seed noise). That script became the root of the search tree.
- **Iterate:** 16 experiments proposed, executed and evaluated by the agent, each branching from a node of its own search tree.

## Iteration log

A turn is one hypothesis-to-score cycle and advances several lineages at once, so `turn` and `slot` say when an experiment ran and which lineage produced it. The convergence rule is measured against turns; the 50-cap counts scripts.

| # | turn | slot | phase | parent | status | secs | primary | vs baseline | hypothesis |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | - | eda | - | ok | 1 | n/a | - | In the data-inspection stage, quantify temporal drift, entity cold-sta |
| 1 | 0 | - | baseline | - | ok | 17 | 0.6035 | +0.0019 | Reproduce the official baseline at the modeling stage using a 16-dimen |
| 2 | 1 | 0 | improve | #1 | kept | 77 | 0.6041 | +0.0025 | Target prediction formation and fusion by comparing expanded-field FM, |
| 3 | 1 | 1 | improve | #1 | kept | 37 | 0.6041 | +0.0025 | Target prediction formation under drift by comparing recency-weighted  |
| 4 | 1 | 2 | improve | #1 | kept | 64 | 0.6045 | +0.0029 | Target prediction formation and temporal adaptation by comparing DCN c |
| 5 | 1 | 3 | improve | #1 | rejected | 36 | 0.6045 | +0.0029 | Target prediction formation and supervision by comparing recency-weigh |
| 6 | 2 | 0 | improve | #4 | kept | 98 | 0.6046 | +0.0030 | Target sequence-context prediction and non-parametric conditioning: DI |
| 7 | 2 | 1 | improve | #4 | kept | 23 | 0.6046 | +0.0030 | Target the training-objective and prediction-formation stages by compa |
| 8 | 2 | 2 | improve | #4 | kept | 348 | 0.6047 | +0.0031 | Target prediction formation with a breadth comparison of field-aware f |
| 9 | 2 | 3 | improve | #4 | kept | 90 | 0.6048 | +0.0032 | Target the supervision stage with leakage-free multi-task MMoE, PLE, a |
| 10 | 3 | 0 | improve | #9 | kept | 4 | 0.6048 | +0.0032 | Target the prediction-formation and temporal-adaptation stages with co |
| 11 | 3 | 1 | improve | #9 | kept | 3 | 0.6048 | +0.0032 | Target low-correlation prediction formation and logged-slate context b |
| 12 | 3 | 2 | improve | #9 | kept | 13 | 0.6048 | +0.0032 | Target prediction formation with three low-correlation models over lea |
| 13 | 3 | 3 | improve | #9 | kept | 16 | 0.6048 | +0.0032 | Target personalized residual-ranking formation with three structurally |
| 14 | 4 | 0 | improve | #10 | kept | 6 | 0.6048 | +0.0032 | Target drift-robust prediction formation with three new estimators—a d |
| 15 | 4 | 1 | improve | #10 | kept | 22 | 0.6048 | +0.0032 | Target the training-objective stage with conditional-utility learning: |
| 16 | 4 | 2 | improve | #10 | kept | 15 | 0.6048 | +0.0032 | Target safe logged-slate feature construction and prediction formation |
| 17 | 4 | 3 | improve | #10 | kept | 9 | 0.6051 | +0.0035 | Target metric-specific post-model reranking with diversity-aware slate |

## Portfolio

- Lineages advanced per turn: **4**
- Turns: **4**  ·  scripts executed: **18** of 50
- A turn is one hypothesis-to-score cycle -- Figure 1's iteration, and what the convergence rule is measured against. Scripts are reported separately so the run can be checked against the stricter reading of the 50-iteration cap as well.
- Lineages archived: **4**  ·  revived from the archive: **1**

### Did the lineages actually disagree?

Within-user rank correlation between the live slots' validation predictions. This is the acceptance test for the whole design: above ~0.95 the extra slots are three copies of one agent and bought nothing.

| turn | scripts | mean corr | max corr | alert |
|---|---|---|---|---|
| 1 | #2, #3, #4, #5 | 0.9001 | 0.9072 | no |
| 2 | #6, #7, #8, #9 | 0.8222 | 0.9596 | **YES** |
| 3 | #10, #11, #12, #13 | 0.1818 | 0.8813 | no |
| 4 | #14, #15, #16, #17 | 0.7620 | 0.8418 | no |

### Lineages retired and replaced

A slot that stops beating its own best is archived, not stopped -- the run's convergence counter is untouched by it. Refill alternates fresh drafts with a revival, and a revival is chosen for score AND for disagreeing with what is already live.

| turn | slot | archived at | choice | detail |
|---|---|---|---|---|
| 2 | 0 | 0.6046 | fresh | new draft |
| 2 | 1 | 0.6046 | fresh | new draft |
| 2 | 2 | 0.6047 | fresh | new draft |
| 2 | 3 | 0.6048 | revived | entry #3 (primary 0.6048) |

### Controller blend of the portfolio

Every turn the controller blends the trusted incumbent, the live slots and the archive, choosing members on one half of validation and confirming on the other. A blend that wins the selection fold and loses the confirmation fold is refused: it won the split, not the task.

| turn | pool | accepted | members | fold A | fold B | reason |
|---|---|---|---|---|---|---|
| 1 | 3 | no | - | - | - | no member improved fold A |
| 2 | 4 | no | - | - | - | no member improved fold A |
| 3 | 8 | no | - | - | - | no member improved fold A |
| 4 | 8 | no | - | - | - | no member improved fold A |

## Code diff applied, per iteration

Each iteration edits a parent script. This is that edit, reconstructed from the two full sources in the ledger, truncated to the first 40 lines per iteration.


**#1 -> #2**  (+365 / -78 lines)

```diff
--- iter_1.py
+++ iter_2.py
@@ -13,11 +13,30 @@
 
 START = time.time()
-SEED = 2024
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-K = 16
+SEED = 2026
+BATCH_SIZE = 8192
+RANK = 16
 LR = 0.001
-BATCH_SIZE = 8192
-CHECKPOINT_EPOCHS = {4, 8, 12, 16, 20}
-MAX_EPOCHS = max(CHECKPOINT_EPOCHS)
+
+FIELDS = [
+    "user_id",
+    "video_id",
+    "author_id",
+    "tab",
+    "duration_bucket",
+    "tag",
+    "upload_type",
+    "music_type",
+    "hour",
+]
+STAT_FIELDS = [
+    "video_id",
+    "author_id",
+    "tab",
+    "duration_bucket",
+    "tag",
+    "upload_type",
+    "music_type",
+    "hour",
+]
 
 random.seed(SEED)
@@ -26,37 +45,34 @@
... 497 more diff lines
```

**#1 -> #3**  (+423 / -138 lines)

```diff
--- iter_1.py
+++ iter_3.py
@@ -2,4 +2,5 @@
 import time
 import json
+import gc
 import math
 import random
@@ -7,18 +8,11 @@
 import torch
 import torch.nn as nn
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
 from pipeline.evaluate import evaluate
 
-
 START = time.time()
-SEED = 2024
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-K = 16
-LR = 0.001
-BATCH_SIZE = 8192
-CHECKPOINT_EPOCHS = {4, 8, 12, 16, 20}
-MAX_EPOCHS = max(CHECKPOINT_EPOCHS)
-
+SEED = 314159
 random.seed(SEED)
 np.random.seed(SEED)
@@ -26,138 +20,373 @@
 torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
 
-
-cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
-offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
-total_cardinality = int(sum(cards))
-zero_rows = torch.as_tensor(offsets, dtype=torch.long)
-
-
-def encode(split):
... 565 more diff lines
```

**#1 -> #4**  (+360 / -88 lines)

```diff
--- iter_1.py
+++ iter_4.py
@@ -13,11 +13,21 @@
 
 START = time.time()
-SEED = 2024
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-K = 16
-LR = 0.001
+SEED = 7319
+FIELDS = [
+    "user_id",
+    "video_id",
+    "author_id",
+    "tab",
+    "duration_bucket",
+    "tag",
+    "upload_type",
+    "music_type",
+    "hour",
+]
+EMBED_DIM = 8
 BATCH_SIZE = 8192
-CHECKPOINT_EPOCHS = {4, 8, 12, 16, 20}
-MAX_EPOCHS = max(CHECKPOINT_EPOCHS)
+CHECKPOINTS = (2, 4, 6)
+MAX_EPOCHS = max(CHECKPOINTS)
+LR = 0.002
 
 random.seed(SEED)
@@ -26,122 +36,284 @@
 torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
 
-
 cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
 offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
 total_cardinality = int(sum(cards))
-zero_rows = torch.as_tensor(offsets, dtype=torch.long)
+n_fields = len(FIELDS)
+flat_dim = n_fields * EMBED_DIM
... 499 more diff lines
```

**#1 -> #5**  (+341 / -123 lines)

```diff
--- iter_1.py
+++ iter_5.py
@@ -13,11 +13,26 @@
 
 START = time.time()
-SEED = 2024
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-K = 16
-LR = 0.001
-BATCH_SIZE = 8192
-CHECKPOINT_EPOCHS = {4, 8, 12, 16, 20}
-MAX_EPOCHS = max(CHECKPOINT_EPOCHS)
+SEED = 7319
+EPOCHS = 3
+BATCH_SIZE = 16384
+HALF_LIFE_DAYS = 9.0
+
+FIELDS = [
+    "user_id",
+    "video_id",
+    "author_id",
+    "tab",
+    "duration_bucket",
+    "tag",
+    "upload_type",
+    "music_type",
+    "hour",
+    "user_active_degree",
+    "onehot_feat3",
+    "onehot_feat8",
+]
+USER_FIELD_POS = [0, 9, 10, 11]
+ITEM_FIELD_POS = [1, 2, 3, 4, 5, 6, 7, 8]
+AUX_CANDIDATES = ["is_click", "is_like", "is_follow", "is_comment"]
 
 random.seed(SEED)
@@ -26,122 +41,237 @@
 torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
 
-
... 483 more diff lines
```

**#4 -> #6**  (+455 / -206 lines)

```diff
--- iter_4.py
+++ iter_6.py
@@ -2,5 +2,4 @@
 import time
 import json
-import math
 import random
 import numpy as np
@@ -13,5 +12,10 @@
 
 START = time.time()
-SEED = 7319
+SEED = 19427
+random.seed(SEED)
+np.random.seed(SEED)
+torch.manual_seed(SEED)
+torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
+
 FIELDS = [
     "user_id",
@@ -19,31 +23,21 @@
     "author_id",
     "tab",
+    "tag",
     "duration_bucket",
-    "tag",
     "upload_type",
-    "music_type",
     "hour",
 ]
-EMBED_DIM = 8
+VIDEO_FIELD = FIELDS.index("video_id")
+CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
+OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
+TOTAL_CARD = int(sum(CARDS))
+VIDEO_OFFSET = int(OFFSETS[VIDEO_FIELD])
+
+HISTORY_LEN = 12
+EMBED_DIM = 12
 BATCH_SIZE = 8192
... 808 more diff lines
```

**#4 -> #7**  (+458 / -293 lines)

```diff
--- iter_4.py
+++ iter_7.py
@@ -2,5 +2,4 @@
 import time
 import json
-import math
 import random
 import numpy as np
@@ -13,21 +12,22 @@
 
 START = time.time()
-SEED = 7319
+SEED = 19417
+DEVICE = torch.device("cpu")
+BATCH_SIZE = 8192
+CHECKPOINTS = (2, 4)
+MAX_EPOCHS = max(CHECKPOINTS)
+N_HARD_NEGATIVES = 3
+
 FIELDS = [
     "user_id",
     "video_id",
     "author_id",
+    "tag",
     "tab",
     "duration_bucket",
-    "tag",
-    "upload_type",
-    "music_type",
-    "hour",
 ]
-EMBED_DIM = 8
-BATCH_SIZE = 8192
-CHECKPOINTS = (2, 4, 6)
-MAX_EPOCHS = max(CHECKPOINTS)
-LR = 0.002
+CARDS = [int(FEATURE_CARDINALITIES[x]) for x in FIELDS]
+OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
+TOTAL_CARD = int(sum(CARDS))
 
... 841 more diff lines
```

**#4 -> #8**  (+272 / -225 lines)

```diff
--- iter_4.py
+++ iter_8.py
@@ -2,5 +2,4 @@
 import time
 import json
-import math
 import random
 import numpy as np
@@ -13,5 +12,5 @@
 
 START = time.time()
-SEED = 7319
+SEED = 19427
 FIELDS = [
     "user_id",
@@ -27,7 +26,8 @@
 EMBED_DIM = 8
 BATCH_SIZE = 8192
-CHECKPOINTS = (2, 4, 6)
+LR = 0.002
+CHECKPOINTS = (2, 4)
 MAX_EPOCHS = max(CHECKPOINTS)
-LR = 0.002
+HALF_LIFE = 7.0
 
 random.seed(SEED)
@@ -36,9 +36,8 @@
 torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
 
-cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
+cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
 offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
 total_cardinality = int(sum(cards))
 n_fields = len(FIELDS)
-flat_dim = n_fields * EMBED_DIM
 
 pair_i, pair_j = np.triu_indices(n_fields, k=1)
@@ -58,42 +57,128 @@
 
 
... 631 more diff lines
```

**#4 -> #9**  (+335 / -221 lines)

```diff
--- iter_4.py
+++ iter_9.py
@@ -2,5 +2,4 @@
 import time
 import json
-import math
 import random
 import numpy as np
@@ -13,5 +12,5 @@
 
 START = time.time()
-SEED = 7319
+SEED = 9473
 FIELDS = [
     "user_id",
@@ -25,9 +24,11 @@
     "hour",
 ]
+AUX_CANDIDATES = ["is_click", "is_like", "is_follow"]
 EMBED_DIM = 8
 BATCH_SIZE = 8192
-CHECKPOINTS = (2, 4, 6)
+CHECKPOINTS = (2, 4)
 MAX_EPOCHS = max(CHECKPOINTS)
 LR = 0.002
+HALF_LIFE = 7.0
 
 random.seed(SEED)
@@ -36,14 +37,9 @@
 torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
 
-cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
+cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
 offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
 total_cardinality = int(sum(cards))
 n_fields = len(FIELDS)
-flat_dim = n_fields * EMBED_DIM
-
-pair_i, pair_j = np.triu_indices(n_fields, k=1)
-pair_i_t = torch.as_tensor(pair_i, dtype=torch.long)
... 695 more diff lines
```

**#9 -> #10**  (+478 / -489 lines)

```diff
--- iter_9.py
+++ iter_10.py
@@ -2,8 +2,7 @@
 import time
 import json
-import random
 import numpy as np
-import torch
-import torch.nn as nn
+from scipy import sparse
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -12,386 +11,375 @@
 
 START = time.time()
-SEED = 9473
-FIELDS = [
-    "user_id",
+HALF_LIFE = 8.0
+EPS = 1e-5
+
+ENTITY_FIELDS = [
     "video_id",
     "author_id",
+    "tag",
     "tab",
     "duration_bucket",
-    "tag",
     "upload_type",
     "music_type",
-    "hour",
+    "onehot_feat3",
+    "onehot_feat8",
 ]
-AUX_CANDIDATES = ["is_click", "is_like", "is_follow"]
-EMBED_DIM = 8
-BATCH_SIZE = 8192
-CHECKPOINTS = (2, 4)
-MAX_EPOCHS = max(CHECKPOINTS)
... 1008 more diff lines
```

**#9 -> #11**  (+379 / -490 lines)

```diff
--- iter_9.py
+++ iter_11.py
@@ -2,8 +2,7 @@
 import time
 import json
-import random
 import numpy as np
-import torch
-import torch.nn as nn
+from scipy import sparse
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -13,385 +12,268 @@
 START = time.time()
 SEED = 9473
-FIELDS = [
-    "user_id",
-    "video_id",
-    "author_id",
-    "tab",
-    "duration_bucket",
-    "tag",
-    "upload_type",
-    "music_type",
-    "hour",
-]
-AUX_CANDIDATES = ["is_click", "is_like", "is_follow"]
-EMBED_DIM = 8
-BATCH_SIZE = 8192
-CHECKPOINTS = (2, 4)
-MAX_EPOCHS = max(CHECKPOINTS)
-LR = 0.002
-HALF_LIFE = 7.0
-
-random.seed(SEED)
+LATENT_DIM = 32
+SMOOTHING = 80.0
+
 np.random.seed(SEED)
... 908 more diff lines
```

**#9 -> #12**  (+390 / -506 lines)

```diff
--- iter_9.py
+++ iter_12.py
@@ -6,30 +6,11 @@
 import torch
 import torch.nn as nn
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
 from pipeline.evaluate import evaluate
 
-
 START = time.time()
-SEED = 9473
-FIELDS = [
-    "user_id",
-    "video_id",
-    "author_id",
-    "tab",
-    "duration_bucket",
-    "tag",
-    "upload_type",
-    "music_type",
-    "hour",
-]
-AUX_CANDIDATES = ["is_click", "is_like", "is_follow"]
-EMBED_DIM = 8
-BATCH_SIZE = 8192
-CHECKPOINTS = (2, 4)
-MAX_EPOCHS = max(CHECKPOINTS)
-LR = 0.002
-HALF_LIFE = 7.0
-
+SEED = 24117
 random.seed(SEED)
 np.random.seed(SEED)
@@ -37,361 +18,264 @@
 torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
 
-cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
-offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
... 933 more diff lines
```

**#9 -> #13**  (+313 / -428 lines)

```diff
--- iter_9.py
+++ iter_13.py
@@ -3,4 +3,5 @@
 import json
 import random
+import gc
 import numpy as np
 import torch
@@ -10,7 +11,12 @@
 from pipeline.evaluate import evaluate
 
-
 START = time.time()
-SEED = 9473
+SEED = 27183
+BATCH_SIZE = 16384
+EPOCHS = 3
+LR = 0.003
+HALF_LIFE = 7.0
+EMBED_DIM = 12
+
 FIELDS = [
     "user_id",
@@ -23,12 +29,7 @@
     "music_type",
     "hour",
+    "video_type",
 ]
-AUX_CANDIDATES = ["is_click", "is_like", "is_follow"]
-EMBED_DIM = 8
-BATCH_SIZE = 8192
-CHECKPOINTS = (2, 4)
-MAX_EPOCHS = max(CHECKPOINTS)
-LR = 0.002
-HALF_LIFE = 7.0
+FIELD_INDEX = {name: i for i, name in enumerate(FIELDS)}
 
 random.seed(SEED)
@@ -39,27 +40,55 @@
 cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
... 832 more diff lines
```

**#10 -> #14**  (+550 / -387 lines)

```diff
--- iter_10.py
+++ iter_14.py
@@ -3,6 +3,4 @@
 import json
 import numpy as np
-from scipy import sparse
-from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -11,9 +9,32 @@
 
 START = time.time()
-HALF_LIFE = 8.0
-EPS = 1e-5
-
-ENTITY_FIELDS = [
-    "video_id",
+EPS = 1e-9
+
+NB_FIELDS = [
+    "duration_bucket",
+    "tab",
+    "tag",
+    "upload_type",
+    "music_type",
+    "video_type",
+    "hour",
+    "user_active_degree",
+    "fans_user_num_range",
+    "follow_user_num_range",
+    "friend_user_num_range",
+    "register_days_bucket",
+    "onehot_feat1",
+    "onehot_feat2",
+    "onehot_feat3",
+    "onehot_feat4",
+    "onehot_feat7",
+    "onehot_feat8",
+    "onehot_feat9",
+    "onehot_feat10",
... 1048 more diff lines
```

**#10 -> #15**  (+376 / -432 lines)

```diff
--- iter_10.py
+++ iter_15.py
@@ -3,6 +3,5 @@
 import json
 import numpy as np
-from scipy import sparse
-from scipy.sparse.linalg import svds
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -11,8 +10,11 @@
 
 START = time.time()
-HALF_LIFE = 8.0
-EPS = 1e-5
-
-ENTITY_FIELDS = [
+EPS = 1e-9
+HALF_LIFE = 9.0
+USER_SMOOTH = 8.0
+MAIN_SMOOTH = 25.0
+PAIR_SMOOTH = 18.0
+
+MAIN_FIELDS = [
     "video_id",
     "author_id",
@@ -22,9 +24,32 @@
     "upload_type",
     "music_type",
+    "video_type",
+    "hour",
+    "user_active_degree",
+    "is_video_author",
+    "onehot_feat1",
+    "onehot_feat2",
     "onehot_feat3",
+    "onehot_feat4",
+    "onehot_feat7",
     "onehot_feat8",
+    "onehot_feat11",
... 886 more diff lines
```

**#10 -> #16**  (+516 / -462 lines)

```diff
--- iter_10.py
+++ iter_16.py
@@ -3,6 +3,5 @@
 import json
 import numpy as np
-from scipy import sparse
-from scipy.sparse.linalg import svds
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -11,6 +10,5 @@
 
 START = time.time()
-HALF_LIFE = 8.0
-EPS = 1e-5
+EPS = 1e-8
 
 ENTITY_FIELDS = [
@@ -19,366 +17,480 @@
     "tag",
     "tab",
+    "upload_type",
     "duration_bucket",
-    "upload_type",
-    "music_type",
     "onehot_feat3",
     "onehot_feat8",
 ]
 
-PAIR_FIELDS = [
-    "video_id",
-    "author_id",
+CAT_NUM_FIELDS = [
+    "hour",
+    "tab",
     "tag",
-    "tab",
+    "upload_type",
     "duration_bucket",
-    "upload_type",
... 1024 more diff lines
```

**#10 -> #17**  (+295 / -497 lines)

```diff
--- iter_10.py
+++ iter_17.py
@@ -3,6 +3,4 @@
 import json
 import numpy as np
-from scipy import sparse
-from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -11,31 +9,6 @@
 
 START = time.time()
-HALF_LIFE = 8.0
-EPS = 1e-5
-
-ENTITY_FIELDS = [
-    "video_id",
-    "author_id",
-    "tag",
-    "tab",
-    "duration_bucket",
-    "upload_type",
-    "music_type",
-    "onehot_feat3",
-    "onehot_feat8",
-]
-
-PAIR_FIELDS = [
-    "video_id",
-    "author_id",
-    "tag",
-    "tab",
-    "duration_bucket",
-    "upload_type",
-]
-
-SMOOTH_ENTITY = 18.0
-SMOOTH_PAIR = 12.0
-SVD_RANK = 24
+TOP_L = 12
... 818 more diff lines
```

## Search tree

Each line is an executed script; indentation is the edit it was derived from. A node marked `[retired]` produced three children that failed to improve on it, so the search abandoned that branch and backtracked.

```
#1 0.6035 [retired]  Reproduce the official baseline at the modeling stage using a 16-dimen
  #2 0.6041  Target prediction formation and fusion by comparing expanded-field FM,
  #3 0.6041  Target prediction formation under drift by comparing recency-weighted 
  #4 0.6045 [retired]  Target prediction formation and temporal adaptation by comparing DCN c
    #6 0.6046  Target sequence-context prediction and non-parametric conditioning: DI
    #7 0.6046  Target the training-objective and prediction-formation stages by compa
    #8 0.6047  Target prediction formation with a breadth comparison of field-aware f
    #9 0.6048 [retired]  Target the supervision stage with leakage-free multi-task MMoE, PLE, a
      #10 0.6048 [retired]  Target the prediction-formation and temporal-adaptation stages with co
        #14 0.6048  Target drift-robust prediction formation with three new estimators—a d
        #15 0.6048  Target the training-objective stage with conditional-utility learning:
        #16 0.6048  Target safe logged-slate feature construction and prediction formation
        #17 0.6051  Target metric-specific post-model reranking with diversity-aware slate
      #11 0.6048  Target low-correlation prediction formation and logged-slate context b
      #12 0.6048  Target prediction formation with three low-correlation models over lea
      #13 0.6048  Target personalized residual-ranking formation with three structurally
```

## What the agent established

The agent's belief set, revised after every scored iteration rather than appended to, so later evidence can demote an earlier conclusion instead of piling up beside it. A claim marked `(invalidated)` was contradicted by a later result. Machine-readable form with per-claim evidence in `knowledge.json`.

```
- (active) A 16-dimensional Factorization Machine using user_id, video_id, author_id, tab, and duration_bucket, with validation-selected training duration and train+validation refitting, reproduces the official baseline within seed noise (0.6035 versus 0.6016) but provides no measurable improvement. [iters 1]
- (active) Across all tested lineages, now including day-consensus generative Naive Bayes, conditional-utility learning, bagged tree models, and diversity-aware reranking, the best primary score is 0.6051—only 0.0016 above the 0.6035 FM baseline and therefore not a measurable gain. [iters 2,3,4,5,6,7]
Ruled out by evidence (do not revisit without a new mechanism):
- (invalidated) The earlier low-agreement characterization of the live outputs is invalidated: the latest outputs have substantial within-user rank agreement, with mean pairwise correlation 0.762 and pairwise values from 0.690 to 0.842, though they are not identical. [iters 5,6,7]
```

## Alternatives compared inside iterations

257 candidate solutions were built and scored across 15 iteration(s). The convergence rule charges one iteration per experiment, so searching inside an iteration buys comparisons the iteration budget cannot.

| iteration | candidates (validation primary) |
|---|---|
| #2 | expanded_fm_standalone 0.6033, expanded_fm_best_incumbent_blend 0.6040, nfm_standalone 0.6002, nfm_best_incumbent_blend 0.6036, recency_empirical_bayes_standalone 0.5902, recency_empirical_bayes_best_incumbent_blend 0.6038 |
| #3 | empirical_bayes 0.5802, empirical_bayes_blend_0.20 0.6035, empirical_bayes_blend_0.35 0.6005, empirical_bayes_blend_0.50 0.5954, empirical_bayes_blend_0.65 0.5898, empirical_bayes_blend_0.80 0.5830, gbdt_binary_recency 0.4473, gbdt_binary_recency_blend_0.20 0.6010 |
| #4 | dcn_uniform_best_blend 0.6041, dcn_uniform_blend_alpha 0.4000, dcn_uniform_epoch 4.0000, dcn_uniform_standalone 0.6037, fibinet_recency3_best_blend 0.6043, fibinet_recency3_blend_alpha 0.4000, fibinet_recency3_epoch 4.0000, fibinet_recency3_standalone 0.6029 |
| #6 | din_attention_best_blend 0.6046, din_attention_blend_alpha 0.3000, din_attention_epoch 2.0000, din_attention_standalone 0.6030, gru_history_best_blend 0.6045, gru_history_blend_alpha 0.1000, gru_history_epoch 2.0000, gru_history_standalone 0.6029 |
| #7 | empirical_bayes_h3_full_best_blend 0.6045, empirical_bayes_h3_full_blend_alpha 0.0500, empirical_bayes_h3_full_standalone 0.5906, empirical_bayes_h3_marginal_best_blend 0.6045, empirical_bayes_h3_marginal_blend_alpha 0.0500, empirical_bayes_h3_marginal_standalone 0.5907, empirical_bayes_h7_full_best_blend 0.6045, empirical_bayes_h7_full_blend_alpha 0.1000 |
| #8 | autoint_best_blend 0.6045, autoint_blend_alpha 0.1000, autoint_epoch 2.0000, autoint_standalone 0.6027, field_aware_fm_best_blend 0.6045, field_aware_fm_blend_alpha 0.1000, field_aware_fm_epoch 2.0000, field_aware_fm_standalone 0.6025 |
| #9 | esmm_best_blend 0.6047, esmm_blend_alpha 0.4000, esmm_epoch 4.0000, esmm_standalone 0.6036, mmoe_best_blend 0.6048, mmoe_blend_alpha 0.4000, mmoe_epoch 4.0000, mmoe_standalone 0.6039 |
| #10 | entity_bayes_alpha 0.0000, entity_bayes_best_blend 0.6045, entity_bayes_standalone 0.5912, latent_svd_alpha 0.0000, latent_svd_best_blend 0.6045, latent_svd_standalone 0.5811, personalized_bayes_alpha 0.1250, personalized_bayes_best_blend 0.6046 |
| #11 | chronology_fatigue_alpha 0.0500, chronology_fatigue_blend 0.6046, chronology_fatigue_standalone 0.4949, positive_svd_alpha 0.0000, positive_svd_blend 0.6045, positive_svd_standalone 0.5366, signed_svd_alpha 0.0000, signed_svd_blend 0.6045 |
| #12 | bagged_rf_blend_0.0 0.6045, bagged_rf_blend_0.2 0.6017, bagged_rf_blend_0.4 0.5819, bagged_rf_blend_0.6 0.5385, bagged_rf_blend_0.8 0.4986, bagged_rf_blend_1.0 0.4883, bagged_rf_standalone 0.4887, dense_linear_blend_0.0 0.6045 |
| #13 | all_equal_blend_0.0 0.6045, all_equal_blend_0.125 0.6045, all_equal_blend_0.25 0.6046, all_equal_blend_0.375 0.6041, all_equal_blend_0.5 0.6027, all_equal_blend_0.625 0.6010, all_equal_blend_0.75 0.5992, all_equal_blend_0.875 0.5970 |
| #14 | cross_family_consensus_standalone 0.5850, day_consensus_nb_alpha 0.0000, day_consensus_nb_best_blend 0.6045, day_consensus_nb_standalone 0.5672, graph_propagation_alpha 0.0000, graph_propagation_best_blend 0.6045, graph_propagation_standalone 0.5769, preference_kernel_alpha 0.0625 |
| #15 | conditional_additive_gam_alpha 0.0000, conditional_additive_gam_best_blend 0.6045, conditional_additive_gam_standalone 0.5952, conditional_boosted_trees_alpha 0.1600, conditional_boosted_trees_best_blend 0.6046, conditional_boosted_trees_standalone 0.5993, conditional_interaction_gam_alpha 0.0800, conditional_interaction_gam_best_blend 0.6046 |
| #16 | bagged_random_forest_alpha 0.0000, bagged_random_forest_best_blend 0.6045, bagged_random_forest_standalone 0.4657, lsh_neighborhood_alpha 0.0625, lsh_neighborhood_best_blend 0.6046, lsh_neighborhood_standalone 0.5769, rff_kernel_ridge_alpha 0.1250, rff_kernel_ridge_best_blend 0.6046 |
| #17 | author_alpha 0.7500, author_best 0.6050, author_penalty 0.0800, author_tag_alpha 0.7500, author_tag_best 0.6047, author_tag_penalty 0.0800, tag_alpha 0.7500, tag_best 0.6048 |

## Resource usage (Feasibility & Practicality)

- **Agent wall-clock to converged result: 0.46 h (27 min)**
- Total LLM tokens: **319,616** (214,538 in / 105,078 out), including the knowledge-revision stage
- Iterations used: **18 of 50** (16 accepted scores, 0 failed, 1 rejected)
- Compute inside generated scripts: **0.24 h (15 min)** on CPU.
- Mean tokens per iteration: 17,756
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

Across 38 development runs of this agent, 307 iterations were executed and 28 failed. Every one was handled in-loop; none was escalated to a human. Failure taxonomy:

- `}`: 7
- `TIMEOUT`: 4
- `TypeError`: 4
- `ValueError`: 4
- `IndexError`: 2
- `AttributeError`: 2
- `TimeoutError`: 2
- `lightgbm.basic.LightGBMError`: 1

Each recovery path, with a concrete instance:

- **Retry with source.** `r38_1k` #9 crashed with `lightgbm.basic.LightGBMError: Number of rows 13924 exceeds upper limit`. The traceback *and the failing script* went back to the proposer, which fixed it: #10 scored 0.6704.
- **Timeout, handled as distinct from a bug.** `r38_1k` #11 was killed at the limit after 4373s. The feedback says the approach is too slow rather than wrong, because re-running a slow script unchanged just times out again.

## Result

- Best validation primary: **0.6051** (baseline 0.6016, delta +0.0035)
- From iteration #17: Target metric-specific post-model reranking with diversity-aware slate optimization; penalizing redundant videos/authors/tags among the incumbent’s top candidates can replace correlated low-utility impressions and raise nDCG@5 while partial rank blending limits GAUC damage.
- Selection is on validation only, per the scoring rules; the hidden test set was never used to choose between iterations.
- Submission: written to `runs/r88_4slot/submission.csv`

### Results table

| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| validation (best iteration) | 0.6718 | 0.5383 | 0.6051 |
| official baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
| hidden test (this submission) | 0.6675 | 0.5329 | **0.6002** |
| official baseline (test) | 0.6610 | 0.5282 | 0.5946 |

**Absolute delta over baseline on hidden test: GAUC +0.0065, nDCG@5 +0.0047, mean +0.0056** (primary +0.0056).

Per the scoring formula, delta(m) = score_agent(m) - score_baseline(m), averaged over metrics.
