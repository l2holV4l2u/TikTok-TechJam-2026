# Run report - r89_5slot

> 1 scored iteration(s) were rejected by the integrity critic. They are excluded from the search tree, cross-run memory, and submission.

## Loop stages executed by the agent

- **Inspect data (EDA):** completed at iteration #0 - the agent wrote and ran its own exploratory script; its findings are in `eda_report.txt` and were carried into every later prompt.
- **Reproduce official baseline (Requirement 1):** 1 attempt(s); the agent's own pipeline reached validation primary **0.6015** against the official 0.6016 (delta -0.0001, inside the baseline's 5-seed noise). That script became the root of the search tree.
- **Iterate:** 20 experiments proposed, executed and evaluated by the agent, each branching from a node of its own search tree.

## Iteration log

A turn is one hypothesis-to-score cycle and advances several lineages at once, so `turn` and `slot` say when an experiment ran and which lineage produced it. The convergence rule is measured against turns; the 50-cap counts scripts.

| # | turn | slot | phase | parent | status | secs | primary | vs baseline | hypothesis |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | - | eda | - | ok | 1 | n/a | - | In the data-inspection stage, quantify temporal/user sparsity, entity  |
| 1 | 0 | - | baseline | - | ok | 8 | 0.6015 | -0.0001 | Reproduce the official baseline stage by training a rank-16 categorica |
| 2 | 1 | 0 | improve | #1 | ok | 99 | 0.6037 | +0.0021 | Target prediction formation and fusion by comparing an expanded pairwi |
| 3 | 1 | 1 | improve | #1 | kept | 112 | 0.6040 | +0.0024 | Target prediction formation and temporal-drift handling by comparing r |
| 4 | 1 | 2 | improve | #1 | kept | 113 | 0.6041 | +0.0025 | Target prediction formation and drift-aware training by comparing rece |
| 5 | 1 | 3 | improve | #1 | kept | 68 | 0.6042 | +0.0026 | Target prediction formation and temporal-interest representation by co |
| 6 | 1 | 4 | improve | #1 | kept | 10 | 0.6042 | +0.0026 | Target the logged-impression context and post-model fusion stages: com |
| 7 | 2 | 0 | improve | #6 | kept | 6 | 0.6043 | +0.0027 | Target leakage-free non-parametric personalization and score-fusion st |
| 8 | 2 | 1 | improve | #6 | rejected | 63 | 0.6044 | +0.0028 | Target the supervision and prediction-formation stages by comparing a  |
| 9 | 2 | 2 | improve | #6 | kept | 46 | 0.6044 | +0.0028 | Target the ranking-loss and prediction-formation stages by comparing t |
| 10 | 2 | 3 | improve | #6 | kept | 69 | 0.6044 | +0.0028 | Target prediction formation under temporal drift by comparing DCN cros |
| 11 | 2 | 4 | improve | #6 | kept | 51 | 0.6050 | +0.0034 | Target the loss and prediction-formation stages with pointwise-anchore |
| 12 | 3 | 0 | improve | #11 | kept | 4 | 0.6051 | +0.0035 | Target the personalization and prediction-formation stages with low-ra |
| 13 | 3 | 1 | improve | #11 | kept | 61 | 0.6054 | +0.0038 | Target train-only auxiliary-supervision and prediction formation by co |
| 14 | 3 | 2 | improve | #11 | kept | 39 | 0.6055 | +0.0039 | Target the structured-loss stage by comparing setwise ListNet additive |
| 15 | 3 | 3 | improve | #11 | kept | 14 | 0.6055 | +0.0039 | Target temporal-drift and prediction-formation stages by comparing poo |
| 16 | 3 | 4 | improve | #11 | kept | 55 | 0.6055 | +0.0039 | Target prediction formation and loss isolation by training pointwise,  |
| 17 | 4 | 0 | improve | #14 | kept | 160 | 0.6057 | +0.0041 | I am testing prediction formation across explicit high-order xDeepFM i |
| 18 | 4 | 1 | improve | #14 | kept | 42 | 0.6057 | +0.0041 | Target list-context prediction formation under activity drift by compa |
| 19 | 4 | 2 | improve | #14 | kept | 23 | 0.6057 | +0.0041 | Target drift-robust prediction formation by comparing a recency-weight |
| 20 | 4 | 3 | improve | #12 | ok | 14 | 0.6057 | +0.0041 | Target prediction formation with a breadth comparison of stationary ca |
| 21 | 4 | 4 | improve | #14 | kept | 3 | 0.6057 | +0.0041 | Target memory-based collaborative prediction and fusion by comparing i |

## Portfolio

- Lineages advanced per turn: **5**
- Turns: **4**  ·  scripts executed: **22** of 50
- A turn is one hypothesis-to-score cycle -- Figure 1's iteration, and what the convergence rule is measured against. Scripts are reported separately so the run can be checked against the stricter reading of the 50-iteration cap as well.
- Lineages archived: **5**  ·  revived from the archive: **1**

### Did the lineages actually disagree?

Within-user rank correlation between the live slots' validation predictions. This is the acceptance test for the whole design: above ~0.95 the extra slots are three copies of one agent and bought nothing.

| turn | scripts | mean corr | max corr | alert |
|---|---|---|---|---|
| 1 | #2, #3, #4, #5, #6 | 0.6171 | 0.9323 | no |
| 2 | #7, #8, #9, #10, #11 | 0.7760 | 0.9300 | no |
| 3 | #12, #13, #14, #15, #16 | 0.4783 | 0.8009 | no |
| 4 | #17, #18, #19, #20, #21 | 0.3431 | 0.6843 | no |

### Lineages retired and replaced

A slot that stops beating its own best is archived, not stopped -- the run's convergence counter is untouched by it. Refill alternates fresh drafts with a revival, and a revival is chosen for score AND for disagreeing with what is already live.

| turn | slot | archived at | choice | detail |
|---|---|---|---|---|
| 3 | 0 | 0.6051 | fresh | new draft |
| 3 | 1 | 0.6054 | fresh | new draft |
| 3 | 2 | 0.6055 | fresh | new draft |
| 3 | 3 | 0.6055 | revived | entry #0 (primary 0.6051) |
| 3 | 4 | 0.6055 | fresh | new draft |

### Controller blend of the portfolio

Every turn the controller blends the trusted incumbent, the live slots and the archive, choosing members on one half of validation and confirming on the other. A blend that wins the selection fold and loses the confirmation fold is refused: it won the split, not the task.

| turn | pool | accepted | members | fold A | fold B | reason |
|---|---|---|---|---|---|---|
| 1 | 5 | no | - | - | - | no member improved fold A |
| 2 | 5 | no | - | - | - | no member improved fold A |
| 3 | 5 | no | - | - | - | no member improved fold A |
| 4 | 10 | no | - | - | - | no member improved fold A |

## Code diff applied, per iteration

Each iteration edits a parent script. This is that edit, reconstructed from the two full sources in the ledger, truncated to the first 40 lines per iteration.


**#1 -> #2**  (+368 / -145 lines)

```diff
--- iter_1.py
+++ iter_2.py
@@ -3,4 +3,5 @@
 import json
 import random
+import gc
 import numpy as np
 import torch
@@ -11,12 +12,23 @@
 
 
-START_TIME = time.time()
-SEED = 2024
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
+START = time.time()
+SEED = 314159
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
 K = 16
+FM_MAX_EPOCHS = 9
+DEEP_MAX_EPOCHS = 6
+BATCH_SIZE = 4096
+PRED_BATCH_SIZE = 32768
 LR = 0.001
-BATCH_SIZE = 2048
-PRED_BATCH_SIZE = 32768
-MAX_EPOCHS = 10
 
 random.seed(SEED)
@@ -25,197 +37,408 @@
 torch.set_num_threads(min(8, os.cpu_count() or 1))
... 546 more diff lines
```

**#1 -> #3**  (+419 / -157 lines)

```diff
--- iter_1.py
+++ iter_3.py
@@ -3,20 +3,24 @@
 import json
 import random
+import gc
 import numpy as np
 import torch
 import torch.nn as nn
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
 from pipeline.evaluate import evaluate
 
-
-START_TIME = time.time()
-SEED = 2024
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-K = 16
-LR = 0.001
-BATCH_SIZE = 2048
+START = time.time()
+SEED = 7319
+FIELDS = [
+    "user_id", "video_id", "author_id", "tab", "duration_bucket",
+    "tag", "upload_type", "music_type", "hour"
+]
+EMBED_DIM = 8
+BATCH_SIZE = 4096
 PRED_BATCH_SIZE = 32768
-MAX_EPOCHS = 10
+TORCH_EPOCHS = 7
+HALF_LIFE_DAYS = 10.0
 
 random.seed(SEED)
@@ -25,149 +29,242 @@
 torch.set_num_threads(min(8, os.cpu_count() or 1))
 
-
-def make_offsets():
... 604 more diff lines
```

**#1 -> #4**  (+439 / -163 lines)

```diff
--- iter_1.py
+++ iter_4.py
@@ -3,7 +3,9 @@
 import json
 import random
+import gc
 import numpy as np
 import torch
 import torch.nn as nn
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -11,13 +13,6 @@
 
 
-START_TIME = time.time()
-SEED = 2024
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-K = 16
-LR = 0.001
-BATCH_SIZE = 2048
-PRED_BATCH_SIZE = 32768
-MAX_EPOCHS = 10
-
+START = time.time()
+SEED = 731
 random.seed(SEED)
 np.random.seed(SEED)
@@ -25,149 +20,304 @@
 torch.set_num_threads(min(8, os.cpu_count() or 1))
 
-
-def make_offsets():
-    offsets = []
-    total = 0
-    for name in FIELDS:
-        offsets.append(total)
-        total += int(FEATURE_CARDINALITIES[name])
-    return np.asarray(offsets, dtype=np.int64), total
-
... 618 more diff lines
```

**#1 -> #5**  (+394 / -167 lines)

```diff
--- iter_1.py
+++ iter_5.py
@@ -10,13 +10,12 @@
 from pipeline.evaluate import evaluate
 
-
-START_TIME = time.time()
-SEED = 2024
+START = time.time()
+SEED = 7319
 FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-K = 16
-LR = 0.001
-BATCH_SIZE = 2048
-PRED_BATCH_SIZE = 32768
-MAX_EPOCHS = 10
+BATCH = 8192
+PRED_BATCH = 32768
+EPOCHS = 3
+DIN_EPOCHS = 5
+HIST_K = 20
 
 random.seed(SEED)
@@ -25,43 +24,223 @@
 torch.set_num_threads(min(8, os.cpu_count() or 1))
 
-
-def make_offsets():
-    offsets = []
-    total = 0
-    for name in FIELDS:
-        offsets.append(total)
-        total += int(FEATURE_CARDINALITIES[name])
-    return np.asarray(offsets, dtype=np.int64), total
-
-
-OFFSETS, TOTAL_CARDINALITY = make_offsets()
-
-
-def make_matrix(split):
... 565 more diff lines
```

**#1 -> #6**  (+350 / -179 lines)

```diff
--- iter_1.py
+++ iter_6.py
@@ -7,16 +7,12 @@
 import torch.nn as nn
 
-from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.data import load
 from pipeline.evaluate import evaluate
 
-
-START_TIME = time.time()
-SEED = 2024
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-K = 16
-LR = 0.001
-BATCH_SIZE = 2048
-PRED_BATCH_SIZE = 32768
-MAX_EPOCHS = 10
+START = time.time()
+SEED = 7319
+BATCH = 8192
+EPOCHS = 4
+PRED_BATCH = 65536
 
 random.seed(SEED)
@@ -26,83 +22,175 @@
 
 
-def make_offsets():
-    offsets = []
-    total = 0
-    for name in FIELDS:
-        offsets.append(total)
-        total += int(FEATURE_CARDINALITIES[name])
-    return np.asarray(offsets, dtype=np.int64), total
-
-
-OFFSETS, TOTAL_CARDINALITY = make_offsets()
-
-
... 530 more diff lines
```

**#6 -> #7**  (+383 / -338 lines)

```diff
--- iter_6.py
+++ iter_7.py
@@ -2,35 +2,45 @@
 import time
 import json
-import random
 import numpy as np
-import torch
-import torch.nn as nn
-
-from pipeline.data import load
+
+from pipeline.data import load, FEATURE_CARDINALITIES
 from pipeline.evaluate import evaluate
 
 START = time.time()
-SEED = 7319
-BATCH = 8192
-EPOCHS = 4
-PRED_BATCH = 65536
-
-random.seed(SEED)
-np.random.seed(SEED)
-torch.manual_seed(SEED)
-torch.set_num_threads(min(8, os.cpu_count() or 1))
-
-
-def sequence_features(s):
-    """Safe features determined solely by the impressions in the scored split."""
-    user = np.asarray(s.user_id, dtype=np.int64)
-    tm = np.asarray(s.time_ms, dtype=np.int64)
-    n = user.size
+EPS = 1e-5
+
+ENTITY_FIELDS = [
+    "video_id",
+    "author_id",
+    "tag",
+    "onehot_feat3",
+    "duration_bucket",
... 729 more diff lines
```

**#6 -> #8**  (+295 / -318 lines)

```diff
--- iter_6.py
+++ iter_8.py
@@ -7,11 +7,11 @@
 import torch.nn as nn
 
-from pipeline.data import load
+from pipeline.data import load, FEATURE_CARDINALITIES
 from pipeline.evaluate import evaluate
 
 START = time.time()
-SEED = 7319
-BATCH = 8192
-EPOCHS = 4
+SEED = 18473
+BATCH = 6144
+EPOCHS = 3
 PRED_BATCH = 65536
 
@@ -21,353 +21,330 @@
 torch.set_num_threads(min(8, os.cpu_count() or 1))
 
-
-def sequence_features(s):
-    """Safe features determined solely by the impressions in the scored split."""
-    user = np.asarray(s.user_id, dtype=np.int64)
-    tm = np.asarray(s.time_ms, dtype=np.int64)
-    n = user.size
-    row = np.arange(n, dtype=np.int64)
-
-    # Ties follow original row position, as specified by the API.
-    order = np.lexsort((row, tm, user))
-    us = user[order]
-    ts = tm[order]
-
-    new_user = np.empty(n, dtype=bool)
-    new_user[0] = True
-    new_user[1:] = us[1:] != us[:-1]
-    starts = np.flatnonzero(new_user)
-    counts = np.diff(np.r_[starts, n])
-    start_for_row = np.repeat(starts, counts)
... 638 more diff lines
```

**#6 -> #9**  (+400 / -305 lines)

```diff
--- iter_6.py
+++ iter_9.py
@@ -6,14 +6,11 @@
 import torch
 import torch.nn as nn
-
-from pipeline.data import load
+import lightgbm as lgb
+
+from pipeline.data import load, FEATURE_CARDINALITIES
 from pipeline.evaluate import evaluate
 
 START = time.time()
-SEED = 7319
-BATCH = 8192
-EPOCHS = 4
-PRED_BATCH = 65536
-
+SEED = 84631
 random.seed(SEED)
 np.random.seed(SEED)
@@ -21,107 +18,30 @@
 torch.set_num_threads(min(8, os.cpu_count() or 1))
 
-
-def sequence_features(s):
-    """Safe features determined solely by the impressions in the scored split."""
-    user = np.asarray(s.user_id, dtype=np.int64)
-    tm = np.asarray(s.time_ms, dtype=np.int64)
-    n = user.size
-    row = np.arange(n, dtype=np.int64)
-
-    # Ties follow original row position, as specified by the API.
-    order = np.lexsort((row, tm, user))
-    us = user[order]
-    ts = tm[order]
-
-    new_user = np.empty(n, dtype=bool)
-    new_user[0] = True
-    new_user[1:] = us[1:] != us[:-1]
... 740 more diff lines
```

**#6 -> #10**  (+402 / -314 lines)

```diff
--- iter_6.py
+++ iter_10.py
@@ -6,13 +6,17 @@
 import torch
 import torch.nn as nn
-
-from pipeline.data import load
+import torch.nn.functional as F
+
+from pipeline.data import load, FEATURE_CARDINALITIES
 from pipeline.evaluate import evaluate
 
 START = time.time()
-SEED = 7319
-BATCH = 8192
-EPOCHS = 4
-PRED_BATCH = 65536
+SEED = 9143
+EMBED_DIM = 12
+HISTORY_LEN = 8
+EPOCHS = 3
+BATCH_SIZE = 8192
+PRED_BATCH = 32768
+HALF_LIFE_DAYS = 5.0
 
 random.seed(SEED)
@@ -21,16 +25,65 @@
 torch.set_num_threads(min(8, os.cpu_count() or 1))
 
-
-def sequence_features(s):
-    """Safe features determined solely by the impressions in the scored split."""
-    user = np.asarray(s.user_id, dtype=np.int64)
-    tm = np.asarray(s.time_ms, dtype=np.int64)
+FIELDS = [
+    "user_id",
+    "video_id",
+    "author_id",
+    "tab",
+    "tag",
... 747 more diff lines
```

**#6 -> #11**  (+397 / -253 lines)

```diff
--- iter_6.py
+++ iter_11.py
@@ -6,13 +6,14 @@
 import torch
 import torch.nn as nn
-
-from pipeline.data import load
+import torch.nn.functional as F
+
+from pipeline.data import load, FEATURE_CARDINALITIES
 from pipeline.evaluate import evaluate
 
 START = time.time()
-SEED = 7319
+SEED = 19427
 BATCH = 8192
-EPOCHS = 4
 PRED_BATCH = 65536
+EPOCHS = 2
 
 random.seed(SEED)
@@ -21,7 +22,18 @@
 torch.set_num_threads(min(8, os.cpu_count() or 1))
 
-
-def sequence_features(s):
-    """Safe features determined solely by the impressions in the scored split."""
+CAT_FIELDS = [
+    "video_id",
+    "author_id",
+    "tab",
+    "duration_bucket",
+    "tag",
+    "upload_type",
+    "music_type",
+    "hour",
+    "onehot_feat3",
+]
+
+
... 739 more diff lines
```

**#11 -> #12**  (+450 / -478 lines)

```diff
--- iter_11.py
+++ iter_12.py
@@ -2,9 +2,7 @@
 import time
 import json
-import random
 import numpy as np
-import torch
-import torch.nn as nn
-import torch.nn.functional as F
+from scipy import sparse
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -12,317 +10,269 @@
 
 START = time.time()
-SEED = 19427
-BATCH = 8192
-PRED_BATCH = 65536
-EPOCHS = 2
-
-random.seed(SEED)
+SEED = 28411
+RANK = 32
+BLEND_WEIGHTS = (0.04, 0.07, 0.10, 0.14, 0.19, 0.25, 0.32, 0.40)
+
 np.random.seed(SEED)
-torch.manual_seed(SEED)
-torch.set_num_threads(min(8, os.cpu_count() or 1))
-
-CAT_FIELDS = [
-    "video_id",
-    "author_id",
-    "tab",
-    "duration_bucket",
-    "tag",
-    "upload_type",
-    "music_type",
-    "hour",
... 947 more diff lines
```

**#11 -> #13**  (+401 / -392 lines)

```diff
--- iter_11.py
+++ iter_13.py
@@ -12,5 +12,5 @@
 
 START = time.time()
-SEED = 19427
+SEED = 28471
 BATCH = 8192
 PRED_BATCH = 65536
@@ -23,4 +23,5 @@
 
 CAT_FIELDS = [
+    "user_id",
     "video_id",
     "author_id",
@@ -32,66 +33,32 @@
     "hour",
     "onehot_feat3",
+    "onehot_feat8",
+    "user_active_degree",
 ]
 
+NUM_FIELDS = [
+    "duration_ms",
+    "user_fans_user_num",
+    "user_follow_user_num",
+    "user_friend_user_num",
+    "user_register_days",
+]
+
 
 def make_features(s):
-    user = np.asarray(s.user_id, dtype=np.int64)
-    tm = np.asarray(s.time_ms, dtype=np.int64)
-    n = user.size
-    row = np.arange(n, dtype=np.int64)
-
-    order = np.lexsort((row, tm, user))
-    us = user[order]
-    ts = tm[order]
... 864 more diff lines
```

**#11 -> #14**  (+396 / -387 lines)

```diff
--- iter_11.py
+++ iter_14.py
@@ -7,4 +7,5 @@
 import torch.nn as nn
 import torch.nn.functional as F
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -12,9 +13,5 @@
 
 START = time.time()
-SEED = 19427
-BATCH = 8192
-PRED_BATCH = 65536
-EPOCHS = 2
-
+SEED = 73129
 random.seed(SEED)
 np.random.seed(SEED)
@@ -23,17 +20,22 @@
 
 CAT_FIELDS = [
-    "video_id",
-    "author_id",
-    "tab",
-    "duration_bucket",
-    "tag",
-    "upload_type",
-    "music_type",
-    "hour",
-    "onehot_feat3",
+    "user_id", "video_id", "author_id", "tab", "tag",
+    "duration_bucket", "upload_type", "music_type", "video_type",
+    "onehot_feat3", "onehot_feat7", "onehot_feat8",
+    "user_active_degree", "fans_user_num_range",
+    "follow_user_num_range", "friend_user_num_range",
+    "register_days_bucket", "is_video_author", "hour",
 ]
 
-
... 869 more diff lines
```

**#11 -> #15**  (+359 / -442 lines)

```diff
--- iter_11.py
+++ iter_15.py
@@ -2,5 +2,6 @@
 import time
 import json
-import random
+from datetime import datetime
+
 import numpy as np
 import torch
@@ -12,15 +13,14 @@
 
 START = time.time()
-SEED = 19427
-BATCH = 8192
+SEED = 73129
+BATCH = 16384
 PRED_BATCH = 65536
 EPOCHS = 2
 
-random.seed(SEED)
 np.random.seed(SEED)
 torch.manual_seed(SEED)
 torch.set_num_threads(min(8, os.cpu_count() or 1))
 
-CAT_FIELDS = [
+BASE_FIELDS = [
     "video_id",
     "author_id",
@@ -34,295 +34,43 @@
 ]
 
-
-def make_features(s):
-    user = np.asarray(s.user_id, dtype=np.int64)
-    tm = np.asarray(s.time_ms, dtype=np.int64)
-    n = user.size
-    row = np.arange(n, dtype=np.int64)
-
-    order = np.lexsort((row, tm, user))
... 844 more diff lines
```

**#11 -> #16**  (+361 / -274 lines)

```diff
--- iter_11.py
+++ iter_16.py
@@ -12,8 +12,9 @@
 
 START = time.time()
-SEED = 19427
+SEED = 28183
 BATCH = 8192
 PRED_BATCH = 65536
-EPOCHS = 2
+EPOCHS = 1
+HALF_LIFE_DAYS = 10.0
 
 random.seed(SEED)
@@ -23,4 +24,5 @@
 
 CAT_FIELDS = [
+    "user_id",
     "video_id",
     "author_id",
@@ -35,4 +37,15 @@
 
 
+def date_to_ordinal(date):
+    date = np.asarray(date, dtype=np.int64)
+    year = date // 10000
+    month = (date // 100) % 100
+    day = date % 100
+
+    # All benchmark dates are in one spring month range, but this expression
+    # also remains monotone if a month boundary is crossed.
+    return (year * 372 + month * 31 + day).astype(np.float32)
+
+
 def make_features(s):
     user = np.asarray(s.user_id, dtype=np.int64)
@@ -50,7 +63,7 @@
     starts = np.flatnonzero(new_user)
     counts = np.diff(np.r_[starts, n])
-    start_sorted = np.repeat(starts, counts)
... 787 more diff lines
```

**#14 -> #17**  (+453 / -468 lines)

```diff
--- iter_14.py
+++ iter_17.py
@@ -7,5 +7,4 @@
 import torch.nn as nn
 import torch.nn.functional as F
-import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -13,5 +12,5 @@
 
 START = time.time()
-SEED = 73129
+SEED = 91427
 random.seed(SEED)
 np.random.seed(SEED)
@@ -19,354 +18,357 @@
 torch.set_num_threads(min(8, os.cpu_count() or 1))
 
-CAT_FIELDS = [
-    "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "music_type", "video_type",
-    "onehot_feat3", "onehot_feat7", "onehot_feat8",
-    "user_active_degree", "fans_user_num_range",
-    "follow_user_num_range", "friend_user_num_range",
-    "register_days_bucket", "is_video_author", "hour",
+FIELDS = [
+    "user_id",
+    "video_id",
+    "author_id",
+    "tab",
+    "tag",
+    "duration_bucket",
+    "upload_type",
+    "music_type",
+    "onehot_feat3",
+    "onehot_feat8",
+    "user_active_degree",
+    "hour",
 ]
-
... 952 more diff lines
```

**#14 -> #18**  (+372 / -441 lines)

```diff
--- iter_14.py
+++ iter_18.py
@@ -7,5 +7,4 @@
 import torch.nn as nn
 import torch.nn.functional as F
-import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -13,5 +12,5 @@
 
 START = time.time()
-SEED = 73129
+SEED = 94037
 random.seed(SEED)
 np.random.seed(SEED)
@@ -19,302 +18,246 @@
 torch.set_num_threads(min(8, os.cpu_count() or 1))
 
-CAT_FIELDS = [
-    "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "music_type", "video_type",
-    "onehot_feat3", "onehot_feat7", "onehot_feat8",
-    "user_active_degree", "fans_user_num_range",
-    "follow_user_num_range", "friend_user_num_range",
-    "register_days_bucket", "is_video_author", "hour",
+FIELDS = [
+    "user_id",
+    "video_id",
+    "author_id",
+    "tag",
+    "tab",
+    "duration_bucket",
+    "upload_type",
+    "onehot_feat3",
+    "onehot_feat8",
 ]
-
-NUM_FIELDS = [
-    "duration_ms",
-    "user_fans_user_num",
... 859 more diff lines
```

**#14 -> #19**  (+330 / -461 lines)

```diff
--- iter_14.py
+++ iter_19.py
@@ -4,7 +4,4 @@
 import random
 import numpy as np
-import torch
-import torch.nn as nn
-import torch.nn.functional as F
 import lightgbm as lgb
 
@@ -13,17 +10,15 @@
 
 START = time.time()
-SEED = 73129
+SEED = 41773
 random.seed(SEED)
 np.random.seed(SEED)
-torch.manual_seed(SEED)
-torch.set_num_threads(min(8, os.cpu_count() or 1))
 
 CAT_FIELDS = [
-    "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "music_type", "video_type",
-    "onehot_feat3", "onehot_feat7", "onehot_feat8",
-    "user_active_degree", "fans_user_num_range",
-    "follow_user_num_range", "friend_user_num_range",
-    "register_days_bucket", "is_video_author", "hour",
+    "video_id", "author_id", "tab", "tag", "duration_bucket",
+    "upload_type", "music_type", "video_type", "onehot_feat3",
+    "onehot_feat7", "onehot_feat8", "user_active_degree",
+    "fans_user_num_range", "follow_user_num_range",
+    "friend_user_num_range", "register_days_bucket",
+    "is_video_author", "hour",
 ]
 
@@ -36,305 +31,198 @@
 ]
 
-
-def sequential_features(s):
... 822 more diff lines
```

**#12 -> #20**  (+365 / -410 lines)

```diff
--- iter_12.py
+++ iter_20.py
@@ -3,6 +3,5 @@
 import json
 import numpy as np
-from scipy import sparse
-from scipy.sparse.linalg import svds
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -10,277 +9,54 @@
 
 START = time.time()
-SEED = 28411
-RANK = 32
-BLEND_WEIGHTS = (0.04, 0.07, 0.10, 0.14, 0.19, 0.25, 0.32, 0.40)
-
+SEED = 73129
 np.random.seed(SEED)
 
-
-def arrays_from_splits(splits, field, need_y=True):
-    users = np.concatenate([
+NB_FIELDS = [
+    "video_id", "author_id", "tab", "tag", "duration_bucket",
+    "upload_type", "music_type", "video_type", "onehot_feat1",
+    "onehot_feat3", "onehot_feat7", "onehot_feat8",
+]
+
+LGB_FIELDS = [
+    "user_id", "video_id", "author_id", "tab", "tag",
+    "duration_bucket", "upload_type", "music_type", "video_type",
+    "hour", "user_active_degree", "fans_user_num_range",
+    "follow_user_num_range", "friend_user_num_range",
+    "register_days_bucket", "onehot_feat1", "onehot_feat3",
+    "onehot_feat7", "onehot_feat8",
+]
+
+NUM_FIELDS = [
+    "duration_ms", "user_fans_user_num", "user_follow_user_num",
... 818 more diff lines
```

**#14 -> #21**  (+294 / -495 lines)

```diff
--- iter_14.py
+++ iter_21.py
@@ -2,10 +2,7 @@
 import time
 import json
-import random
+import gc
 import numpy as np
-import torch
-import torch.nn as nn
-import torch.nn.functional as F
-import lightgbm as lgb
+import scipy.sparse as sp
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -13,337 +10,135 @@
 
 START = time.time()
-SEED = 73129
-random.seed(SEED)
-np.random.seed(SEED)
-torch.manual_seed(SEED)
-torch.set_num_threads(min(8, os.cpu_count() or 1))
-
-CAT_FIELDS = [
-    "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "music_type", "video_type",
-    "onehot_feat3", "onehot_feat7", "onehot_feat8",
-    "user_active_degree", "fans_user_num_range",
-    "follow_user_num_range", "friend_user_num_range",
-    "register_days_bucket", "is_video_author", "hour",
-]
-
-NUM_FIELDS = [
-    "duration_ms",
-    "user_fans_user_num",
-    "user_follow_user_num",
-    "user_friend_user_num",
-    "user_register_days",
-]
... 798 more diff lines
```

## Search tree

Each line is an executed script; indentation is the edit it was derived from. A node marked `[retired]` produced three children that failed to improve on it, so the search abandoned that branch and backtracked.

```
#1 0.6015  Reproduce the official baseline stage by training a rank-16 categorica
  #2 0.6037  Target prediction formation and fusion by comparing an expanded pairwi
  #3 0.6040  Target prediction formation and temporal-drift handling by comparing r
  #4 0.6041  Target prediction formation and drift-aware training by comparing rece
  #5 0.6042  Target prediction formation and temporal-interest representation by co
  #6 0.6042 [retired]  Target the logged-impression context and post-model fusion stages: com
    #7 0.6043  Target leakage-free non-parametric personalization and score-fusion st
    #9 0.6044  Target the ranking-loss and prediction-formation stages by comparing t
    #10 0.6044  Target prediction formation under temporal drift by comparing DCN cros
    #11 0.6050 [retired]  Target the loss and prediction-formation stages with pointwise-anchore
      #12 0.6051  Target the personalization and prediction-formation stages with low-ra
        #20 0.6057  Target prediction formation with a breadth comparison of stationary ca
      #13 0.6054  Target train-only auxiliary-supervision and prediction formation by co
      #14 0.6055 [retired]  Target the structured-loss stage by comparing setwise ListNet additive
        #17 0.6057  I am testing prediction formation across explicit high-order xDeepFM i
        #18 0.6057  Target list-context prediction formation under activity drift by compa
        #19 0.6057  Target drift-robust prediction formation by comparing a recency-weight
        #21 0.6057  Target memory-based collaborative prediction and fusion by comparing i
      #15 0.6055  Target temporal-drift and prediction-formation stages by comparing poo
      #16 0.6055  Target prediction formation and loss isolation by training pointwise,
```

## What the agent established

The agent's belief set, revised after every scored iteration rather than appended to, so later evidence can demote an earlier conclusion instead of piling up beside it. A claim marked `(invalidated)` was contradicted by a later result. Machine-readable form with per-claim evidence in `knowledge.json`.

```
- (active) A rank-16 categorical Factorization Machine trained with Adam at lr=0.001 on user_id, video_id, author_id, tab, and duration_bucket reproduces the official baseline within seed noise (0.6015 versus 0.6016), so it has no measurable gain. [iters 1]
- (active) Accepted bundled experiments in iterations 2-7 and 9-21 scored 0.6037-0.6057, at least 0.0022 above the implemented FM baseline, but their bundled designs do not identify the causal component. [iters 2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21]
- (active) No accepted experiment has exceeded another by more than 0.002: iterations 17-21 all scored 0.6057, only 0.0002 above the prior 0.6055 best, so no tested pipeline has demonstrated superiority under the gain criterion. [iters 2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21]
- (active) Iteration 8 was rejected and therefore provides no metric evidence about the proposed shared-bottom multitask approach. [iters 8]
Ruled out by evidence (do not revisit without a new mechanism):
- (invalidated) The identity of the distinct-ranking lineage is not stable: slot 0 was isolated in iterations 12-16, whereas in iterations 17-21 slot 3 is isolated (correlations 0.064-0.090) and slots 0-2 are substantially more aligned (0.545-0.684). [iters 12,13,14,15,16,17,18,19,20,21]
```

## Alternatives compared inside iterations

696 candidate solutions were built and scored across 19 iteration(s). The convergence rule charges one iteration per experiment, so searching inside an iteration buys comparisons the iteration budget cannot.

| iteration | candidates (validation primary) |
|---|---|
| #2 | deepfm_blend_0.25 0.6022, deepfm_blend_0.40 0.6030, deepfm_blend_0.55 0.6031, deepfm_blend_0.70 0.6031, deepfm_blend_0.85 0.6030, deepfm_standalone 0.6028, empirical_bayes_blend_0.25 0.6003, empirical_bayes_blend_0.40 0.5983 |
| #3 | lightgbm_rank_a0.25 0.6023, lightgbm_rank_a0.40 0.6028, lightgbm_rank_a0.55 0.6034, lightgbm_rank_a0.70 0.6028, lightgbm_rank_a0.85 0.6029, lightgbm_rank_a1.00 0.6029, lightgbm_standalone 0.6028, lightgbm_z_a0.25 0.6027 |
| #4 | binary_lgb_recency_blend_0.20 0.6020, binary_lgb_recency_blend_0.35 0.6025, binary_lgb_recency_blend_0.50 0.6024, binary_lgb_recency_blend_0.65 0.6024, binary_lgb_recency_blend_0.80 0.6019, binary_lgb_recency_raw 0.6016, lambdarank_user_grouped_blend_0.20 0.6024, lambdarank_user_grouped_blend_0.35 0.6027 |
| #5 | din_blend 0.6021, din_raw 0.5954, latent_mf_blend 0.6020, latent_mf_raw 0.5942, mmoe_blend 0.6027, mmoe_raw 0.6018 |
| #6 | position_histogram 0.5412, position_histogram_blend_0.08 0.6016, position_histogram_blend_0.14 0.6016, position_histogram_blend_0.20 0.6016, position_histogram_blend_0.28 0.6006, position_histogram_blend_0.36 0.5991, position_histogram_blend_0.45 0.5970, position_linear 0.5001 |
| #7 | adaptive_memory_h2p5 0.5810, adaptive_memory_h2p5_rankblend_0.10 0.6041, adaptive_memory_h2p5_rankblend_0.20 0.6037, adaptive_memory_h2p5_rankblend_0.30 0.6025, adaptive_memory_h2p5_rankblend_0.40 0.5999, adaptive_memory_h2p5_zblend_0.10 0.6043, adaptive_memory_h2p5_zblend_0.18 0.6041, adaptive_memory_h2p5_zblend_0.26 0.6039 |
| #9 | lambda_top5 0.5909, lambda_top5_rankblend_0.10 0.6041, lambda_top5_rankblend_0.18 0.6039, lambda_top5_rankblend_0.26 0.6036, lambda_top5_rankblend_0.34 0.6032, lambda_top5_rankblend_0.44 0.6026, lambda_top5_rankblend_0.56 0.5975, lambda_top5_scoreblend_0.10 0.6043 |
| #10 | dcn_blend_0.08 0.6044, dcn_blend_0.14 0.6044, dcn_blend_0.20 0.6044, dcn_blend_0.28 0.6044, dcn_blend_0.36 0.6044, dcn_blend_0.45 0.6044, dcn_blend_0.55 0.6043, dcn_standalone 0.6029 |
| #11 | incumbent 0.6042, pairwise_deep 0.5967, pairwise_deep_rankblend_0.08 0.6041, pairwise_deep_rankblend_0.14 0.6043, pairwise_deep_rankblend_0.20 0.6044, pairwise_deep_rankblend_0.28 0.6044, pairwise_deep_rankblend_0.36 0.6037, pairwise_deep_rankblend_0.45 0.6032 |
| #12 | author_residual_svd 0.5779, author_residual_svd_rankblend_0.04 0.6050, author_residual_svd_rankblend_0.07 0.6050, author_residual_svd_rankblend_0.10 0.6049, author_residual_svd_rankblend_0.14 0.6049, author_residual_svd_rankblend_0.19 0.6048, author_residual_svd_rankblend_0.25 0.6040, author_residual_svd_rankblend_0.32 0.6031 |
| #13 | incumbent 0.6050, mmoe 0.6024, mmoe_rankblend_0.08 0.6050, mmoe_rankblend_0.14 0.6050, mmoe_rankblend_0.20 0.6050, mmoe_rankblend_0.28 0.6051, mmoe_rankblend_0.36 0.6047, mmoe_rankblend_0.45 0.6047 |
| #14 | incumbent 0.6050, listnet_additive 0.6008, listnet_additive_rankblend_0.10 0.6050, listnet_additive_rankblend_0.18 0.6051, listnet_additive_rankblend_0.26 0.6053, listnet_additive_rankblend_0.34 0.6049, listnet_additive_rankblend_0.42 0.6048, listnet_additive_rankblend_0.50 0.6037 |
| #15 | cross_wide 0.5917, cross_wide_rankblend_0.05 0.6050, cross_wide_rankblend_0.10 0.6051, cross_wide_rankblend_0.15 0.6050, cross_wide_rankblend_0.22 0.6046, cross_wide_rankblend_0.30 0.6045, cross_wide_rankblend_0.40 0.6034, cross_wide_rankblend_0.52 0.5978 |
| #16 | incumbent 0.6050, pointwise_autoint 0.5980, pointwise_autoint_rankblend_0.08 0.6050, pointwise_autoint_rankblend_0.14 0.6050, pointwise_autoint_rankblend_0.20 0.6050, pointwise_autoint_rankblend_0.28 0.6050, pointwise_autoint_rankblend_0.36 0.6048, pointwise_autoint_rankblend_0.45 0.6048 |
| #17 | autoint_rankblend_0.08 0.6056, autoint_rankblend_0.14 0.6056, autoint_rankblend_0.20 0.6056, autoint_rankblend_0.28 0.6055, autoint_rankblend_0.36 0.6053, autoint_rankblend_0.46 0.6053, autoint_raw 0.6036, autoint_zblend_0.08 0.6056 |
| #18 | bigru 0.5996, bigru_rankblend_0.10 0.6056, bigru_rankblend_0.20 0.6057, bigru_rankblend_0.30 0.6056, bigru_rankblend_0.40 0.6050, bigru_rankblend_0.50 0.6037, deepsets 0.6012, deepsets_rankblend_0.10 0.6056 |
| #19 | categorical_random_forest 0.5899, categorical_random_forest_best_blend 0.6056, generative_additive 0.5919, generative_additive_best_blend 0.6056, incumbent 0.6055, pair_empirical_bayes 0.5911, pair_empirical_bayes_best_blend 0.6056 |
| #20 | association_rules 0.4918, association_rules_rankblend_0.05 0.6056, association_rules_rankblend_0.10 0.6055, association_rules_rankblend_0.16 0.6049, association_rules_rankblend_0.23 0.6020, association_rules_rankblend_0.31 0.5972, association_rules_rankblend_0.40 0.5848, association_rules_rankblend_0.50 0.5652 |
| #21 | incumbent 0.6055, knn_cosine 0.5536, knn_cosine_rankblend_0.05 0.6056, knn_cosine_rankblend_0.10 0.6055, knn_cosine_rankblend_0.18 0.6051, knn_cosine_rankblend_0.26 0.6029, knn_cosine_rankblend_0.34 0.5991, knn_cosine_rankblend_0.44 0.5956 |

## Resource usage (Feasibility & Practicality)

- **Agent wall-clock to converged result: 0.55 h (33 min)**
- Total LLM tokens: **429,394** (289,545 in / 139,849 out), including the knowledge-revision stage
- Iterations used: **22 of 50** (20 accepted scores, 0 failed, 1 rejected)
- Compute inside generated scripts: **0.29 h (18 min)** on CPU.
- Mean tokens per iteration: 19,518
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

Across 38 development runs of this agent, 303 iterations were executed and 28 failed. Every one was handled in-loop; none was escalated to a human. Failure taxonomy:

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

- Best validation primary: **0.6057** (baseline 0.6016, delta +0.0041)
- From iteration #21: Target memory-based collaborative prediction and fusion by comparing item-item cosine neighborhoods, resource-allocation diffusion, and positive-PMI association profiles; these form scores from a user’s train-only positive neighborhood rather than learned identity parameters, potentially remaining complementary under temporal drift and improving the incumbent through within-user rank blending.
- Selection is on validation only, per the scoring rules; the hidden test set was never used to choose between iterations.
- Submission: written to `runs/r89_5slot/submission.csv`

### Results table

| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| validation (best iteration) | 0.6730 | 0.5385 | 0.6057 |
| official baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
| hidden test (this submission) | 0.6680 | 0.5325 | **0.6003** |
| official baseline (test) | 0.6610 | 0.5282 | 0.5946 |

**Absolute delta over baseline on hidden test: GAUC +0.0070, nDCG@5 +0.0043, mean +0.0057** (primary +0.0057).

Per the scoring formula, delta(m) = score_agent(m) - score_baseline(m), averaged over metrics.
