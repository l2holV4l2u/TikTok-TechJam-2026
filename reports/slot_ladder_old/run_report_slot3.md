# Run report - r87_3slot

> 1 scored iteration(s) were rejected by the integrity critic. They are excluded from the search tree, cross-run memory, and submission.

## Loop stages executed by the agent

- **Inspect data (EDA):** completed at iteration #0 - the agent wrote and ran its own exploratory script; its findings are in `eda_report.txt` and were carried into every later prompt.
- **Reproduce official baseline (Requirement 1):** 1 attempt(s); the agent's own pipeline reached validation primary **0.6023** against the official 0.6016 (delta +0.0007, inside the baseline's 5-seed noise). That script became the root of the search tree.
- **Iterate:** 12 experiments proposed, executed and evaluated by the agent, each branching from a node of its own search tree.

## Iteration log

A turn is one hypothesis-to-score cycle and advances several lineages at once, so `turn` and `slot` say when an experiment ran and which lineage produced it. The convergence rule is measured against turns; the 50-cap counts scripts.

| # | turn | slot | phase | parent | status | secs | primary | vs baseline | hypothesis |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | - | eda | - | ok | 1 | n/a | - | INSPECT DATA stage: quantify drift, sparsity, overlap, categorical lab |
| 1 | 0 | - | baseline | - | ok | 6 | 0.6023 | +0.0007 | Reproduce the official baseline at the modeling stage using a k=16 Fac |
| 2 | 1 | 0 | improve | #1 | kept | 202 | 0.6042 | +0.0026 | Target prediction formation and fusion by comparing field-aware factor |
| 3 | 1 | 1 | improve | #1 | kept | 38 | 0.6042 | +0.0026 | Target the prediction and training-loss stages by comparing an additiv |
| 4 | 1 | 2 | improve | #1 | rejected | 41 | 0.6043 | +0.0027 | Target sequence modeling, multi-task supervision, and latent preferenc |
| 5 | 2 | 0 | improve | #3 | ok | 12 | 0.6043 | +0.0027 | Target the training-loss and prediction-formation stages by comparing  |
| 6 | 2 | 1 | improve | #3 | kept | 45 | 0.6049 | +0.0033 | Target prediction formation and temporal-drift training by comparing e |
| 7 | 2 | 2 | improve | #3 | kept | 42 | 0.6052 | +0.0036 | Target multi-task representation learning by comparing a single-task M |
| 8 | 3 | 0 | improve | #7 | kept | 8 | 0.6052 | +0.0036 | Target the drift-sensitive history representation and prediction stage |
| 9 | 3 | 1 | improve | #7 | kept | 63 | 0.6055 | +0.0039 | Target prediction formation and score aggregation by comparing PNN pai |
| 10 | 3 | 2 | improve | #7 | kept | 69 | 0.6055 | +0.0039 | Target prediction formation with three structurally distinct families— |
| 11 | 4 | 0 | improve | #9 | kept | 2 | 0.6057 | +0.0041 | Target drift-sensitive prediction formation by comparing temporal extr |
| 12 | 4 | 1 | improve | #9 | kept | 43 | 0.6057 | +0.0041 | Target prediction formation and drift-aware training by comparing fiel |
| 13 | 4 | 2 | improve | #9 | kept | 22 | 0.6057 | +0.0041 | Target prediction formation with two structurally different, underexpl |

## Portfolio

- Lineages advanced per turn: **3**
- Turns: **4**  ·  scripts executed: **14** of 50
- A turn is one hypothesis-to-score cycle -- Figure 1's iteration, and what the convergence rule is measured against. Scripts are reported separately so the run can be checked against the stricter reading of the 50-iteration cap as well.
- Lineages archived: **3**  ·  revived from the archive: **0**

### Did the lineages actually disagree?

Within-user rank correlation between the live slots' validation predictions. This is the acceptance test for the whole design: above ~0.95 the extra slots are three copies of one agent and bought nothing.

| turn | scripts | mean corr | max corr | alert |
|---|---|---|---|---|
| 1 | #2, #3, #4 | 0.9140 | 0.9140 | no |
| 2 | #5, #6, #7 | 0.6345 | 0.9095 | no |
| 3 | #8, #9, #10 | 0.7636 | 0.8327 | no |
| 4 | #11, #12, #13 | 0.7405 | 0.7890 | no |

### Lineages retired and replaced

A slot that stops beating its own best is archived, not stopped -- the run's convergence counter is untouched by it. Refill alternates fresh drafts with a revival, and a revival is chosen for score AND for disagreeing with what is already live.

| turn | slot | archived at | choice | detail |
|---|---|---|---|---|
| 2 | 0 | 0.6043 | fresh | new draft |
| 2 | 1 | 0.6049 | fresh | new draft |
| 4 | 2 | 0.6057 | fresh | new draft |

### Controller blend of the portfolio

Every turn the controller blends the trusted incumbent, the live slots and the archive, choosing members on one half of validation and confirming on the other. A blend that wins the selection fold and loses the confirmation fold is refused: it won the split, not the task.

| turn | pool | accepted | members | fold A | fold B | reason |
|---|---|---|---|---|---|---|
| 1 | 2 | no | - | - | - | no member improved fold A |
| 2 | 3 | no | - | - | - | no member improved fold A |
| 3 | 5 | no | - | - | - | no member improved fold A |
| 4 | 5 | no | - | - | - | no member improved fold A |

## Code diff applied, per iteration

Each iteration edits a parent script. This is that edit, reconstructed from the two full sources in the ledger, truncated to the first 40 lines per iteration.


**#1 -> #2**  (+457 / -113 lines)

```diff
--- iter_1.py
+++ iter_2.py
@@ -1,4 +1,5 @@
 import os
 import time
+import json
 import random
 import numpy as np
@@ -11,10 +12,22 @@
 
 START = time.time()
-SEED = 2024
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-K = 16
-LR = 0.001
-BATCH_SIZE = 4096
-EPOCHS = 5
+SEED = 7319
+BATCH_SIZE = 8192
+EPOCHS = 4
+RANK = 8
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
+
+torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
 
 
@@ -25,72 +38,143 @@
 
 
... 616 more diff lines
```

**#1 -> #3**  (+338 / -156 lines)

```diff
--- iter_1.py
+++ iter_3.py
@@ -1,191 +1,368 @@
 import os
 import time
+import json
+import gc
 import random
 import numpy as np
 import torch
 import torch.nn as nn
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
+from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
 
 START = time.time()
-SEED = 2024
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-K = 16
-LR = 0.001
-BATCH_SIZE = 4096
-EPOCHS = 5
-
-
-def seed_everything(seed):
-    random.seed(seed)
-    np.random.seed(seed)
-    torch.manual_seed(seed)
-
-
+SEED = 2718
 torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
-seed_everything(SEED)
-
-cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
-offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
-total_cardinality = int(sum(cardinalities))
... 511 more diff lines
```

**#1 -> #4**  (+361 / -136 lines)

```diff
--- iter_1.py
+++ iter_4.py
@@ -1,4 +1,5 @@
 import os
 import time
+import json
 import random
 import numpy as np
@@ -11,13 +12,18 @@
 
 START = time.time()
-SEED = 2024
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-K = 16
-LR = 0.001
-BATCH_SIZE = 4096
-EPOCHS = 5
-
-
-def seed_everything(seed):
+SEED = 7319
+BATCH = 8192
+EPOCHS = 3
+FIELDS = [
+    "user_id", "video_id", "author_id", "tab", "duration_bucket",
+    "tag", "upload_type", "music_type", "hour",
+]
+EMBED_DIM = 12
+HIST_LEN = 8
+
+torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
+
+
+def seed_all(seed):
     random.seed(seed)
     np.random.seed(seed)
@@ -25,130 +31,269 @@
 
 
-torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
... 515 more diff lines
```

**#3 -> #5**  (+352 / -296 lines)

```diff
--- iter_3.py
+++ iter_5.py
@@ -7,204 +7,261 @@
 import torch
 import torch.nn as nn
-import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
-from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
 
 START = time.time()
-SEED = 2718
-torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
+SEED = 7319
+THREADS = max(1, min(8, os.cpu_count() or 1))
+torch.set_num_threads(THREADS)
 np.random.seed(SEED)
 random.seed(SEED)
 torch.manual_seed(SEED)
 
-CAT_FIELDS = [
-    "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "music_type",
-    "onehot_feat3", "onehot_feat8", "onehot_feat1",
-    "onehot_feat7", "user_active_degree",
-    "register_days_bucket", "register_days_range",
-    "fans_user_num_range", "follow_user_num_range",
-    "friend_user_num_range", "hour", "is_live_streamer",
+LATENT_FIELDS = [
+    "video_id",
+    "author_id",
+    "tab",
+    "duration_bucket",
+    "tag",
 ]
-NUM_FIELDS = [
-    "duration_ms", "user_fans_user_num", "user_follow_user_num",
-    "user_friend_user_num", "user_register_days",
... 680 more diff lines
```

**#3 -> #6**  (+311 / -289 lines)

```diff
--- iter_3.py
+++ iter_6.py
@@ -4,207 +4,255 @@
 import gc
 import random
+import datetime
 import numpy as np
 import torch
 import torch.nn as nn
-import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
-from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
 
 START = time.time()
-SEED = 2718
-torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
+SEED = 7331
+THREADS = max(1, min(8, os.cpu_count() or 1))
+torch.set_num_threads(THREADS)
 np.random.seed(SEED)
 random.seed(SEED)
 torch.manual_seed(SEED)
 
-CAT_FIELDS = [
-    "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "music_type",
-    "onehot_feat3", "onehot_feat8", "onehot_feat1",
-    "onehot_feat7", "user_active_degree",
-    "register_days_bucket", "register_days_range",
-    "fans_user_num_range", "follow_user_num_range",
-    "friend_user_num_range", "hour", "is_live_streamer",
+FIELDS = [
+    "user_id",
+    "video_id",
+    "author_id",
+    "tab",
+    "tag",
... 637 more diff lines
```

**#3 -> #7**  (+371 / -277 lines)

```diff
--- iter_3.py
+++ iter_7.py
@@ -7,14 +7,13 @@
 import torch
 import torch.nn as nn
-import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
-from pipeline.history import historical_features
 from pipeline.evaluate import evaluate
 
 
 START = time.time()
-SEED = 2718
-torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
+SEED = 6187
+THREADS = max(1, min(8, os.cpu_count() or 1))
+torch.set_num_threads(THREADS)
 np.random.seed(SEED)
 random.seed(SEED)
@@ -34,45 +33,24 @@
     "user_friend_user_num", "user_register_days",
 ]
-WIDE_FIELDS = list(CAT_FIELDS)
-WIDE_EPOCHS = 3
-WIDE_BATCH = 8192
-
-
-def make_tree_matrix(split_name, split):
+AUX_KEYS = [
+    "is_click", "is_like", "is_follow",
+    "is_comment", "is_forward", "is_profile_enter",
+]
+
+EMBED_DIM = 8
+BATCH_SIZE = 8192
+EPOCHS = 2
+AUX_WEIGHT = 0.18
+
+cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
... 696 more diff lines
```

**#7 -> #8**  (+458 / -393 lines)

```diff
--- iter_7.py
+++ iter_8.py
@@ -5,6 +5,7 @@
 import random
 import numpy as np
-import torch
-import torch.nn as nn
+import lightgbm as lgb
+from scipy.sparse import coo_matrix
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -13,19 +14,13 @@
 
 START = time.time()
-SEED = 6187
+SEED = 7351
 THREADS = max(1, min(8, os.cpu_count() or 1))
-torch.set_num_threads(THREADS)
 np.random.seed(SEED)
 random.seed(SEED)
-torch.manual_seed(SEED)
 
 CAT_FIELDS = [
     "user_id", "video_id", "author_id", "tab", "tag",
     "duration_bucket", "upload_type", "music_type",
-    "onehot_feat3", "onehot_feat8", "onehot_feat1",
-    "onehot_feat7", "user_active_degree",
-    "register_days_bucket", "register_days_range",
-    "fans_user_num_range", "follow_user_num_range",
-    "friend_user_num_range", "hour", "is_live_streamer",
+    "onehot_feat3", "onehot_feat8",
 ]
 NUM_FIELDS = [
@@ -33,279 +28,340 @@
     "user_friend_user_num", "user_register_days",
 ]
-AUX_KEYS = [
-    "is_click", "is_like", "is_follow",
-    "is_comment", "is_forward", "is_profile_enter",
... 901 more diff lines
```

**#7 -> #9**  (+324 / -278 lines)

```diff
--- iter_7.py
+++ iter_9.py
@@ -13,5 +13,5 @@
 
 START = time.time()
-SEED = 6187
+SEED = 93217
 THREADS = max(1, min(8, os.cpu_count() or 1))
 torch.set_num_threads(THREADS)
@@ -25,15 +25,12 @@
     "onehot_feat3", "onehot_feat8", "onehot_feat1",
     "onehot_feat7", "user_active_degree",
-    "register_days_bucket", "register_days_range",
-    "fans_user_num_range", "follow_user_num_range",
-    "friend_user_num_range", "hour", "is_live_streamer",
+    "register_days_bucket", "fans_user_num_range", "hour",
 ]
 NUM_FIELDS = [
-    "duration_ms", "user_fans_user_num", "user_follow_user_num",
-    "user_friend_user_num", "user_register_days",
-]
-AUX_KEYS = [
-    "is_click", "is_like", "is_follow",
-    "is_comment", "is_forward", "is_profile_enter",
+    "duration_ms",
+    "user_fans_user_num",
+    "user_follow_user_num",
+    "user_friend_user_num",
+    "user_register_days",
 ]
 
@@ -41,5 +38,4 @@
 BATCH_SIZE = 8192
 EPOCHS = 2
-AUX_WEIGHT = 0.18
 
 cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
@@ -47,53 +43,59 @@
 total_cardinality = int(sum(cards))
 n_fields = len(CAT_FIELDS)
... 742 more diff lines
```

**#7 -> #10**  (+472 / -315 lines)

```diff
--- iter_7.py
+++ iter_10.py
@@ -7,4 +7,7 @@
 import torch
 import torch.nn as nn
+import lightgbm as lgb
+from scipy import sparse
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -13,19 +16,19 @@
 
 START = time.time()
-SEED = 6187
+SEED = 7429
 THREADS = max(1, min(8, os.cpu_count() or 1))
-torch.set_num_threads(THREADS)
 np.random.seed(SEED)
 random.seed(SEED)
 torch.manual_seed(SEED)
-
-CAT_FIELDS = [
-    "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "music_type",
-    "onehot_feat3", "onehot_feat8", "onehot_feat1",
-    "onehot_feat7", "user_active_degree",
+torch.set_num_threads(THREADS)
+
+GBDT_CAT_FIELDS = [
+    "video_id", "author_id", "tab", "tag", "duration_bucket",
+    "upload_type", "music_type", "onehot_feat3", "onehot_feat8",
+    "onehot_feat1", "onehot_feat7", "user_active_degree",
     "register_days_bucket", "register_days_range",
     "fans_user_num_range", "follow_user_num_range",
     "friend_user_num_range", "hour", "is_live_streamer",
+    "is_video_author", "video_type",
 ]
 NUM_FIELDS = [
@@ -33,4 +36,12 @@
     "user_friend_user_num", "user_register_days",
... 901 more diff lines
```

**#9 -> #11**  (+477 / -446 lines)

```diff
--- iter_9.py
+++ iter_11.py
@@ -3,8 +3,7 @@
 import json
 import gc
-import random
 import numpy as np
-import torch
-import torch.nn as nn
+from scipy import sparse
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -13,319 +12,419 @@
 
 START = time.time()
-SEED = 93217
-THREADS = max(1, min(8, os.cpu_count() or 1))
-torch.set_num_threads(THREADS)
+SEED = 74129
 np.random.seed(SEED)
-random.seed(SEED)
-torch.manual_seed(SEED)
-
-CAT_FIELDS = [
-    "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "music_type",
-    "onehot_feat3", "onehot_feat8", "onehot_feat1",
-    "onehot_feat7", "user_active_degree",
-    "register_days_bucket", "fans_user_num_range", "hour",
+
+CONTENT_FIELDS = [
+    "video_id",
+    "author_id",
+    "tag",
+    "duration_bucket",
+    "upload_type",
+    "music_type",
+    "onehot_feat3",
+    "onehot_feat8",
... 956 more diff lines
```

**#9 -> #12**  (+299 / -298 lines)

```diff
--- iter_9.py
+++ iter_12.py
@@ -4,4 +4,6 @@
 import gc
 import random
+from datetime import datetime
+
 import numpy as np
 import torch
@@ -13,17 +15,25 @@
 
 START = time.time()
-SEED = 93217
+SEED = 48173
 THREADS = max(1, min(8, os.cpu_count() or 1))
 torch.set_num_threads(THREADS)
+torch.set_num_interop_threads(1)
 np.random.seed(SEED)
 random.seed(SEED)
 torch.manual_seed(SEED)
 
-CAT_FIELDS = [
-    "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "music_type",
-    "onehot_feat3", "onehot_feat8", "onehot_feat1",
-    "onehot_feat7", "user_active_degree",
-    "register_days_bucket", "fans_user_num_range", "hour",
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
... 747 more diff lines
```

**#9 -> #13**  (+286 / -376 lines)

```diff
--- iter_9.py
+++ iter_13.py
@@ -5,6 +5,7 @@
 import random
 import numpy as np
-import torch
-import torch.nn as nn
+import lightgbm as lgb
+from scipy import sparse
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -13,10 +14,8 @@
 
 START = time.time()
-SEED = 93217
+SEED = 74031
 THREADS = max(1, min(8, os.cpu_count() or 1))
-torch.set_num_threads(THREADS)
 np.random.seed(SEED)
 random.seed(SEED)
-torch.manual_seed(SEED)
 
 CAT_FIELDS = [
@@ -25,5 +24,7 @@
     "onehot_feat3", "onehot_feat8", "onehot_feat1",
     "onehot_feat7", "user_active_degree",
-    "register_days_bucket", "fans_user_num_range", "hour",
+    "register_days_bucket", "register_days_range",
+    "fans_user_num_range", "follow_user_num_range",
+    "friend_user_num_range", "hour", "video_type",
 ]
 NUM_FIELDS = [
@@ -34,290 +35,186 @@
     "user_register_days",
 ]
-
-EMBED_DIM = 8
-BATCH_SIZE = 8192
-EPOCHS = 2
... 751 more diff lines
```

## Search tree

Each line is an executed script; indentation is the edit it was derived from. A node marked `[retired]` produced three children that failed to improve on it, so the search abandoned that branch and backtracked.

```
#1 0.6023 [retired]  Reproduce the official baseline at the modeling stage using a k=16 Fac
  #2 0.6042  Target prediction formation and fusion by comparing field-aware factor
  #3 0.6042 [retired]  Target the prediction and training-loss stages by comparing an additiv
    #5 0.6043  Target the training-loss and prediction-formation stages by comparing 
    #6 0.6049  Target prediction formation and temporal-drift training by comparing e
    #7 0.6052 [retired]  Target multi-task representation learning by comparing a single-task M
      #8 0.6052  Target the drift-sensitive history representation and prediction stage
      #9 0.6055 [retired]  Target prediction formation and score aggregation by comparing PNN pai
        #11 0.6057  Target drift-sensitive prediction formation by comparing temporal extr
        #12 0.6057  Target prediction formation and drift-aware training by comparing fiel
        #13 0.6057  Target prediction formation with two structurally different, underexpl
      #10 0.6055  Target prediction formation with three structurally distinct families—
```

## What the agent established

The agent's belief set, revised after every scored iteration rather than appended to, so later evidence can demote an earlier conclusion instead of piling up beside it. A claim marked `(invalidated)` was contradicted by a later result. Machine-readable form with per-claim evidence in `knowledge.json`.

```
- (active) A k=16 Factorization Machine trained with Adam at lr=0.001 on user_id, video_id, author_id, tab, and duration_bucket interactions reproduces the official validation baseline within seed noise (0.6023 versus 0.6016). [iters 1]
- (active) The field-aware/nonlinear-fusion and additive-wide/recency-weighted pointwise candidates both scored 0.6042, measurably above the 0.6016 official baseline but only 0.0019 above the stored FM, so they do not measurably improve on that FM. [iters 2]
- (active) The multi-task candidate scored 0.6052, measurably above the official baseline and stored FM; later history/drift, interaction-fusion, alternative prediction, temporal-EB/generative, field-weighted FM/NFM, and tree-family candidates at 0.6052-0.6057 produced no measurable gain over it. [iters 3,4,5]
- (active) The retired prediction/temporal-drift candidate scored 0.6049, measurably above the stored FM by 0.0026 but indistinguishable from the 0.6052 multi-task candidate and subsequent 0.6052-0.6057 candidates. [iters 3,4,5]
- (active) The retired pairwise-loss/prediction candidate scored 0.6043, which is not measurably different from the prior 0.6042 candidates or the stored FM under the greater-than-0.002 gain threshold. [iters 3]
- (active) The best numerical score is 0.6057, attained this turn by all three tested lineages, but its 0.0002 edge over the prior 0.6055 best and 0.00
```

## Alternatives compared inside iterations

282 candidate solutions were built and scored across 11 iteration(s). The convergence rule charges one iteration per experiment, so searching inside an iteration buys comparisons the iteration budget cannot.

| iteration | candidates (validation primary) |
|---|---|
| #2 | deepfm_blend_0.10 0.6030, deepfm_blend_0.20 0.6028, deepfm_blend_0.30 0.6033, deepfm_blend_0.40 0.6036, deepfm_blend_0.50 0.6036, deepfm_blend_0.60 0.6038, deepfm_blend_0.70 0.6040, deepfm_blend_0.80 0.6037 |
| #3 | incumbent 0.6023, lambdamart 0.5931, lambdamart_blend_0.15 0.6022, lambdamart_blend_0.25 0.6008, lambdamart_blend_0.35 0.6007, lambdamart_blend_0.50 0.5993, lambdamart_blend_0.65 0.5973, lambdamart_blend_0.80 0.5957 |
| #5 | bpr_eb_ensemble 0.5819, bpr_eb_ensemble_blend_0.10 0.6043, bpr_eb_ensemble_blend_0.20 0.6042, bpr_eb_ensemble_blend_0.30 0.6035, bpr_eb_ensemble_blend_0.40 0.6030, bpr_eb_ensemble_blend_0.55 0.6007, bpr_eb_ensemble_blend_0.70 0.5967, bpr_eb_ensemble_blend_0.85 0.5900 |
| #6 | autoint 0.6032, autoint_blend_0.10 0.6042, autoint_blend_0.20 0.6042, autoint_blend_0.30 0.6042, autoint_blend_0.45 0.6041, autoint_blend_0.60 0.6043, dcn 0.6045, dcn_blend_0.10 0.6041 |
| #7 | hard_shared_multitask 0.6039, hard_shared_multitask_blend_0.15 0.6042, hard_shared_multitask_blend_0.25 0.6043, hard_shared_multitask_blend_0.35 0.6042, hard_shared_multitask_blend_0.50 0.6044, hard_shared_multitask_blend_0.65 0.6046, incumbent 0.6042, mmoe 0.6033 |
| #8 | empirical_bayes_recent 0.5698, empirical_bayes_recent_blend_0.15 0.6051, empirical_bayes_recent_blend_0.25 0.6048, empirical_bayes_recent_blend_0.35 0.6042, empirical_bayes_recent_blend_0.50 0.6014, empirical_bayes_recent_blend_0.65 0.5958, empirical_bayes_stable 0.5766, empirical_bayes_stable_blend_0.15 0.6051 |
| #9 | autoint 0.6032, autoint_rankblend_0.15 0.6053, autoint_rankblend_0.25 0.6054, autoint_rankblend_0.35 0.6054, autoint_rankblend_0.50 0.6045, autoint_rankblend_0.65 0.6037, autoint_scoreblend_0.15 0.6052, autoint_scoreblend_0.25 0.6052 |
| #10 | gbdt_stationary 0.5970, gbdt_stationary_blend_0.10 0.6052, gbdt_stationary_blend_0.20 0.6053, gbdt_stationary_blend_0.30 0.6053, gbdt_stationary_blend_0.40 0.6052, gbdt_stationary_blend_0.50 0.6049, incumbent 0.6052, ple_multitask 0.6038 |
| #11 | hierarchical_cohort 0.5832, hierarchical_cohort_rankblend_0.10 0.6056, hierarchical_cohort_rankblend_0.20 0.6053, hierarchical_cohort_rankblend_0.30 0.6042, hierarchical_cohort_rankblend_0.40 0.6020, hierarchical_cohort_rankblend_0.55 0.5960, hierarchical_cohort_rankblend_0.70 0.5897, implicit_svd 0.5449 |
| #12 | fwfm 0.6022, fwfm_rankblend_0.15 0.6056, fwfm_rankblend_0.25 0.6055, fwfm_rankblend_0.35 0.6050, fwfm_rankblend_0.50 0.6041, fwfm_rankblend_0.65 0.6035, incumbent 0.6055, nfm 0.6009 |
| #13 | extra_random_forest 0.5898, extra_random_forest_rankblend_0.10 0.6057, extra_random_forest_rankblend_0.20 0.6055, extra_random_forest_rankblend_0.30 0.6048, extra_random_forest_rankblend_0.40 0.6028, extra_random_forest_rankblend_0.55 0.5982, extra_random_forest_rankblend_0.70 0.5938, extra_random_forest_scoreblend_0.10 0.6053 |

## Resource usage (Feasibility & Practicality)

- **Agent wall-clock to converged result: 0.37 h (22 min)**
- Total LLM tokens: **246,618** (162,632 in / 83,986 out), including the knowledge-revision stage
- Iterations used: **14 of 50** (12 accepted scores, 0 failed, 1 rejected)
- Compute inside generated scripts: **0.16 h (10 min)** on CPU.
- Mean tokens per iteration: 17,616
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

Across 38 development runs of this agent, 311 iterations were executed and 28 failed. Every one was handled in-loop; none was escalated to a human. Failure taxonomy:

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
- From iteration #12: Target prediction formation and drift-aware training by comparing field-weighted FM, NFM bi-interaction pooling, and xDeepFM explicit high-order crosses under identical recency weighting; their distinct interaction biases may recover complementary within-user preferences, while rank blending with the trusted incumbent limits calibration drift.
- Selection is on validation only, per the scoring rules; the hidden test set was never used to choose between iterations.
- Submission: written to `runs/r87_3slot/submission.csv`

### Results table

| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| validation (best iteration) | 0.6727 | 0.5388 | 0.6057 |
| official baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
| hidden test (this submission) | 0.6676 | 0.5332 | **0.6004** |
| official baseline (test) | 0.6610 | 0.5282 | 0.5946 |

**Absolute delta over baseline on hidden test: GAUC +0.0066, nDCG@5 +0.0050, mean +0.0058** (primary +0.0058).

Per the scoring formula, delta(m) = score_agent(m) - score_baseline(m), averaged over metrics.
