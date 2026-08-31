# Run report - r86_2slot

> 1 scored iteration(s) were rejected by the integrity critic. They are excluded from the search tree, cross-run memory, and submission.

## Loop stages executed by the agent

- **Inspect data (EDA):** completed at iteration #0 - the agent wrote and ran its own exploratory script; its findings are in `eda_report.txt` and were carried into every later prompt.
- **Reproduce official baseline (Requirement 1):** 2 attempt(s); the agent's own pipeline reached validation primary **0.6016** against the official 0.6016 (delta -0.0000, inside the baseline's 5-seed noise). That script became the root of the search tree.
- **Iterate:** 8 experiments proposed, executed and evaluated by the agent, each branching from a node of its own search tree.

## Iteration log

A turn is one hypothesis-to-score cycle and advances several lineages at once, so `turn` and `slot` say when an experiment ran and which lineage produced it. The convergence rule is measured against turns; the 50-cap counts scripts.

| # | turn | slot | phase | parent | status | secs | primary | vs baseline | hypothesis |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | - | eda | - | ok | 2 | n/a | - | In the data-inspection stage, quantify drift, entity cold-start, per-u |
| 1 | 0 | - | baseline | - | reverted | 13 | 0.6039 | +0.0023 | Reproduce the official baseline stage by training a k=16 Factorization |
| 2 | 0 | - | baseline | - | ok | 10 | 0.6016 | -0.0000 | Reproduce the baseline FM at the modeling stage by strictly whitelisti |
| 3 | 1 | 0 | improve | #2 | kept | 59 | 0.6036 | +0.0020 | Target prediction formation and fusion by comparing expanded-field fac |
| 4 | 1 | 1 | improve | #2 | ok | 20 | 0.6038 | +0.0022 | Target drift-aware training and prediction formation by comparing rece |
| 5 | 2 | 0 | improve | #4 | rejected | 66 | 0.6042 | +0.0026 | Target prediction formation and training supervision by comparing DCN  |
| 6 | 2 | 1 | improve | #4 | kept | 41 | 0.6039 | +0.0023 | Target the prediction-formation and ranking-loss stages by comparing d |
| 7 | 3 | 0 | improve | #6 | kept | 48 | 0.6043 | +0.0027 | Target prediction formation with a breadth comparison between AutoInt  |
| 8 | 3 | 1 | improve | #6 | kept | 46 | 0.6044 | +0.0028 | Target sequential prediction formation by comparing DIN target-attenti |
| 9 | 4 | 0 | improve | #8 | kept | 6 | 0.6044 | +0.0028 | Target prediction formation and drift-robust personalization by compar |
| 10 | 4 | 1 | improve | #8 | kept | 3 | 0.6045 | +0.0029 | Target the two-stage reranking and drift-adaptation stages with hierar |

## Portfolio

- Lineages advanced per turn: **2**
- Turns: **4**  ·  scripts executed: **11** of 50
- A turn is one hypothesis-to-score cycle -- Figure 1's iteration, and what the convergence rule is measured against. Scripts are reported separately so the run can be checked against the stricter reading of the 50-iteration cap as well.
- Lineages archived: **2**  ·  revived from the archive: **0**

### Did the lineages actually disagree?

Within-user rank correlation between the live slots' validation predictions. This is the acceptance test for the whole design: above ~0.95 the extra slots are three copies of one agent and bought nothing.

| turn | scripts | mean corr | max corr | alert |
|---|---|---|---|---|
| 1 | #3, #4 | 0.8415 | 0.8415 | no |
| 2 | #5, #6 | 0.7385 | 0.7385 | no |
| 3 | #7, #8 | 0.9241 | 0.9241 | no |
| 4 | #9, #10 | 0.2933 | 0.2933 | no |

### Lineages retired and replaced

A slot that stops beating its own best is archived, not stopped -- the run's convergence counter is untouched by it. Refill alternates fresh drafts with a revival, and a revival is chosen for score AND for disagreeing with what is already live.

| turn | slot | archived at | choice | detail |
|---|---|---|---|---|
| 2 | 0 | 0.6036 | fresh | new draft |
| 3 | 1 | 0.6044 | fresh | new draft |

### Controller blend of the portfolio

Every turn the controller blends the trusted incumbent, the live slots and the archive, choosing members on one half of validation and confirming on the other. A blend that wins the selection fold and loses the confirmation fold is refused: it won the split, not the task.

| turn | pool | accepted | members | fold A | fold B | reason |
|---|---|---|---|---|---|---|
| 1 | 2 | no | - | - | - | no member improved fold A |
| 2 | 2 | no | - | - | - | no member improved fold A |
| 3 | 3 | no | - | - | - | no member improved fold A |
| 4 | 4 | no | - | - | - | no member improved fold A |

## Code diff applied, per iteration

Each iteration edits a parent script. This is that edit, reconstructed from the two full sources in the ledger, truncated to the first 40 lines per iteration.


**#2 -> #3**  (+303 / -79 lines)

```diff
--- iter_2.py
+++ iter_3.py
@@ -6,4 +6,5 @@
 import torch
 import torch.nn as nn
+import lightgbm as lgb
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -13,9 +14,20 @@
 START = time.time()
 SEED = 2022
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-EMBED_DIM = 16
-LEARNING_RATE = 0.001
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
+CARDINALITIES = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
 BATCH_SIZE = 4096
-EPOCHS = 5
+FM_EPOCHS = 5
+FM_DIM = 16
+FM_LR = 0.001
 
 np.random.seed(SEED)
@@ -24,35 +36,5 @@
 
 
-class FactorizationMachine(nn.Module):
-    def __init__(self, cardinalities, embedding_dim):
-        super().__init__()
-        self.offsets = torch.tensor(
... 409 more diff lines
```

**#2 -> #4**  (+404 / -110 lines)

```diff
--- iter_2.py
+++ iter_4.py
@@ -3,4 +3,5 @@
 import json
 import gc
+import math
 import numpy as np
 import torch
@@ -10,111 +11,362 @@
 from pipeline.evaluate import evaluate
 
-
 START = time.time()
-SEED = 2022
-FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-EMBED_DIM = 16
-LEARNING_RATE = 0.001
-BATCH_SIZE = 4096
-EPOCHS = 5
-
+SEED = 24029
+BATCH_SIZE = 8192
+DEVICE = torch.device("cpu")
+torch.manual_seed(SEED)
 np.random.seed(SEED)
-torch.manual_seed(SEED)
 torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
 
-
-class FactorizationMachine(nn.Module):
-    def __init__(self, cardinalities, embedding_dim):
-        super().__init__()
-        self.offsets = torch.tensor(
-            np.cumsum([0] + cardinalities[:-1]), dtype=torch.long
-        )
-        total_cardinality = int(sum(cardinalities))
-
-        self.linear = nn.Embedding(total_cardinality, 1)
-        self.embedding = nn.Embedding(total_cardinality, embedding_dim)
-        self.bias = nn.Parameter(torch.zeros(1))
... 518 more diff lines
```

**#4 -> #5**  (+497 / -380 lines)

```diff
--- iter_4.py
+++ iter_5.py
@@ -3,5 +3,4 @@
 import json
 import gc
-import math
 import numpy as np
 import torch
@@ -12,69 +11,281 @@
 
 START = time.time()
-SEED = 24029
-BATCH_SIZE = 8192
-DEVICE = torch.device("cpu")
+SEED = 73129
+BATCH = 8192
+PRED_BATCH = 16384
+THREADS = max(1, min(16, os.cpu_count() or 1))
+torch.set_num_threads(THREADS)
 torch.manual_seed(SEED)
 np.random.seed(SEED)
-torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
-
-DEEP_FIELDS = [
+
+FIELDS = [
     "user_id", "video_id", "author_id", "tab", "duration_bucket",
     "tag", "upload_type", "music_type", "hour",
 ]
-MF_FIELDS = ["user_id", "video_id", "author_id"]
-HALF_LIFE = 7.0
-
-
-def day_weights(dates):
-    dates = np.asarray(dates, dtype=np.int64)
-    # All fitting windows end in April, so day-of-month differences are exact.
-    day = dates % 100
-    age = day.max() - day
-    w = np.exp2(-age.astype(np.float32) / HALF_LIFE)
-    return w / np.mean(w)
... 909 more diff lines
```

**#4 -> #6**  (+394 / -380 lines)

```diff
--- iter_4.py
+++ iter_6.py
@@ -3,8 +3,8 @@
 import json
 import gc
-import math
 import numpy as np
-import torch
-import torch.nn as nn
+import lightgbm as lgb
+from scipy import sparse
+from scipy.sparse.linalg import svds
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -12,269 +12,258 @@
 
 START = time.time()
-SEED = 24029
-BATCH_SIZE = 8192
-DEVICE = torch.device("cpu")
-torch.manual_seed(SEED)
+SEED = 73129
 np.random.seed(SEED)
-torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
-
-DEEP_FIELDS = [
+
+CAT_FIELDS = [
     "user_id", "video_id", "author_id", "tab", "duration_bucket",
-    "tag", "upload_type", "music_type", "hour",
+    "tag", "upload_type", "music_type", "hour", "user_active_degree",
+    "fans_user_num_range", "follow_user_num_range",
+    "friend_user_num_range", "register_days_range",
+    "video_type", "onehot_feat1", "onehot_feat3",
+    "onehot_feat7", "onehot_feat8", "onehot_feat11",
 ]
-MF_FIELDS = ["user_id", "video_id", "author_id"]
-HALF_LIFE = 7.0
-
-
... 807 more diff lines
```

**#6 -> #7**  (+343 / -345 lines)

```diff
--- iter_6.py
+++ iter_7.py
@@ -1,10 +1,10 @@
 import os
+import gc
+import json
 import time
-import json
-import gc
+import random
 import numpy as np
-import lightgbm as lgb
-from scipy import sparse
-from scipy.sparse.linalg import svds
+import torch
+import torch.nn as nn
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -12,236 +12,37 @@
 
 START = time.time()
-SEED = 73129
+SEED = 92741
+random.seed(SEED)
 np.random.seed(SEED)
-
-CAT_FIELDS = [
+torch.manual_seed(SEED)
+torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
+
+FIELDS = [
     "user_id", "video_id", "author_id", "tab", "duration_bucket",
     "tag", "upload_type", "music_type", "hour", "user_active_degree",
     "fans_user_num_range", "follow_user_num_range",
     "friend_user_num_range", "register_days_range",
-    "video_type", "onehot_feat1", "onehot_feat3",
-    "onehot_feat7", "onehot_feat8", "onehot_feat11",
+    "video_type", "onehot_feat1", "onehot_feat3", "onehot_feat8",
 ]
-NUM_FIELDS = [
... 731 more diff lines
```

**#6 -> #8**  (+429 / -315 lines)

```diff
--- iter_6.py
+++ iter_8.py
@@ -3,8 +3,8 @@
 import json
 import gc
+import math
 import numpy as np
-import lightgbm as lgb
-from scipy import sparse
-from scipy.sparse.linalg import svds
+import torch
+import torch.nn as nn
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -12,14 +12,13 @@
 
 START = time.time()
-SEED = 73129
+SEED = 82417
 np.random.seed(SEED)
+torch.manual_seed(SEED)
+torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
 
 CAT_FIELDS = [
-    "user_id", "video_id", "author_id", "tab", "duration_bucket",
-    "tag", "upload_type", "music_type", "hour", "user_active_degree",
-    "fans_user_num_range", "follow_user_num_range",
-    "friend_user_num_range", "register_days_range",
-    "video_type", "onehot_feat1", "onehot_feat3",
-    "onehot_feat7", "onehot_feat8", "onehot_feat11",
+    "user_id", "video_id", "author_id", "tab", "tag",
+    "duration_bucket", "upload_type", "music_type",
+    "hour", "user_active_degree",
 ]
 NUM_FIELDS = [
@@ -27,6 +26,16 @@
     "user_friend_user_num", "user_register_days",
 ]
+HISTORY_LENGTH = 12
+EMBED_DIM = 12
... 806 more diff lines
```

**#8 -> #9**  (+517 / -513 lines)

```diff
--- iter_8.py
+++ iter_9.py
@@ -3,22 +3,27 @@
 import json
 import gc
-import math
+import warnings
 import numpy as np
-import torch
-import torch.nn as nn
-
-from pipeline.data import load, FEATURE_CARDINALITIES
+import lightgbm as lgb
+from scipy import sparse
+from scipy.sparse.linalg import svds
+
+from pipeline.data import load
 from pipeline.evaluate import evaluate
 
+warnings.filterwarnings("ignore")
 START = time.time()
-SEED = 82417
-np.random.seed(SEED)
-torch.manual_seed(SEED)
-torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
+SEED = 73129
+rng = np.random.default_rng(SEED)
 
 CAT_FIELDS = [
     "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "music_type",
-    "hour", "user_active_degree",
+    "duration_bucket", "upload_type", "music_type", "hour",
+    "user_active_degree", "fans_user_num_range",
+    "follow_user_num_range", "friend_user_num_range",
+    "register_days_range", "video_type",
+    "onehot_feat0", "onehot_feat1", "onehot_feat2",
+    "onehot_feat3", "onehot_feat7", "onehot_feat8",
+    "onehot_feat11", "onehot_feat12",
 ]
... 1053 more diff lines
```

**#8 -> #10**  (+390 / -541 lines)

```diff
--- iter_8.py
+++ iter_10.py
@@ -3,8 +3,5 @@
 import json
 import gc
-import math
 import numpy as np
-import torch
-import torch.nn as nn
 
 from pipeline.data import load, FEATURE_CARDINALITIES
@@ -12,30 +9,63 @@
 
 START = time.time()
-SEED = 82417
-np.random.seed(SEED)
-torch.manual_seed(SEED)
-torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
-
-CAT_FIELDS = [
-    "user_id", "video_id", "author_id", "tab", "tag",
-    "duration_bucket", "upload_type", "music_type",
-    "hour", "user_active_degree",
+
+MARGINAL_FIELDS = [
+    "video_id", "author_id", "tag", "duration_bucket",
+    "upload_type", "music_type", "tab", "hour",
 ]
-NUM_FIELDS = [
-    "duration_ms", "user_fans_user_num", "user_follow_user_num",
-    "user_friend_user_num", "user_register_days",
+PAIR_FIELDS = [
+    "video_id", "author_id", "tag", "duration_bucket",
+    "upload_type", "music_type", "tab",
 ]
-HISTORY_LENGTH = 12
-EMBED_DIM = 12
-BATCH_SIZE = 8192
-
-FIELD_OFFSETS = {}
... 932 more diff lines
```

## Search tree

Each line is an executed script; indentation is the edit it was derived from. A node marked `[retired]` produced three children that failed to improve on it, so the search abandoned that branch and backtracked.

```
#2 0.6016  Reproduce the baseline FM at the modeling stage by strictly whitelisti
  #3 0.6036  Target prediction formation and fusion by comparing expanded-field fac
  #4 0.6038 [retired]  Target drift-aware training and prediction formation by comparing rece
    #6 0.6039 [retired]  Target the prediction-formation and ranking-loss stages by comparing d
      #7 0.6043  Target prediction formation with a breadth comparison between AutoInt 
      #8 0.6044 [retired]  Target sequential prediction formation by comparing DIN target-attenti
        #9 0.6044  Target prediction formation and drift-robust personalization by compar
        #10 0.6045  Target the two-stage reranking and drift-adaptation stages with hierar
```

## What the agent established

The agent's belief set, revised after every scored iteration rather than appended to, so later evidence can demote an earlier conclusion instead of piling up beside it. A claim marked `(invalidated)` was contradicted by a later result. Machine-readable form with per-claim evidence in `knowledge.json`.

```
- (qualified) The k=16 FM at lr=0.001 scored 0.6039 only with the reverted implementation containing prior feature/label leakage; with five categorical fields strictly whitelisted and validation labels reserved for the permitted test refit, it scored 0.6016. [iters 1,2]
- (active) Expanded-field factorization/gradient-boosting fusion (0.6036), drift-aware MF/NFM (0.6038), ranking-loss work (0.6039), AutoInt (0.6043), DIN/GRU sequential encoding (0.6044), structurally varied prediction/drift personalization (0.6044), and hierarchical empirical-Bayes user×content reranking (0.6045) have no measurable primary-metric d [iters 3,4,5,6,7]
- (qualified) Within-user rank agreement is strongly lineage-pair dependent: earlier observed correlations include 0.738, 0.842, and 0.924, whereas the two newest lineages correlate only 0.293 despite indistinguishable primary scores. [iters 3,4,5,6,7]
```

## Alternatives compared inside iterations

41 candidate solutions were built and scored across 7 iteration(s). The convergence rule charges one iteration per experiment, so searching inside an iteration buys comparisons the iteration budget cannot.

| iteration | candidates (validation primary) |
|---|---|
| #3 | empirical_bayes 0.5921, empirical_bayes_incumbent_blend 0.6023, expanded_fm 0.6016, expanded_fm_incumbent_blend 0.6028, lightgbm_binary 0.6017, lightgbm_binary_incumbent_blend 0.6036 |
| #4 | recency_empirical_bayes_blend 0.6018, recency_latent_mf_blend 0.6018, recency_nfm_blend 0.6022 |
| #6 | lambdarank_blend 0.6038, positive_svd_cf_blend 0.6038, positive_transition_blend 0.6038 |
| #7 | autoint_blend 0.6043, eb_broad_personalization_blend 0.6037, eb_memorization_blend 0.6037, eb_preference_residual_blend 0.6037, eb_stationary_content_blend 0.6037 |
| #8 | din_blend 0.6044, din_raw 0.6040, gru_blend 0.6040, gru_raw 0.6022 |
| #9 | empirical_balanced_blend 0.6044, empirical_balanced_raw 0.5807, empirical_entity_blend 0.6044, empirical_entity_raw 0.5806, empirical_personal_blend 0.6044, empirical_personal_raw 0.5807, lightgbm_target_stats_blend 0.6044, lightgbm_target_stats_raw 0.5612 |
| #10 | recent3_hierarchical_personal_blend 0.6043, recent3_hierarchical_personal_raw 0.5869, recent7_hierarchical_personal_blend 0.6044, recent7_hierarchical_personal_raw 0.5872, recent7_preference_residual_blend 0.6040, recent7_preference_residual_raw 0.5048, stable_hierarchical_personal_blend 0.6044, stable_hierarchical_personal_raw 0.5862 |

## Resource usage (Feasibility & Practicality)

- **Agent wall-clock to converged result: 0.26 h (15 min)**
- Total LLM tokens: **179,572** (118,343 in / 61,229 out), including the knowledge-revision stage
- Iterations used: **11 of 50** (9 accepted scores, 0 failed, 1 rejected)
- Compute inside generated scripts: **0.09 h (5 min)** on CPU.
- Mean tokens per iteration: 16,325
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

Across 38 development runs of this agent, 314 iterations were executed and 28 failed. Every one was handled in-loop; none was escalated to a human. Failure taxonomy:

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

- Best validation primary: **0.6045** (baseline 0.6016, delta +0.0029)
- From iteration #10: Target the two-stage reranking and drift-adaptation stages with hierarchical empirical-Bayes user×content preferences estimated under several temporal half-lives; these sparse posterior residuals can correct the trusted model’s ordering using recent user-specific tastes while shrinking unreliable pairs toward stable item priors.
- Selection is on validation only, per the scoring rules; the hidden test set was never used to choose between iterations.
- Submission: written to `runs/r86_2slot/submission.csv`

### Results table

| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| validation (best iteration) | 0.6709 | 0.5380 | 0.6045 |
| official baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
| hidden test (this submission) | 0.6668 | 0.5322 | **0.5995** |
| official baseline (test) | 0.6610 | 0.5282 | 0.5946 |

**Absolute delta over baseline on hidden test: GAUC +0.0058, nDCG@5 +0.0040, mean +0.0049** (primary +0.0049).

Per the scoring formula, delta(m) = score_agent(m) - score_baseline(m), averaged over metrics.
