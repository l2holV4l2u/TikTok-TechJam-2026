#!/usr/bin/env python3
import os
import time
import json
import numpy as np
import lightgbm as lgb
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

t0 = time.time()

SEED = 12345
np.random.seed(SEED)

# Utility: safe save to ITER_OUT
OUTDIR = os.environ.get("ITER_OUT", "")
ARTDIR = os.environ.get("RUN_ARTIFACTS", "")

def save_valid_scores(arr):
    if OUTDIR:
        np.save(os.path.join(OUTDIR, "scores_valid.npy"), np.asarray(arr, dtype=np.float64))

def save_test_scores(arr):
    if OUTDIR:
        np.save(os.path.join(OUTDIR, "scores_test.npy"), np.asarray(arr, dtype=np.float64))

# Load splits
train = load("train")
valid = load("valid")
# Build counts & smoothed rates from the TRAIN split only (allowed)
# Fields we will use
USE_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "hour",
    "is_live_streamer", "is_video_author"
]

# numeric raw fields
NUM_FIELDS = ["duration_ms", "user_fans_user_num", "user_follow_user_num",
              "user_friend_user_num", "user_register_days"]

# Helper to get int arrays
def cat(name, s):
    return s.X[name].astype(np.int64)

# Prepare base counts and positives for author, video, user from train
y_train = train.y.astype(np.int32)
uids_train = cat("user_id", train)
aids_train = cat("author_id", train)
vids_train = cat("video_id", train)

# cardinalities (for safety)
user_card = int(FEATURE_CARDINALITIES.get("user_id", uids_train.max() + 1))
author_card = int(FEATURE_CARDINALITIES.get("author_id", aids_train.max() + 1))
video_card = int(FEATURE_CARDINALITIES.get("video_id", vids_train.max() + 1))

# counts and positive counts on full train (for valid/test use)
user_total = np.bincount(uids_train, minlength=user_card).astype(np.int32)
user_pos = np.bincount(uids_train, weights=y_train, minlength=user_card).astype(np.int32)

author_total = np.bincount(aids_train, minlength=author_card).astype(np.int32)
author_pos = np.bincount(aids_train, weights=y_train, minlength=author_card).astype(np.int32)

video_total = np.bincount(vids_train, minlength=video_card).astype(np.int32)
video_pos = np.bincount(vids_train, weights=y_train, minlength=video_card).astype(np.int32)

# Functions to compute per-row leave-one-out smoothed rates for train and full-train rates for valid/test
def laplace_rate(pos_arr, tot_arr, ids, y=None, alpha=1.0, beta=2.0, leave_one_out=False):
    # ids: per-row ids indexing into pos_arr/tot_arr
    if leave_one_out:
        # subtract the row itself
        tot_excl = tot_arr[ids] - 1
        pos_excl = pos_arr[ids] - y
        # avoid negative totals (shouldn't happen), clamp
        tot_excl = np.maximum(tot_excl, 0)
        pos_excl = np.maximum(pos_excl, 0)
        return (pos_excl + alpha) / (tot_excl + beta)
    else:
        tot = tot_arr[ids]
        pos = pos_arr[ids]
        # where tot==0, this yields alpha/beta (the prior)
        return (pos + alpha) / (tot + beta)

# Build feature matrix function
def build_features(split, is_train):
    n = len(split.y) if is_train else len(split.user_id)
    # categorical-derived ints (we keep some small-cardinality ints as-is)
    f_tab = split.X["tab"].astype(np.int32)
    f_hour = split.X["hour"].astype(np.int32)
    f_live = split.X["is_live_streamer"].astype(np.int32)
    f_vid_authorflag = split.X["is_video_author"].astype(np.int32)

    # numeric raw features: get from split.num and log1p transform
    nums = split.num
    def safe_log1p(arr):
        a = arr.astype(np.float64)
        # fill NaN with 0 (unknown -> treat as zero-ish)
        a = np.nan_to_num(a, nan=0.0)
        # heavy-tail scaling: log1p then cast
        return np.log1p(a).astype(np.float32)

    f_duration = safe_log1p(nums["duration_ms"])
    f_fans = safe_log1p(nums["user_fans_user_num"])
    f_follow = safe_log1p(nums["user_follow_user_num"])
    f_friend = safe_log1p(nums["user_friend_user_num"])
    f_regdays = safe_log1p(nums["user_register_days"])

    # author/video/user ids for indexing
    uids = split.X["user_id"].astype(np.int64)
    aids = split.X["author_id"].astype(np.int64)
    vids = split.X["video_id"].astype(np.int64)

    # author/video/user smoothed rates
    if is_train:
        # leave-one-out using split.y
        rates_author = laplace_rate(author_pos, author_total, aids, y=split.y, leave_one_out=True)
        rates_video = laplace_rate(video_pos, video_total, vids, y=split.y, leave_one_out=True)
        rates_user = laplace_rate(user_pos, user_total, uids, y=split.y, leave_one_out=True)
        # counts excl (we'll include total counts as features too, clipped)
        author_cnt = np.maximum(author_total[aids] - 1, 0).astype(np.float32)
        video_cnt = np.maximum(video_total[vids] - 1, 0).astype(np.float32)
        user_cnt = np.maximum(user_total[uids] - 1, 0).astype(np.float32)
    else:
        # valid/test: use full-train counts (no leakage)
        rates_author = laplace_rate(author_pos, author_total, aids, leave_one_out=False)
        rates_video = laplace_rate(video_pos, video_total, vids, leave_one_out=False)
        rates_user = laplace_rate(user_pos, user_total, uids, leave_one_out=False)
        author_cnt = author_total[aids].astype(np.float32)
        video_cnt = video_total[vids].astype(np.float32)
        user_cnt = user_total[uids].astype(np.float32)

    # popularity features: global counts (log scaled)
    author_cnt_log = np.log1p(author_cnt).astype(np.float32)
    video_cnt_log = np.log1p(video_cnt).astype(np.float32)
    user_cnt_log = np.log1p(user_cnt).astype(np.float32)

    # Assemble feature matrix (float32)
    feat_cols = [
        rates_author.astype(np.float32),
        rates_video.astype(np.float32),
        rates_user.astype(np.float32),
        author_cnt_log,
        video_cnt_log,
        user_cnt_log,
        f_duration.astype(np.float32),
        f_fans.astype(np.float32),
        f_follow.astype(np.float32),
        f_friend.astype(np.float32),
        f_regdays.astype(np.float32),
        f_tab.astype(np.float32),
        f_hour.astype(np.float32),
        f_live.astype(np.float32),
        f_vid_authorflag.astype(np.float32),
    ]
    X = np.vstack([c.reshape(-1) for c in feat_cols]).T.astype(np.float32)
    return X

# Build matrices
X_train = build_features(train, is_train=True)
y_train = train.y.astype(np.float32)
uids_train = train.user_id.astype(np.int64)

X_valid = build_features(valid, is_train=False)
y_valid = valid.y.astype(np.float32)
uids_valid = valid.user_id.astype(np.int64)

# For test later, build features now (must not touch test.y)
te = load("test")
X_test = build_features(te, is_train=False)
uids_test = te.user_id.astype(np.int64)

# LightGBM lambdarank requires groups contiguous by query. We'll sort by user_id for train and valid.
def sort_by_user(X, y, user_ids):
    perm = np.argsort(user_ids, kind="mergesort")
    Xs = X[perm]
    ys = y[perm]
    u_sorted = user_ids[perm]
    # compute group lengths
    _, counts = np.unique(u_sorted, return_counts=True)
    return Xs, ys, perm, counts

Xtr_s, ytr_s, perm_tr, group_tr = sort_by_user(X_train, y_train, uids_train)
Xva_s, yva_s, perm_va, group_va = sort_by_user(X_valid, y_valid, uids_valid)

# Create lgb datasets
dtrain = lgb.Dataset(Xtr_s, label=ytr_s, group=group_tr, free_raw_data=False)
dvalid = lgb.Dataset(Xva_s, label=yva_s, group=group_va, reference=dtrain, free_raw_data=False)

# LightGBM parameters for lambdarank targeting nDCG@5
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 50,
    "verbose": -1,
    "seed": SEED,
    "force_row_wise": True,
}

# Train with early stopping (callback style required in lightgbm 4.7)
num_boost_round = 300
early_stopping_rounds = 40

bst = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                valid_sets=[dvalid], callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])

# Predict on validation: predictions correspond to sorted validation order; map back to original order
preds_va_sorted = bst.predict(Xva_s, num_iteration=bst.best_iteration)
preds_va = np.empty_like(preds_va_sorted)
preds_va[perm_va] = preds_va_sorted  # map to original valid ordering

# Save raw valid scores (must save these exact array)
save_valid_scores(preds_va)

# Evaluate
res_raw = evaluate(valid.user_id, valid.y, preds_va)
primary_raw = float(res_raw["primary"])
gauc_raw = float(res_raw["gauc"])
ndcg5_raw = float(res_raw["ndcg@5"])

# Prepare candidate dictionary
candidates = {"lgb_raw": primary_raw}

# Optionally blend with incumbent predictions from RUN_ARTIFACTS if present
best_final_scores = preds_va.copy()
best_final_primary = primary_raw
best_choice_name = "lgb_raw"
inc_path = os.path.join(ARTDIR, "incumbent_valid_scores.npy") if ARTDIR else None
if inc_path and os.path.exists(inc_path):
    try:
        incumbent_valid = np.load(inc_path)
        # pick convex blend weight by grid search (validate on valid only)
        best_alpha = 0.0
        for alpha in np.linspace(0.0, 1.0, 21):
            blended = alpha * incumbent_valid + (1.0 - alpha) * preds_va
            r = evaluate(valid.user_id, valid.y, blended)
            prim = float(r["primary"])
            if prim > best_final_primary + 1e-12:
                best_final_primary = prim
                best_alpha = float(alpha)
                best_final_scores = blended.copy()
                best_choice_name = f"blend_incumbent_alpha{alpha:.3f}"
        candidates["blend_with_incumbent_best_alpha"] = best_alpha
        candidates["blend_with_incumbent_primary"] = best_final_primary
        # if we found improvement, update gauc/ndcg5 accordingly
        if best_choice_name != "lgb_raw":
            res_blend = evaluate(valid.user_id, valid.y, best_final_scores)
            gauc_raw = float(res_blend["gauc"])
            ndcg5_raw = float(res_blend["ndcg@5"])
    except Exception as e:
        print(f"FINDINGS failed_to_load_or_blend_incumbent: {e}")

# Print candidate summary
print("CANDIDATES " + json.dumps(candidates))

# Save the chosen valid scores (best_final_scores) to ITER_OUT (overwriting previous)
save_valid_scores(best_final_scores)

# Evaluate final chosen predictions to get metrics (redo to be safe)
res_final = evaluate(valid.user_id, valid.y, best_final_scores)
primary = float(res_final["primary"])
gauc = float(res_final["gauc"])
ndcg5 = float(res_final["ndcg@5"])

# Produce test predictions using same model
# For predict we didn't sort test; we can predict in original order (no grouping needed)
preds_test = bst.predict(X_test, num_iteration=bst.best_iteration)

# If we blended with incumbent, apply same blend weight (best_alpha) using incumbent_test if available
if best_choice_name.startswith("blend_incumbent") and ARTDIR:
    inc_test_path = os.path.join(ARTDIR, "incumbent_test_scores.npy")
    if os.path.exists(inc_test_path):
        try:
            incumbent_test = np.load(inc_test_path)
            alpha = float(candidates.get("blend_with_incumbent_best_alpha", 0.0))
            preds_test = alpha * incumbent_test + (1.0 - alpha) * preds_test
        except Exception as e:
            print(f"FINDINGS failed_to_load_incumbent_test_for_blending: {e}")

# Save test scores
save_test_scores(preds_test)

# Feature importances
try:
    imps = bst.feature_importance(importance_type="gain")
    names = [f"f{i}" for i in range(X_train.shape[1])]
    imp_pairs = sorted(zip(names, imps), key=lambda x: -x[1])[:10]
    print("FINDINGS top_feature_importances " + json.dumps(imp_pairs))
except Exception as e:
    print(f"FINDINGS failed_to_report_feature_importance: {e}")

t1 = time.time()
wall = float(t1 - t0)

# Final METRICS line
print(f'METRICS {{"primary": {primary:.6f}, "gauc": {gauc:.6f}, "ndcg@5": {ndcg5:.6f}, "gpu_seconds": {wall:.3f}}}')