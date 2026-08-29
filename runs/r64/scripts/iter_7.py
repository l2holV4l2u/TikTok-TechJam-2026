#!/usr/bin/env python3
import os
import time
import json
import numpy as np
import lightgbm as lgb
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

# Timer
t0 = time.time()

SEED = 12345
np.random.seed(SEED)

OUTDIR = os.environ.get("ITER_OUT", "")
ARTDIR = os.environ.get("RUN_ARTIFACTS", "")

def save_valid_scores(arr):
    if OUTDIR:
        np.save(os.path.join(OUTDIR, "scores_valid.npy"), np.asarray(arr, dtype=np.float64))

def save_test_scores(arr):
    if OUTDIR:
        np.save(os.path.join(OUTDIR, "scores_test.npy"), np.asarray(arr, dtype=np.float64))

# Simple utilities
def safe_log1p_column(arr):
    a = arr.astype(np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
    return np.log1p(a).astype(np.float32)

def laplace_rate_vec(pos_arr, tot_arr, idxs, alpha=1.0, beta=2.0):
    # idxs indexes into pos_arr/tot_arr (vectorized)
    tot = tot_arr[idxs]
    pos = pos_arr[idxs]
    return ((pos + alpha) / (tot + beta)).astype(np.float32)

# Robust mapping for arbitrary linearized keys -> counts/pos arrays
def map_keys_to_counts(row_keys, unique_keys, counts, pos_counts):
    # unique_keys is sorted ascending as returned by np.unique
    idx = np.searchsorted(unique_keys, row_keys)
    found = (idx < len(unique_keys)) & (unique_keys[idx] == row_keys)
    out_counts = np.zeros(len(row_keys), dtype=np.int32)
    out_pos = np.zeros(len(row_keys), dtype=np.int32)
    if found.any():
        sel = np.nonzero(found)[0]
        out_counts[sel] = counts[idx[sel]]
        out_pos[sel] = pos_counts[idx[sel]]
    return out_counts, out_pos

# Load splits
train = load("train")
valid = load("valid")
test = load("test")  # only features accessible; touching test.y will raise if it existed

# Extract basic arrays
y_tr = train.y.astype(np.int32)
uids_tr = train.X["user_id"].astype(np.int64)
aids_tr = train.X["author_id"].astype(np.int64)
vids_tr = train.X["video_id"].astype(np.int64)

uids_va = valid.X["user_id"].astype(np.int64)
aids_va = valid.X["author_id"].astype(np.int64)
vids_va = valid.X["video_id"].astype(np.int64)

uids_te = test.X["user_id"].astype(np.int64)
aids_te = test.X["author_id"].astype(np.int64)
vids_te = test.X["video_id"].astype(np.int64)

# Cardinalities (from metadata)
user_card = int(FEATURE_CARDINALITIES.get("user_id", int(uids_tr.max()) + 1))
author_card = int(FEATURE_CARDINALITIES.get("author_id", int(aids_tr.max()) + 1))
video_card = int(FEATURE_CARDINALITIES.get("video_id", int(vids_tr.max()) + 1))

# Compute per-entity totals and positives (train-only; valid/test must use train stats)
user_total = np.bincount(uids_tr, minlength=user_card).astype(np.int32)
user_pos = np.bincount(uids_tr, weights=y_tr, minlength=user_card).astype(np.int32)

author_total = np.bincount(aids_tr, minlength=author_card).astype(np.int32)
author_pos = np.bincount(aids_tr, weights=y_tr, minlength=author_card).astype(np.int32)

video_total = np.bincount(vids_tr, minlength=video_card).astype(np.int32)
video_pos = np.bincount(vids_tr, weights=y_tr, minlength=video_card).astype(np.int32)

# Build unique pair statistics for (user,author) and (user,video) from train only.
# We'll linearize pair keys safely: user * (other_card + 1) + other_id
mul_author = author_card + 3
mul_video = video_card + 3

pair_ua_tr_keys = (uids_tr.astype(np.int64) * mul_author) + aids_tr.astype(np.int64)
pair_uv_tr_keys = (uids_tr.astype(np.int64) * mul_video) + vids_tr.astype(np.int64)

# Unique keys and inverse mapping for train pairs
ua_unique_keys, ua_inv = np.unique(pair_ua_tr_keys, return_inverse=True)
ua_counts = np.bincount(ua_inv).astype(np.int32)
ua_pos = np.bincount(ua_inv, weights=y_tr).astype(np.int32)

uv_unique_keys, uv_inv = np.unique(pair_uv_tr_keys, return_inverse=True)
uv_counts = np.bincount(uv_inv).astype(np.int32)
uv_pos = np.bincount(uv_inv, weights=y_tr).astype(np.int32)

# For train rows we have inv arrays (ua_inv, uv_inv) to index into counts/pos for leave-one-out
# For valid/test we will map via searchsorted with safety checks implemented above.

# Build feature matrix function
def build_features(split, split_name="train"):
    # split_name in {"train","valid","test"} only matters for leave-one-out vs full-train usage
    nrows = len(split.y) if split_name == "train" else len(split.X["user_id"])
    # Basic categorical small-cardinality ints kept as floats
    f_tab = split.X["tab"].astype(np.int32).astype(np.float32)
    f_hour = split.X["hour"].astype(np.int32).astype(np.float32)
    f_live = split.X["is_live_streamer"].astype(np.int32).astype(np.float32)
    f_vid_authorflag = split.X["is_video_author"].astype(np.int32).astype(np.float32)

    # Numeric raw features
    nums = split.num
    f_duration = safe_log1p_column(nums["duration_ms"])
    f_fans = safe_log1p_column(nums["user_fans_user_num"])
    f_follow = safe_log1p_column(nums["user_follow_user_num"])
    f_friend = safe_log1p_column(nums["user_friend_user_num"])
    f_regdays = safe_log1p_column(nums["user_register_days"])

    # ids
    uids = split.X["user_id"].astype(np.int64)
    aids = split.X["author_id"].astype(np.int64)
    vids = split.X["video_id"].astype(np.int64)

    # per-entity smoothed rates: for train use leave-one-out; for val/test use full-train stats (no leakage)
    if split_name == "train":
        # leave-one-out for user/author/video
        # map per-row to entity totals
        u_tot_excl = user_total[uids] - 1
        u_pos_excl = user_pos[uids] - split.y
        u_tot_excl = np.maximum(u_tot_excl, 0)
        u_pos_excl = np.maximum(u_pos_excl, 0)
        rate_user = (u_pos_excl + 1.0) / (u_tot_excl + 2.0)

        a_tot_excl = author_total[aids] - 1
        a_pos_excl = author_pos[aids] - split.y
        a_tot_excl = np.maximum(a_tot_excl, 0)
        a_pos_excl = np.maximum(a_pos_excl, 0)
        rate_author = (a_pos_excl + 1.0) / (a_tot_excl + 2.0)

        v_tot_excl = video_total[vids] - 1
        v_pos_excl = video_pos[vids] - split.y
        v_tot_excl = np.maximum(v_tot_excl, 0)
        v_pos_excl = np.maximum(v_pos_excl, 0)
        rate_video = (v_pos_excl + 1.0) / (v_tot_excl + 2.0)

        # entity counts (leave-one-out)
        user_cnt = u_tot_excl.astype(np.float32)
        author_cnt = a_tot_excl.astype(np.float32)
        video_cnt = v_tot_excl.astype(np.float32)
    else:
        # use full-train aggregates (no leakage)
        rate_user = laplace_rate_vec(user_pos, user_total, uids, alpha=1.0, beta=2.0)
        rate_author = laplace_rate_vec(author_pos, author_total, aids, alpha=1.0, beta=2.0)
        rate_video = laplace_rate_vec(video_pos, video_total, vids, alpha=1.0, beta=2.0)

        user_cnt = user_total[uids].astype(np.float32)
        author_cnt = author_total[aids].astype(np.float32)
        video_cnt = video_total[vids].astype(np.float32)

    # Pair keys for (user,author) and (user,video)
    pair_ua_keys = (uids * mul_author) + aids
    pair_uv_keys = (uids * mul_video) + vids

    # Pair stats: for train leave-one-out using ua_inv/uv_inv; for valid/test map to train-unique keys
    if split_name == "train":
        # ua_inv / uv_inv are for the train rows in the original train ordering.
        # We must ensure 'split' is the train object to use these arrays in order.
        # Build per-row pair indices by re-computing unique mapping via searchsorted into ua_unique_keys
        # BUT we already have ua_inv and uv_inv aligned with train ordering above:
        # Confirm lengths match
        if len(pair_ua_keys) != len(ua_inv) or len(pair_uv_keys) != len(uv_inv):
            # Fallback: remap using searchsorted (should not happen)
            ua_idx = np.searchsorted(ua_unique_keys, pair_ua_keys)
            mask = (ua_idx < len(ua_unique_keys)) & (ua_unique_keys[ua_idx] == pair_ua_keys)
            # where not found, set to -1 (should not happen in train)
            ua_idx_final = np.where(mask, ua_idx, -1)
            uv_idx = np.searchsorted(uv_unique_keys, pair_uv_keys)
            mask2 = (uv_idx < len(uv_unique_keys)) & (uv_unique_keys[uv_idx] == pair_uv_keys)
            uv_idx_final = np.where(mask2, uv_idx, -1)
        else:
            ua_idx_final = ua_inv
            uv_idx_final = uv_inv

        # leave-one-out counts and pos
        ua_tot_excl = ua_counts[ua_idx_final] - 1
        ua_pos_excl = ua_pos[ua_idx_final] - split.y
        ua_tot_excl = np.maximum(ua_tot_excl, 0)
        ua_pos_excl = np.maximum(ua_pos_excl, 0)
        rate_ua = (ua_pos_excl + 1.0) / (ua_tot_excl + 2.0)
        ua_cnt = ua_tot_excl.astype(np.float32)

        uv_tot_excl = uv_counts[uv_idx_final] - 1
        uv_pos_excl = uv_pos[uv_idx_final] - split.y
        uv_tot_excl = np.maximum(uv_tot_excl, 0)
        uv_pos_excl = np.maximum(uv_pos_excl, 0)
        rate_uv = (uv_pos_excl + 1.0) / (uv_tot_excl + 2.0)
        uv_cnt = uv_tot_excl.astype(np.float32)
    else:
        # Map keys to counts/pos using robust searchsorted mapping
        ua_row_counts, ua_row_pos = map_keys_to_counts(pair_ua_keys, ua_unique_keys, ua_counts, ua_pos)
        uv_row_counts, uv_row_pos = map_keys_to_counts(pair_uv_keys, uv_unique_keys, uv_counts, uv_pos)

        # Smoothed rate with Laplace prior
        rate_ua = ((ua_row_pos + 1.0) / (ua_row_counts + 2.0)).astype(np.float32)
        ua_cnt = ua_row_counts.astype(np.float32)
        rate_uv = ((uv_row_pos + 1.0) / (uv_row_counts + 2.0)).astype(np.float32)
        uv_cnt = uv_row_counts.astype(np.float32)

    # Log-scale counts
    author_cnt_log = np.log1p(author_cnt).astype(np.float32)
    video_cnt_log = np.log1p(video_cnt).astype(np.float32)
    user_cnt_log = np.log1p(user_cnt).astype(np.float32)
    ua_cnt_log = np.log1p(ua_cnt).astype(np.float32)
    uv_cnt_log = np.log1p(uv_cnt).astype(np.float32)

    # Assemble feature matrix
    feat_cols = [
        rate_author.astype(np.float32),
        rate_video.astype(np.float32),
        rate_user.astype(np.float32),
        author_cnt_log,
        video_cnt_log,
        user_cnt_log,
        rate_ua.astype(np.float32),
        ua_cnt_log,
        rate_uv.astype(np.float32),
        uv_cnt_log,
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
X_train = build_features(train, split_name="train")
y_train = train.y.astype(np.float32)
uids_train = train.X["user_id"].astype(np.int64)

X_valid = build_features(valid, split_name="valid")
y_valid = valid.y.astype(np.float32)
uids_valid = valid.X["user_id"].astype(np.int64)

X_test = build_features(test, split_name="test")
uids_test = test.X["user_id"].astype(np.int64)

# Helper: sort by user to create groups for lambdarank
def sort_by_user(X, y, user_ids):
    perm = np.argsort(user_ids, kind="mergesort")
    Xs = X[perm]
    ys = y[perm]
    u_sorted = user_ids[perm]
    _, counts = np.unique(u_sorted, return_counts=True)
    return Xs, ys, perm, counts

Xtr_s, ytr_s, perm_tr, group_tr = sort_by_user(X_train, y_train, uids_train)
Xva_s, yva_s, perm_va, group_va = sort_by_user(X_valid, y_valid, uids_valid)

# LightGBM lambdarank training (tuned modestly)
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 40,
    "verbose": -1,
    "seed": SEED,
    "force_row_wise": True,
}

dtrain = lgb.Dataset(Xtr_s, label=ytr_s, group=group_tr, free_raw_data=False)
dvalid = lgb.Dataset(Xva_s, label=yva_s, group=group_va, reference=dtrain, free_raw_data=False)

num_boost_round = 400
early_stopping_rounds = 40

bst = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                valid_sets=[dvalid], callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])

# Predict on validation and map back to original ordering
preds_va_sorted = bst.predict(Xva_s, num_iteration=bst.best_iteration)
preds_va = np.empty_like(preds_va_sorted)
preds_va[perm_va] = preds_va_sorted

# Save raw valid scores
save_valid_scores(preds_va)

# Evaluate validation
res = evaluate(valid.user_id, valid.y, preds_va)
primary = float(res["primary"])
gauc = float(res["gauc"])
ndcg5 = float(res["ndcg@5"])

# Print some diagnostics about pair coverage
# fraction of valid rows whose (user,author) pair was unseen in train
pair_ua_va = (uids_va * mul_author) + aids_va
pair_uv_va = (uids_va * mul_video) + vids_va
ua_found_mask = (np.searchsorted(ua_unique_keys, pair_ua_va) < len(ua_unique_keys)) & (ua_unique_keys[np.searchsorted(ua_unique_keys, pair_ua_va, side='left')] == pair_ua_va)
uv_found_mask = (np.searchsorted(uv_unique_keys, pair_uv_va) < len(uv_unique_keys)) & (uv_unique_keys[np.searchsorted(uv_unique_keys, pair_uv_va, side='left')] == pair_uv_va)
frac_ua_seen = float(np.mean(ua_found_mask))
frac_uv_seen = float(np.mean(uv_found_mask))
print("FINDINGS pair_coverage_valid " + json.dumps({"ua_seen_frac": frac_ua_seen, "uv_seen_frac": frac_uv_seen}))

# Save test predictions
preds_test = bst.predict(X_test, num_iteration=bst.best_iteration)
save_test_scores(preds_test)

# Feature importances
try:
    imps = bst.feature_importance(importance_type="gain")
    names = [f"f{i}" for i in range(X_train.shape[1])]
    imp_pairs = sorted(zip(names, imps), key=lambda x: -x[1])[:12]
    print("FINDINGS top_feature_importances " + json.dumps(imp_pairs))
except Exception as e:
    print("FINDINGS feature_importance_failed " + str(e))

t1 = time.time()
wall = float(t1 - t0)

# Final METRICS line
print(f'METRICS {{"primary": {primary:.6f}, "gauc": {gauc:.6f}, "ndcg@5": {ndcg5:.6f}, "gpu_seconds": {wall:.3f}}}')