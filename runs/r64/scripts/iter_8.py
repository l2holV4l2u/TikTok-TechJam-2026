#!/usr/bin/env python3
# Fixed attempt: add leave-one-out user×author and user×video affinity features.
# The previous failure came from unsafe searchsorted indexing when mapping pair keys;
# here we compute pair statistics for train via np.unique+inverse and map valid/test
# by a safe dictionary lookup (no out-of-bounds indexing).
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

OUTDIR = os.environ.get("ITER_OUT", "")
ARTDIR = os.environ.get("RUN_ARTIFACTS", "")

def save_valid_scores(arr):
    if OUTDIR:
        np.save(os.path.join(OUTDIR, "scores_valid.npy"), np.asarray(arr, dtype=np.float64))

def save_test_scores(arr):
    if OUTDIR:
        np.save(os.path.join(OUTDIR, "scores_test.npy"), np.asarray(arr, dtype=np.float64))

# --- Load splits
train = load("train")
valid = load("valid")
test = load("test")

# --- Helper lambdas
def cat(name, s):
    return s.X[name].astype(np.int64)

# numeric log1p safe
def safe_log1p_arr(arr):
    a = arr.astype(np.float64)
    a = np.nan_to_num(a, nan=0.0)
    return np.log1p(a).astype(np.float32)

# --- Basic per-entity counts from TRAIN (allowed)
y_train = train.y.astype(np.int32)
uids_train = cat("user_id", train)
aids_train = cat("author_id", train)
vids_train = cat("video_id", train)

user_card = int(FEATURE_CARDINALITIES.get("user_id", uids_train.max() + 1))
author_card = int(FEATURE_CARDINALITIES.get("author_id", aids_train.max() + 1))
video_card = int(FEATURE_CARDINALITIES.get("video_id", vids_train.max() + 1))

user_total = np.bincount(uids_train, minlength=user_card).astype(np.int32)
user_pos = np.bincount(uids_train, weights=y_train, minlength=user_card).astype(np.int32)

author_total = np.bincount(aids_train, minlength=author_card).astype(np.int32)
author_pos = np.bincount(aids_train, weights=y_train, minlength=author_card).astype(np.int32)

video_total = np.bincount(vids_train, minlength=video_card).astype(np.int32)
video_pos = np.bincount(vids_train, weights=y_train, minlength=video_card).astype(np.int32)

# --- Build pair keys for user×author and user×video on TRAIN, compute counts and pos sums
# Use 64-bit composed keys (uid << 32) | kid to avoid giant dense arrays.
def compose_key(u, v):
    return (u.astype(np.int64) << 32) | v.astype(np.int64)

# user x author
pair_ua_train = compose_key(uids_train, aids_train)
unique_ua_keys, inv_ua = np.unique(pair_ua_train, return_inverse=True)
ua_counts_per_key = np.bincount(inv_ua).astype(np.int32)
ua_pos_per_key = np.bincount(inv_ua, weights=y_train).astype(np.int32)

# user x video
pair_uv_train = compose_key(uids_train, vids_train)
unique_uv_keys, inv_uv = np.unique(pair_uv_train, return_inverse=True)
uv_counts_per_key = np.bincount(inv_uv).astype(np.int32)
uv_pos_per_key = np.bincount(inv_uv, weights=y_train).astype(np.int32)

# Build dicts for mapping valid/test keys -> index in unique arrays (safe lookup)
ua_key_to_idx = {int(k): int(i) for i, k in enumerate(unique_ua_keys)}
uv_key_to_idx = {int(k): int(i) for i, k in enumerate(unique_uv_keys)}

# Laplace smoothing defaults
ALPHA = 1.0
BETA = 2.0

# --- Feature builder using the above precomputed stats
def build_features_for_split(split, is_train):
    # categorical small ints
    f_tab = split.X["tab"].astype(np.int32)
    f_hour = split.X["hour"].astype(np.int32)
    f_live = split.X["is_live_streamer"].astype(np.int32)
    f_vid_authorflag = split.X["is_video_author"].astype(np.int32)

    nums = split.num
    f_duration = safe_log1p_arr(nums["duration_ms"])
    f_fans = safe_log1p_arr(nums["user_fans_user_num"])
    f_follow = safe_log1p_arr(nums["user_follow_user_num"])
    f_friend = safe_log1p_arr(nums["user_friend_user_num"])
    f_regdays = safe_log1p_arr(nums["user_register_days"])

    uids = split.X["user_id"].astype(np.int64)
    aids = split.X["author_id"].astype(np.int64)
    vids = split.X["video_id"].astype(np.int64)

    # entity-level smoothed rates (author/video/user)
    if is_train:
        # leave-one-out for train rows
        rates_author = (author_pos[aids].astype(np.float32) - split.y.astype(np.float32) + ALPHA) / \
                       (np.maximum(author_total[aids] - 1, 0).astype(np.float32) + BETA)
        rates_video = (video_pos[vids].astype(np.float32) - split.y.astype(np.float32) + ALPHA) / \
                      (np.maximum(video_total[vids] - 1, 0).astype(np.float32) + BETA)
        rates_user = (user_pos[uids].astype(np.float32) - split.y.astype(np.float32) + ALPHA) / \
                     (np.maximum(user_total[uids] - 1, 0).astype(np.float32) + BETA)

        author_cnt = np.maximum(author_total[aids] - 1, 0).astype(np.float32)
        video_cnt = np.maximum(video_total[vids] - 1, 0).astype(np.float32)
        user_cnt = np.maximum(user_total[uids] - 1, 0).astype(np.float32)
    else:
        # valid/test use full-train stats (no leakage)
        rates_author = (author_pos[aids].astype(np.float32) + ALPHA) / (author_total[aids].astype(np.float32) + BETA)
        rates_video = (video_pos[vids].astype(np.float32) + ALPHA) / (video_total[vids].astype(np.float32) + BETA)
        rates_user = (user_pos[uids].astype(np.float32) + ALPHA) / (user_total[uids].astype(np.float32) + BETA)

        author_cnt = author_total[aids].astype(np.float32)
        video_cnt = video_total[vids].astype(np.float32)
        user_cnt = user_total[uids].astype(np.float32)

    author_cnt_log = np.log1p(author_cnt).astype(np.float32)
    video_cnt_log = np.log1p(video_cnt).astype(np.float32)
    user_cnt_log = np.log1p(user_cnt).astype(np.float32)

    # --- Pair-level user×author and user×video features
    # Compose keys for the split
    pair_ua_keys = compose_key(uids, aids)
    pair_uv_keys = compose_key(uids, vids)

    if is_train:
        # For train: we already have inv arrays mapping each row to its unique key index
        # Use those inverse mappings computed earlier by np.unique on train order.
        # But here the split is train; we can reconstruct inv by doing np.searchsorted on unique arrays,
        # however to avoid subtle ordering assumptions we will recompute inverse for the train slice
        # via np.unique with return_inverse. It is fast enough because it's the same as earlier computation.
        # However to avoid recomputing, detect identity by checking length equality and reference equality.
        # Simpler path: use mapping from pair key to index (ua_key_to_idx / uv_key_to_idx) then lookup per-row index
        # via vectorized list comprehension (safe).
        idxs_ua = np.fromiter((ua_key_to_idx.get(int(k), -1) for k in pair_ua_keys), dtype=np.int64)
        idxs_uv = np.fromiter((uv_key_to_idx.get(int(k), -1) for k in pair_uv_keys), dtype=np.int64)

        # Retrieve counts/pos and subtract the row itself (leave-one-out)
        ua_counts = np.where(idxs_ua >= 0, ua_counts_per_key[idxs_ua], 0).astype(np.int32)
        ua_pos = np.where(idxs_ua >= 0, ua_pos_per_key[idxs_ua], 0).astype(np.int32)

        uv_counts = np.where(idxs_uv >= 0, uv_counts_per_key[idxs_uv], 0).astype(np.int32)
        uv_pos = np.where(idxs_uv >= 0, uv_pos_per_key[idxs_uv], 0).astype(np.int32)

        ua_cnt_excl = np.maximum(ua_counts - 1, 0).astype(np.float32)
        ua_pos_excl = np.maximum(ua_pos - split.y.astype(np.int32), 0).astype(np.float32)
        uv_cnt_excl = np.maximum(uv_counts - 1, 0).astype(np.float32)
        uv_pos_excl = np.maximum(uv_pos - split.y.astype(np.int32), 0).astype(np.float32)

        ua_rate = (ua_pos_excl + ALPHA) / (ua_cnt_excl + BETA)
        uv_rate = (uv_pos_excl + ALPHA) / (uv_cnt_excl + BETA)
        ua_cnt_log = np.log1p(ua_cnt_excl).astype(np.float32)
        uv_cnt_log = np.log1p(uv_cnt_excl).astype(np.float32)
    else:
        # valid/test: map via dict lookup, where missing -> zero counts/pos
        idxs_ua = np.fromiter((ua_key_to_idx.get(int(k), -1) for k in pair_ua_keys), dtype=np.int64)
        idxs_uv = np.fromiter((uv_key_to_idx.get(int(k), -1) for k in pair_uv_keys), dtype=np.int64)

        ua_counts = np.where(idxs_ua >= 0, ua_counts_per_key[idxs_ua], 0).astype(np.float32)
        ua_pos = np.where(idxs_ua >= 0, ua_pos_per_key[idxs_ua], 0).astype(np.float32)
        uv_counts = np.where(idxs_uv >= 0, uv_counts_per_key[idxs_uv], 0).astype(np.float32)
        uv_pos = np.where(idxs_uv >= 0, uv_pos_per_key[idxs_uv], 0).astype(np.float32)

        ua_rate = (ua_pos + ALPHA) / (ua_counts + BETA)
        uv_rate = (uv_pos + ALPHA) / (uv_counts + BETA)
        ua_cnt_log = np.log1p(ua_counts).astype(np.float32)
        uv_cnt_log = np.log1p(uv_counts).astype(np.float32)

    # Assemble feature matrix
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
        # pair features
        ua_rate.astype(np.float32),
        ua_cnt_log.astype(np.float32),
        uv_rate.astype(np.float32),
        uv_cnt_log.astype(np.float32),
    ]
    X = np.vstack([c.reshape(-1) for c in feat_cols]).T.astype(np.float32)
    return X

# Build matrices
X_train = build_features_for_split(train, is_train=True)
y_tr = train.y.astype(np.float32)
uids_tr = train.user_id.astype(np.int64)

X_valid = build_features_for_split(valid, is_train=False)
y_va = valid.y.astype(np.float32)
uids_va = valid.user_id.astype(np.int64)

X_test = build_features_for_split(test, is_train=False)
uids_te = test.user_id.astype(np.int64)

# Utility: sort by user for lambdarank groups
def sort_by_user(X, y, user_ids):
    perm = np.argsort(user_ids, kind="mergesort")
    Xs = X[perm]
    ys = y[perm]
    u_sorted = user_ids[perm]
    _, counts = np.unique(u_sorted, return_counts=True)
    return Xs, ys, perm, counts

Xtr_s, ytr_s, perm_tr, group_tr = sort_by_user(X_train, y_tr, uids_tr)
Xva_s, yva_s, perm_va, group_va = sort_by_user(X_valid, y_va, uids_va)

# LightGBM lambdarank
dtrain = lgb.Dataset(Xtr_s, label=ytr_s, group=group_tr, free_raw_data=False)
dvalid = lgb.Dataset(Xva_s, label=yva_s, group=group_va, reference=dtrain, free_raw_data=False)

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

num_boost_round = 300
early_stopping_rounds = 40

bst = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                valid_sets=[dvalid], callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])

# Predict on valid and map back to original ordering
preds_va_sorted = bst.predict(Xva_s, num_iteration=bst.best_iteration)
preds_va = np.empty_like(preds_va_sorted)
preds_va[perm_va] = preds_va_sorted

# Save raw valid scores
save_valid_scores(preds_va)

# Evaluate
res_raw = evaluate(valid.user_id, valid.y, preds_va)
primary_raw = float(res_raw["primary"])
gauc_raw = float(res_raw["gauc"])
ndcg5_raw = float(res_raw["ndcg@5"])

candidates = {"lgb_with_pair_stats": primary_raw}

# Try blending with incumbent if available (safe)
best_final_scores = preds_va.copy()
best_final_primary = primary_raw
best_choice_name = "lgb_with_pair_stats"
best_alpha = 0.0

if ARTDIR:
    inc_path = os.path.join(ARTDIR, "incumbent_valid_scores.npy")
    if os.path.exists(inc_path):
        try:
            incumbent_valid = np.load(inc_path)
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
        except Exception as e:
            print("FINDINGS failed_to_load_or_blend_incumbent: " + str(e))

print("CANDIDATES " + json.dumps(candidates))

# Save chosen valid scores
save_valid_scores(best_final_scores)

# Re-evaluate final
res_final = evaluate(valid.user_id, valid.y, best_final_scores)
primary = float(res_final["primary"])
gauc = float(res_final["gauc"])
ndcg5 = float(res_final["ndcg@5"])

# Predict test scores
preds_test = bst.predict(X_test, num_iteration=bst.best_iteration)

# If blended with incumbent, apply same alpha to test using incumbent_test if available
if best_choice_name.startswith("blend_incumbent") and ARTDIR:
    inc_test_path = os.path.join(ARTDIR, "incumbent_test_scores.npy")
    if os.path.exists(inc_test_path):
        try:
            incumbent_test = np.load(inc_test_path)
            preds_test = best_alpha * incumbent_test + (1.0 - best_alpha) * preds_test
        except Exception as e:
            print("FINDINGS failed_to_load_incumbent_test_for_blending: " + str(e))

save_test_scores(preds_test)

# Feature importances (gain)
try:
    imps = bst.feature_importance(importance_type="gain")
    names = [f"f{i}" for i in range(X_train.shape[1])]
    imp_pairs = sorted(zip(names, imps), key=lambda x: -x[1])[:12]
    print("FINDINGS top_feature_importances " + json.dumps(imp_pairs))
except Exception as e:
    print("FINDINGS failed_to_report_feature_importance: " + str(e))

t1 = time.time()
wall = float(t1 - t0)

# Final METRICS line (exact format required)
print(f'METRICS {{"primary": {primary:.6f}, "gauc": {gauc:.6f}, "ndcg@5": {ndcg5:.6f}, "gpu_seconds": {wall:.3f}}}')