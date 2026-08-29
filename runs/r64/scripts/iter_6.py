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

OUTDIR = os.environ.get("ITER_OUT", "")
ARTDIR = os.environ.get("RUN_ARTIFACTS", "")

def save_valid_scores(arr):
    if OUTDIR:
        np.save(os.path.join(OUTDIR, "scores_valid.npy"), np.asarray(arr, dtype=np.float64))

def save_test_scores(arr):
    if OUTDIR:
        np.save(os.path.join(OUTDIR, "scores_test.npy"), np.asarray(arr, dtype=np.float64))

# Safe cat accessor
def cat(name, s):
    return s.X[name].astype(np.int64)

# Load splits
train = load("train")
valid = load("valid")
test = load("test")  # we must not touch test.y (it doesn't exist); only use features

# Numeric fields helper
def safe_log1p_num(arr):
    a = arr.astype(np.float64)
    a = np.nan_to_num(a, nan=0.0)
    return np.log1p(a).astype(np.float32)

# Precompute basic ids and labels from TRAIN
uids_tr = cat("user_id", train)
aids_tr = cat("author_id", train)
vids_tr = cat("video_id", train)
y_tr = train.y.astype(np.int32)

user_card = int(FEATURE_CARDINALITIES.get("user_id", uids_tr.max() + 1))
author_card = int(FEATURE_CARDINALITIES.get("author_id", aids_tr.max() + 1))
video_card = int(FEATURE_CARDINALITIES.get("video_id", vids_tr.max() + 1))

# Global counts from TRAIN (allowed)
user_total = np.bincount(uids_tr, minlength=user_card).astype(np.int32)
user_pos = np.bincount(uids_tr, weights=y_tr, minlength=user_card).astype(np.int32)

author_total = np.bincount(aids_tr, minlength=author_card).astype(np.int32)
author_pos = np.bincount(aids_tr, weights=y_tr, minlength=author_card).astype(np.int32)

video_total = np.bincount(vids_tr, minlength=video_card).astype(np.int32)
video_pos = np.bincount(vids_tr, weights=y_tr, minlength=video_card).astype(np.int32)

# Build pair identifiers for user-author and user-video on TRAIN
# Use combined integer id = user * (K) + item to make unique per-pair identifier; unique count <= rows
ua_pair_tr = uids_tr.astype(np.int64) * (author_card) + aids_tr.astype(np.int64)
uv_pair_tr = uids_tr.astype(np.int64) * (video_card) + vids_tr.astype(np.int64)

# Unique mapping for train pairs
ua_uuniq, ua_inv = np.unique(ua_pair_tr, return_inverse=True)
ua_counts_unique = np.bincount(ua_inv).astype(np.int32)
ua_pos_unique = np.bincount(ua_inv, weights=y_tr).astype(np.int32)

uv_uuniq, uv_inv = np.unique(uv_pair_tr, return_inverse=True)
uv_counts_unique = np.bincount(uv_inv).astype(np.int32)
uv_pos_unique = np.bincount(uv_inv, weights=y_tr).astype(np.int32)

# Utility to compute per-row pair counts/pos for an arbitrary split given the train-unique arrays
def lookup_pair_counts(pair_ids, train_uuniq, train_counts_unique, train_pos_unique):
    # pair_ids: array per-row of combined id for that split
    # train_uuniq sorted; use searchsorted
    idx = np.searchsorted(train_uuniq, pair_ids)
    found = (idx < len(train_uuniq)) & (train_uuniq[idx] == pair_ids)
    counts = np.zeros_like(pair_ids, dtype=np.int32)
    pos = np.zeros_like(pair_ids, dtype=np.int32)
    if found.any():
        counts[found] = train_counts_unique[idx[found]]
        pos[found] = train_pos_unique[idx[found]]
    return counts, pos

# Feature builder that returns float32 matrix; for train do leave-one-out for same-split pairs
def build_features(split, split_name="train"):
    # categorical small ints
    f_tab = split.X["tab"].astype(np.int32)
    f_hour = split.X["hour"].astype(np.int32)
    f_live = split.X["is_live_streamer"].astype(np.int32)
    f_vid_authorflag = split.X["is_video_author"].astype(np.int32)

    # numeric raw fields
    nums = split.num
    f_duration = safe_log1p_num(nums["duration_ms"])
    f_fans = safe_log1p_num(nums["user_fans_user_num"])
    f_follow = safe_log1p_num(nums["user_follow_user_num"])
    f_friend = safe_log1p_num(nums["user_friend_user_num"])
    f_regdays = safe_log1p_num(nums["user_register_days"])

    # ids
    uids = split.X["user_id"].astype(np.int64)
    aids = split.X["author_id"].astype(np.int64)
    vids = split.X["video_id"].astype(np.int64)

    # author/video/user global laplace rates (train stats). For train split we do leave-one-out by subtracting row.
    # alpha/beta smoothing
    alpha = 1.0
    beta = 2.0

    if split_name == "train":
        # leave-one-out for author/video/user
        # author
        auth_tot_ex = author_total[aids] - 1
        auth_pos_ex = author_pos[aids] - split.y
        auth_tot_ex = np.maximum(auth_tot_ex, 0)
        auth_pos_ex = np.maximum(auth_pos_ex, 0)
        rate_author = (auth_pos_ex + alpha) / (auth_tot_ex + beta)
        # video
        vid_tot_ex = video_total[vids] - 1
        vid_pos_ex = video_pos[vids] - split.y
        vid_tot_ex = np.maximum(vid_tot_ex, 0)
        vid_pos_ex = np.maximum(vid_pos_ex, 0)
        rate_video = (vid_pos_ex + alpha) / (vid_tot_ex + beta)
        # user
        user_tot_ex = user_total[uids] - 1
        user_pos_ex = user_pos[uids] - split.y
        user_tot_ex = np.maximum(user_tot_ex, 0)
        user_pos_ex = np.maximum(user_pos_ex, 0)
        rate_user = (user_pos_ex + alpha) / (user_tot_ex + beta)
        # counts log
        author_cnt = np.maximum(author_total[aids] - 1, 0).astype(np.float32)
        video_cnt = np.maximum(video_total[vids] - 1, 0).astype(np.float32)
        user_cnt = np.maximum(user_total[uids] - 1, 0).astype(np.float32)
    else:
        # valid/test: use full TRAIN counts (no leakage)
        rate_author = (author_pos[aids] + alpha) / (author_total[aids] + beta)
        rate_video = (video_pos[vids] + alpha) / (video_total[vids] + beta)
        rate_user = (user_pos[uids] + alpha) / (user_total[uids] + beta)
        author_cnt = author_total[aids].astype(np.float32)
        video_cnt = video_total[vids].astype(np.float32)
        user_cnt = user_total[uids].astype(np.float32)

    author_cnt_log = np.log1p(author_cnt).astype(np.float32)
    video_cnt_log = np.log1p(video_cnt).astype(np.float32)
    user_cnt_log = np.log1p(user_cnt).astype(np.float32)

    # user-author and user-video affinity features:
    pair_ua = uids * (author_card) + aids
    pair_uv = uids * (video_card) + vids

    if split_name == "train":
        # For train, we already have per-row inverse mapping into ua unique arrays: ua_inv, uv_inv correspond to train order
        ua_counts_row = ua_counts_unique[ua_inv].astype(np.int32)
        ua_pos_row = ua_pos_unique[ua_inv].astype(np.int32)
        # leave-one-out
        ua_counts_ex = np.maximum(ua_counts_row - 1, 0).astype(np.int32)
        ua_pos_ex = np.maximum(ua_pos_row - split.y, 0).astype(np.int32)
        ua_rate = (ua_pos_ex + alpha) / (ua_counts_ex + beta)

        uv_counts_row = uv_counts_unique[uv_inv].astype(np.int32)
        uv_pos_row = uv_pos_unique[uv_inv].astype(np.int32)
        uv_counts_ex = np.maximum(uv_counts_row - 1, 0).astype(np.int32)
        uv_pos_ex = np.maximum(uv_pos_row - split.y, 0).astype(np.int32)
        uv_rate = (uv_pos_ex + alpha) / (uv_counts_ex + beta)

        ua_cnt_log = np.log1p(ua_counts_ex.astype(np.float32))
        uv_cnt_log = np.log1p(uv_counts_ex.astype(np.float32))
    else:
        # valid/test: lookup counts in train unique arrays (no leakage)
        ua_counts_row, ua_pos_row = lookup_pair_counts(pair_ua, ua_uuniq, ua_counts_unique, ua_pos_unique)
        uv_counts_row, uv_pos_row = lookup_pair_counts(pair_uv, uv_uuniq, uv_counts_unique, uv_pos_unique)
        ua_rate = (ua_pos_row + alpha) / (ua_counts_row + beta)
        uv_rate = (uv_pos_row + alpha) / (uv_counts_row + beta)
        ua_cnt_log = np.log1p(ua_counts_row.astype(np.float32))
        uv_cnt_log = np.log1p(uv_counts_row.astype(np.float32))

    # Clip rates into [0,1] floats
    rate_author = np.clip(rate_author.astype(np.float32), 0.0, 1.0)
    rate_video = np.clip(rate_video.astype(np.float32), 0.0, 1.0)
    rate_user = np.clip(rate_user.astype(np.float32), 0.0, 1.0)
    ua_rate = np.clip(ua_rate.astype(np.float32), 0.0, 1.0)
    uv_rate = np.clip(uv_rate.astype(np.float32), 0.0, 1.0)

    # Assemble feature matrix
    feat_cols = [
        rate_author,
        rate_video,
        rate_user,
        author_cnt_log,
        video_cnt_log,
        user_cnt_log,
        ua_rate,
        ua_cnt_log,
        uv_rate,
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

# Build features for train, valid, test
X_train = build_features(train, split_name="train")
y_train = train.y.astype(np.float32)
uids_train = train.user_id.astype(np.int64)

X_valid = build_features(valid, split_name="valid")
y_valid = valid.y.astype(np.float32)
uids_valid = valid.user_id.astype(np.int64)

X_test = build_features(test, split_name="valid")  # use same logic as valid/test
uids_test = test.user_id.astype(np.int64)

# Sort by user_id to create groups (LightGBM lambdarank requires contiguous groups)
def sort_by_user(X, y, user_ids):
    perm = np.argsort(user_ids, kind="mergesort")
    Xs = X[perm]
    ys = y[perm]
    u_sorted = user_ids[perm]
    _, counts = np.unique(u_sorted, return_counts=True)
    return Xs, ys, perm, counts, u_sorted

Xtr_s, ytr_s, perm_tr, group_tr, u_sorted_tr = sort_by_user(X_train, y_train, uids_train)
Xva_s, yva_s, perm_va, group_va, u_sorted_va = sort_by_user(X_valid, y_valid, uids_valid)

# Build group weights to emphasize users with more positives (GAUC weights users by positive count).
# We compute training user_pos array (from full train); create per-group weights aligned with group_tr order.
# For each unique user in u_sorted_tr (in sorted order), map to user_pos and clip to reasonable range.
unique_users_tr, start_idxs = np.unique(u_sorted_tr, return_index=True)
# Map each group to that user's positive count
group_user_pos = user_pos[unique_users_tr]
# Clip extreme weights to avoid runaway influence; use sqrt scaling then clip
group_weights = np.sqrt(group_user_pos.astype(np.float32))
group_weights = np.clip(group_weights, 1.0, 50.0)  # min 1, max 50

# Prepare LightGBM datasets with group and group_weight
dtrain = lgb.Dataset(Xtr_s, label=ytr_s, group=group_tr, group_weight=group_weights, free_raw_data=False)
dvalid = lgb.Dataset(Xva_s, label=yva_s, group=group_va, reference=dtrain, free_raw_data=False)

# LightGBM parameters for lambdarank (optimize nDCG@5 while group-weighting encourages GAUC-aligned emphasis)
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

# Predict on validation: map sorted predictions back to original ordering
preds_va_sorted = bst.predict(Xva_s, num_iteration=bst.best_iteration)
preds_va = np.empty_like(preds_va_sorted)
preds_va[perm_va] = preds_va_sorted

# Save raw valid scores (so harness trusts them)
save_valid_scores(preds_va)

# Evaluate
res_raw = evaluate(valid.user_id, valid.y, preds_va)
primary_raw = float(res_raw["primary"])
gauc_raw = float(res_raw["gauc"])
ndcg5_raw = float(res_raw["ndcg@5"])

candidates = {"lgb_ua_uv_groupweight": primary_raw}

# Optionally blend with incumbent from RUN_ARTIFACTS (grid search alphas); keep same policy as earlier runs
best_final_scores = preds_va.copy()
best_final_primary = primary_raw
best_choice_name = "lgb_ua_uv_groupweight"
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
            if best_choice_name != "lgb_ua_uv_groupweight":
                res_tmp = evaluate(valid.user_id, valid.y, best_final_scores)
                gauc_raw = float(res_tmp["gauc"])
                ndcg5_raw = float(res_tmp["ndcg@5"])
        except Exception as e:
            print("FINDINGS failed_to_load_or_blend_incumbent:", e)

print("CANDIDATES " + json.dumps(candidates))

# Save the chosen valid scores
save_valid_scores(best_final_scores)

# Final evaluation
res_final = evaluate(valid.user_id, valid.y, best_final_scores)
primary = float(res_final["primary"])
gauc = float(res_final["gauc"])
ndcg5 = float(res_final["ndcg@5"])

# Produce test predictions
preds_test = bst.predict(X_test, num_iteration=bst.best_iteration)

# If blended with incumbent, apply same blend weight to test (only if incumbent test exists)
if best_choice_name.startswith("blend_incumbent") and ARTDIR:
    inc_test_path = os.path.join(ARTDIR, "incumbent_test_scores.npy")
    if os.path.exists(inc_test_path):
        try:
            incumbent_test = np.load(inc_test_path)
            preds_test = best_alpha * incumbent_test + (1.0 - best_alpha) * preds_test
        except Exception as e:
            print("FINDINGS failed_to_load_incumbent_test_for_blending:", e)

save_test_scores(preds_test)

# Report some findings about the new features
try:
    # fraction of valid rows where user-author history exists in train
    uids_va = cat("user_id", valid)
    aids_va = cat("author_id", valid)
    pair_va = uids_va * (author_card) + aids_va
    exists_in_train = np.searchsorted(ua_uuniq, pair_va)
    mask = (exists_in_train < len(ua_uuniq)) & (ua_uuniq[exists_in_train] == pair_va)
    frac = float(mask.mean())
    print(f"FINDINGS frac_valid_rows_with_ua_history {frac:.4f}")
    # top feature importances
    imps = bst.feature_importance(importance_type="gain")
    names = ["rate_author","rate_video","rate_user","author_cnt_log","video_cnt_log","user_cnt_log",
             "ua_rate","ua_cnt_log","uv_rate","uv_cnt_log","duration","fans","follow","friend","regdays",
             "tab","hour","is_live","is_video_author"]
    imp_pairs = sorted(list(zip(names, imps)), key=lambda x: -x[1])[:10]
    print("FINDINGS top_feature_importances " + json.dumps(imp_pairs))
except Exception as e:
    print("FINDINGS feature_reporting_failed:", e)

t1 = time.time()
wall = float(t1 - t0)

print(f'METRICS {{"primary": {primary:.6f}, "gauc": {gauc:.6f}, "ndcg@5": {ndcg5:.6f}, "gpu_seconds": {wall:.3f}}}')