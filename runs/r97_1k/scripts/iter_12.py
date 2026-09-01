import os
import gc
import json
import time
import warnings
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
    "onehot_feat7",
    "music_type",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

RERANKERS = [
    "identity",
    "video_dedup",
    "creator_quota",
    "author_mmr",
    "tag_xquad",
    "hierarchical_coverage",
]


def make_matrix(split, row_mask=None):
    cols = []

    for field in CAT_FIELDS:
        x = np.asarray(split.X[field], dtype=np.float32)
        if row_mask is not None:
            x = x[row_mask]
        cols.append(x)

    for field in NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float32)
        if row_mask is not None:
            x = x[row_mask]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        cols.append(x.astype(np.float32))

    return np.column_stack(cols).astype(np.float32, copy=False)


def rerank_top5(user_ids, scores, author_ids, video_ids, tag_ids, mode,
                pool_size=40):
    scores = np.asarray(scores, dtype=np.float32)
    if mode == "identity":
        return scores.copy()

    user_ids = np.asarray(user_ids)
    author_ids = np.asarray(author_ids)
    video_ids = np.asarray(video_ids)
    tag_ids = np.asarray(tag_ids)

    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    # Descending score within ascending user. Row position deterministically
    # resolves tied scores.
    order = np.lexsort((rows, -scores, user_ids))
    sorted_users = user_ids[order]

    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )

    result = scores.copy()

    for gi in range(len(boundaries) - 1):
        lo = int(boundaries[gi])
        hi = int(boundaries[gi + 1])
        group_order = order[lo:hi]
        m = min(pool_size, len(group_order))

        if m <= 1:
            continue

        pool = group_order[:m]
        pa = author_ids[pool]
        pv = video_ids[pool]
        pt = tag_ids[pool]

        # Position relevance keeps all rerankers conservative. A penalty of
        # 0.10 is approximately four positions in a 40-item candidate pool.
        relevance = 1.0 - np.arange(m, dtype=np.float32) / float(max(m, 1))
        available = np.ones(m, dtype=bool)
        chosen_local = []

        selected_authors = set()
        selected_videos = set()
        selected_tags = set()

        choose_count = min(5, m)

        for _ in range(choose_count):
            utility = relevance.copy()
            utility[~available] = -1e9

            if mode == "video_dedup":
                if selected_videos:
                    duplicate = np.fromiter(
                        (
                            (int(v) != 0 and int(v) in selected_videos)
                            for v in pv
                        ),
                        dtype=bool,
                        count=m,
                    )
                    utility[duplicate] -= 2.0

            elif mode == "creator_quota":
                if selected_authors:
                    duplicate = np.fromiter(
                        (
                            (int(a) != 0 and int(a) in selected_authors)
                            for a in pa
                        ),
                        dtype=bool,
                        count=m,
                    )
                    utility[duplicate] -= 2.0

            elif mode == "author_mmr":
                if selected_authors:
                    duplicate = np.fromiter(
                        (
                            (int(a) != 0 and int(a) in selected_authors)
                            for a in pa
                        ),
                        dtype=bool,
                        count=m,
                    )
                    utility[duplicate] -= 0.11

            elif mode == "tag_xquad":
                novel = np.fromiter(
                    (
                        (int(t) != 0 and int(t) not in selected_tags)
                        for t in pt
                    ),
                    dtype=bool,
                    count=m,
                )
                utility += 0.09 * novel.astype(np.float32)

            elif mode == "hierarchical_coverage":
                author_duplicate = np.fromiter(
                    (
                        (
                            int(a) != 0
                            and int(a) in selected_authors
                        )
                        for a in pa
                    ),
                    dtype=bool,
                    count=m,
                )
                video_duplicate = np.fromiter(
                    (
                        (
                            int(v) != 0
                            and int(v) in selected_videos
                        )
                        for v in pv
                    ),
                    dtype=bool,
                    count=m,
                )
                tag_novel = np.fromiter(
                    (
                        (
                            int(t) != 0
                            and int(t) not in selected_tags
                        )
                        for t in pt
                    ),
                    dtype=bool,
                    count=m,
                )
                utility -= 0.075 * author_duplicate.astype(np.float32)
                utility -= 0.20 * video_duplicate.astype(np.float32)
                utility += 0.045 * tag_novel.astype(np.float32)

            else:
                raise ValueError("Unknown reranker: " + mode)

            pick = int(np.argmax(utility))
            chosen_local.append(pick)
            available[pick] = False

            a = int(pa[pick])
            v = int(pv[pick])
            t = int(pt[pick])
            if a != 0:
                selected_authors.add(a)
            if v != 0:
                selected_videos.add(v)
            if t != 0:
                selected_tags.add(t)

        chosen_set = set(chosen_local)
        remaining_local = [j for j in range(m) if j not in chosen_set]
        new_local_order = chosen_local + remaining_local
        new_pool_order = pool[np.asarray(new_local_order, dtype=np.int64)]

        # Reassign the original descending score values. Thus only ordering
        # within the small candidate pool changes; score scale and every
        # below-pool comparison remain intact.
        sorted_values = scores[pool].copy()
        result[new_pool_order] = sorted_values

    return result


# -------------------------------------------------------------------------
# Select the reranking algorithm exclusively on a temporal holdout contained
# in TRAIN. Validation is not used for selection.
# -------------------------------------------------------------------------
train = load("train")
train_dates = np.asarray(train.date, dtype=np.int32)
train_labels = np.asarray(train.y, dtype=np.int8)

fit_mask = train_dates <= 20220419
hold_mask = train_dates >= 20220420

X_fit = make_matrix(train, fit_mask)
y_fit = train_labels[fit_mask].astype(np.float32)

fit_dates = train_dates[fit_mask]
age = (20220419 - fit_dates).astype(np.float32)
sample_weight = np.power(0.5, age / 4.0).astype(np.float32)
sample_weight /= np.mean(sample_weight)

dtrain = lgb.Dataset(
    X_fit,
    label=y_fit,
    weight=sample_weight,
    categorical_feature=list(range(len(CAT_FIELDS))),
    free_raw_data=True,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.07,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 250,
    "feature_fraction": 0.88,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "cat_smooth": 30.0,
    "cat_l2": 12.0,
    "num_threads": max(1, os.cpu_count() or 1),
    "seed": 731,
    "verbose": -1,
}

selector_model = lgb.train(
    params,
    dtrain,
    num_boost_round=220,
)

del X_fit, y_fit, dtrain, sample_weight, fit_dates
gc.collect()

X_hold = make_matrix(train, hold_mask)
hold_base = selector_model.predict(
    X_hold, num_iteration=selector_model.current_iteration()
).astype(np.float32)

hold_uid = np.asarray(train.user_id)[hold_mask]
hold_y = train_labels[hold_mask]
hold_author = np.asarray(train.X["author_id"])[hold_mask]
hold_video = np.asarray(train.X["video_id"])[hold_mask]
hold_tag = np.asarray(train.X["tag"])[hold_mask]

holdout_scores = {}
holdout_predictions = {}

for mode in RERANKERS:
    pred = rerank_top5(
        hold_uid,
        hold_base,
        hold_author,
        hold_video,
        hold_tag,
        mode,
    )
    met = evaluate(hold_uid, hold_y, pred)
    holdout_scores[mode] = float(met["primary"])
    holdout_predictions[mode] = pred

selected_mode = max(holdout_scores, key=holdout_scores.get)
identity_holdout = holdout_scores["identity"]
selected_holdout = holdout_scores[selected_mode]

print(
    "FINDINGS holdout_candidates="
    + json.dumps(holdout_scores, sort_keys=True),
    flush=True,
)
print(
    "FINDINGS selected_on_train_holdout=%s holdout_delta_vs_identity=%.6f"
    % (selected_mode, selected_holdout - identity_holdout),
    flush=True,
)

del X_hold, holdout_predictions, hold_base
del hold_uid, hold_y, hold_author, hold_video, hold_tag
gc.collect()

# -------------------------------------------------------------------------
# Apply all fixed rerankers to the trusted incumbent for diagnostics, but
# report the mode already selected above on the TRAIN-only holdout.
# -------------------------------------------------------------------------
valid = load("valid")
valid_uid = np.asarray(valid.user_id)
valid_y = np.asarray(valid.y)

inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)

has_incumbent = bool(
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
)

if has_incumbent:
    valid_base = np.asarray(
        np.load(inc_valid_path, mmap_mode="r"),
        dtype=np.float32,
    ).copy()
else:
    # Legal fallback: the selector was fitted only on a subset of TRAIN.
    X_valid_fallback = make_matrix(valid)
    valid_base = selector_model.predict(
        X_valid_fallback,
        num_iteration=selector_model.current_iteration(),
    ).astype(np.float32)
    del X_valid_fallback
    gc.collect()

valid_author = np.asarray(valid.X["author_id"])
valid_video = np.asarray(valid.X["video_id"])
valid_tag = np.asarray(valid.X["tag"])

candidate_scores = {}
selected_valid = None

for mode in RERANKERS:
    pred = rerank_top5(
        valid_uid,
        valid_base,
        valid_author,
        valid_video,
        valid_tag,
        mode,
    )
    met = evaluate(valid_uid, valid_y, pred)
    candidate_scores[mode] = float(met["primary"])

    if mode == selected_mode:
        selected_valid = pred.copy()

final_metrics = evaluate(valid_uid, valid_y, selected_valid)

print(
    "FINDINGS validation_selected_delta_vs_identity=%.6f"
    % (
        candidate_scores[selected_mode]
        - candidate_scores["identity"]
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(selected_valid, dtype=np.float64),
    )
    # The unmodified external score is saved as the raw component so the
    # effect of this script's slate transformation remains inspectable.
    np.save(
        os.path.join(OUT, "scores_valid_raw.npy"),
        np.asarray(valid_base, dtype=np.float64),
    )

del selected_valid, valid_base
del valid_author, valid_video, valid_tag
gc.collect()

# -------------------------------------------------------------------------
# Test scoring: same train-holdout-selected transformation, with no test
# labels read or used.
# -------------------------------------------------------------------------
test = load("test")

if has_incumbent:
    test_base = np.asarray(
        np.load(inc_test_path, mmap_mode="r"),
        dtype=np.float32,
    ).copy()
else:
    X_test_fallback = make_matrix(test)
    test_base = selector_model.predict(
        X_test_fallback,
        num_iteration=selector_model.current_iteration(),
    ).astype(np.float32)
    del X_test_fallback
    gc.collect()

test_scores = rerank_top5(
    np.asarray(test.user_id),
    test_base,
    np.asarray(test.X["author_id"]),
    np.asarray(test.X["video_id"]),
    np.asarray(test.X["tag"]),
    selected_mode,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)