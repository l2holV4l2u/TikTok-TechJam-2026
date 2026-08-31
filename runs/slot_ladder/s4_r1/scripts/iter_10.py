import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    # Stable row index is the final tie breaker.
    order = np.lexsort(
        (np.arange(n, dtype=np.int64), scores, user_ids)
    )
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.int64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked[multi] = (
        positions[multi] / (repeated_lengths[multi] - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def ordered_context(split):
    """Metadata-only features derived from each user's logged feed order."""
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    n = len(users)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    ou = users[order]
    ot = times[order]

    user_reset = np.r_[True, ou[1:] != ou[:-1]]
    user_starts = np.flatnonzero(user_reset)
    user_ends = np.r_[user_starts[1:], n]
    user_lengths = user_ends - user_starts

    repeated_user_starts = np.repeat(user_starts, user_lengths)
    repeated_user_lengths = np.repeat(user_lengths, user_lengths)
    user_position = np.arange(n, dtype=np.int64) - repeated_user_starts

    prev_gap = np.zeros(n, dtype=np.float64)
    if n > 1:
        gap = (ot[1:] - ot[:-1]).astype(np.float64) / 1000.0
        same_user = ou[1:] == ou[:-1]
        prev_gap[1:] = np.where(same_user, np.maximum(gap, 0.0), 0.0)

    next_gap = np.zeros(n, dtype=np.float64)
    if n > 1:
        gap = (ot[1:] - ot[:-1]).astype(np.float64) / 1000.0
        same_user = ou[1:] == ou[:-1]
        next_gap[:-1] = np.where(same_user, np.maximum(gap, 0.0), 0.0)

    # A new session starts after 30 minutes of inactivity.
    session_reset = user_reset | (prev_gap > 1800.0)
    session_starts = np.flatnonzero(session_reset)
    session_ends = np.r_[session_starts[1:], n]
    session_lengths = session_ends - session_starts
    repeated_session_starts = np.repeat(session_starts, session_lengths)
    repeated_session_lengths = np.repeat(session_lengths, session_lengths)
    session_position = np.arange(n, dtype=np.int64) - repeated_session_starts

    ordered_features = np.column_stack(
        [
            np.log1p(user_position).astype(np.float32),
            (
                user_position /
                np.maximum(repeated_user_lengths - 1, 1)
            ).astype(np.float32),
            np.log1p(repeated_user_lengths).astype(np.float32),
            np.log1p(session_position).astype(np.float32),
            (
                session_position /
                np.maximum(repeated_session_lengths - 1, 1)
            ).astype(np.float32),
            np.log1p(repeated_session_lengths).astype(np.float32),
            np.log1p(np.minimum(prev_gap, 86400.0)).astype(np.float32),
            np.log1p(np.minimum(next_gap, 86400.0)).astype(np.float32),
            session_reset.astype(np.float32),
        ]
    )

    result = np.empty_like(ordered_features)
    result[order] = ordered_features
    return result


CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "hour",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def build_tree_matrix(split_name, split):
    columns = []

    for field in CAT_FIELDS:
        columns.append(np.asarray(split.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        raw = np.asarray(split.num[field], dtype=np.float32)
        columns.append(np.log1p(np.maximum(raw, 0.0)).astype(np.float32))

    for entity in ["video_id", "author_id"]:
        histories = historical_features(split_name, key=entity)
        for name in sorted(histories):
            values = np.asarray(histories[name], dtype=np.float32)
            columns.append(values)

    context = ordered_context(split)
    for j in range(context.shape[1]):
        columns.append(context[:, j])

    return np.ascontiguousarray(
        np.column_stack(columns), dtype=np.float32
    )


x_train_tree = build_tree_matrix("train", train)
x_valid_tree = build_tree_matrix("valid", valid)
x_test_tree = build_tree_matrix("test", test)

train_dates = np.asarray(train.date, dtype=np.int64)
tree_weights = np.exp2(
    (train_dates - train_dates.max()).astype(np.float32) / 4.0
)
tree_weights /= tree_weights.mean()

tree_dataset = lgb.Dataset(
    x_train_tree,
    label=y_train.astype(np.float32),
    weight=tree_weights,
    categorical_feature=list(range(len(CAT_FIELDS))),
    free_raw_data=True,
)

tree_params = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "num_threads": min(8, os.cpu_count() or 1),
    "verbose": -1,
}

tree_model = lgb.train(
    tree_params,
    tree_dataset,
    num_boost_round=320,
)

tree_valid = tree_model.predict(
    x_valid_tree, num_iteration=tree_model.current_iteration()
).astype(np.float64)
tree_test = tree_model.predict(
    x_test_tree, num_iteration=tree_model.current_iteration()
).astype(np.float64)

del x_train_tree, x_valid_tree, x_test_tree, tree_dataset


# ----------------------------------------------------------------------
# Pairwise latent model: positives are compared only with negatives that
# the same user actually received, aligning training with logged-slate AUC.
# ----------------------------------------------------------------------
train_user = np.asarray(train.X["user_id"], dtype=np.int64)
train_video = np.asarray(train.X["video_id"], dtype=np.int64)
valid_user = np.asarray(valid.X["user_id"], dtype=np.int64)
valid_video = np.asarray(valid.X["video_id"], dtype=np.int64)
test_user = np.asarray(test.X["user_id"], dtype=np.int64)
test_video = np.asarray(test.X["video_id"], dtype=np.int64)

n_users = int(
    max(train_user.max(), valid_user.max(), test_user.max()) + 1
)
n_videos = int(
    max(train_video.max(), valid_video.max(), test_video.max()) + 1
)

negative_rows = np.flatnonzero(y_train == 0)
negative_users = train_user[negative_rows]
neg_order = np.argsort(negative_users, kind="stable")
negative_rows_sorted = negative_rows[neg_order]
negative_users_sorted = negative_users[neg_order]

neg_counts = np.bincount(
    negative_users_sorted, minlength=n_users
).astype(np.int64)
neg_starts = np.cumsum(
    np.r_[0, neg_counts[:-1]], dtype=np.int64
)

positive_rows = np.flatnonzero(
    (y_train == 1) & (neg_counts[train_user] > 0)
)
positive_users = train_user[positive_rows]
positive_videos = train_video[positive_rows]
positive_weights = np.exp2(
    (
        train_dates[positive_rows] - train_dates.max()
    ).astype(np.float32) / 4.0
)
positive_weights = positive_weights.astype(np.float32)


class PairwiseMF(nn.Module):
    def __init__(self, users, videos, rank=40):
        super().__init__()
        self.user_embedding = nn.Embedding(users, rank)
        self.video_embedding = nn.Embedding(videos, rank)
        self.video_bias = nn.Embedding(videos, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.025)
        nn.init.normal_(self.video_embedding.weight, std=0.025)
        nn.init.zeros_(self.video_bias.weight)

    def score(self, users, videos):
        interaction = (
            self.user_embedding(users) *
            self.video_embedding(videos)
        ).sum(dim=1)
        return interaction + self.video_bias(videos).squeeze(1)


mf = PairwiseMF(n_users, n_videos)
optimizer = torch.optim.Adam(
    mf.parameters(), lr=0.006, weight_decay=2e-6
)
rng = np.random.default_rng(SEED + 10)
pair_batch = 16384

for epoch in range(5):
    order = rng.permutation(len(positive_rows))
    for lo in range(0, len(order), pair_batch):
        idx = order[lo:lo + pair_batch]
        users_np = positive_users[idx]
        pos_np = positive_videos[idx]

        offsets = (
            rng.random(len(idx)) * neg_counts[users_np]
        ).astype(np.int64)
        selected_negative_rows = negative_rows_sorted[
            neg_starts[users_np] + offsets
        ]
        neg_np = train_video[selected_negative_rows]

        users_t = torch.from_numpy(users_np)
        pos_t = torch.from_numpy(pos_np)
        neg_t = torch.from_numpy(neg_np)
        weights_t = torch.from_numpy(positive_weights[idx])

        optimizer.zero_grad(set_to_none=True)
        margin = mf.score(users_t, pos_t) - mf.score(users_t, neg_t)
        losses = torch.nn.functional.softplus(-margin)
        loss = (losses * weights_t).sum() / weights_t.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mf.parameters(), 5.0)
        optimizer.step()


@torch.inference_mode()
def predict_mf(users, videos):
    mf.eval()
    result = np.empty(len(users), dtype=np.float64)
    batch = 32768
    for lo in range(0, len(users), batch):
        hi = min(lo + batch, len(users))
        values = mf.score(
            torch.from_numpy(users[lo:hi]),
            torch.from_numpy(videos[lo:hi]),
        )
        result[lo:hi] = values.cpu().numpy().astype(np.float64)
    return result


mf_valid = predict_mf(valid_user, valid_video)
mf_test = predict_mf(test_user, test_video)


# ----------------------------------------------------------------------
# Non-parametric first-order positive transition model. It uses only the
# last positive training video for a user and recency-weighted transitions
# between consecutive positive videos.
# ----------------------------------------------------------------------
row_index = np.arange(len(y_train), dtype=np.int64)
chrono = np.lexsort(
    (
        row_index,
        np.asarray(train.time_ms, dtype=np.int64),
        train_user,
    )
)
positive_chrono = chrono[y_train[chrono] == 1]
pc_users = train_user[positive_chrono]
pc_videos = train_video[positive_chrono]
pc_dates = train_dates[positive_chrono]

same_previous_user = np.r_[
    False, pc_users[1:] == pc_users[:-1]
]
transition_from = pc_videos[:-1][same_previous_user[1:]]
transition_to = pc_videos[1:][same_previous_user[1:]]
transition_dates = pc_dates[1:][same_previous_user[1:]]

transition_weight = np.exp2(
    (transition_dates - train_dates.max()).astype(np.float64) / 4.0
)
pair_keys = (
    transition_from.astype(np.int64) * np.int64(n_videos) +
    transition_to.astype(np.int64)
)

if len(pair_keys):
    pair_order = np.argsort(pair_keys, kind="stable")
    sorted_keys = pair_keys[pair_order]
    sorted_weights = transition_weight[pair_order]
    unique_keys, starts = np.unique(sorted_keys, return_index=True)
    key_weights = np.add.reduceat(sorted_weights, starts)
else:
    unique_keys = np.empty(0, dtype=np.int64)
    key_weights = np.empty(0, dtype=np.float64)

last_video = np.zeros(n_users, dtype=np.int64)
if len(pc_users):
    reverse_users = pc_users[::-1]
    _, reverse_first = np.unique(reverse_users, return_index=True)
    last_positions = len(pc_users) - 1 - reverse_first
    last_video[pc_users[last_positions]] = pc_videos[last_positions]


def transition_predict(users, videos):
    source = last_video[users]
    query = (
        source.astype(np.int64) * np.int64(n_videos) +
        videos.astype(np.int64)
    )
    result = np.zeros(len(query), dtype=np.float64)
    if len(unique_keys):
        locations = np.searchsorted(unique_keys, query)
        valid_location = locations < len(unique_keys)
        matched = np.zeros(len(query), dtype=bool)
        matched[valid_location] = (
            unique_keys[locations[valid_location]] ==
            query[valid_location]
        )
        result[matched] = np.log1p(key_weights[locations[matched]])
    return result


transition_valid = transition_predict(valid_user, valid_video)
transition_test = transition_predict(test_user, test_video)


shared_dir = os.environ.get("SHARED_ARTIFACTS")
if not shared_dir:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared_dir, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared_dir, "incumbent_test_scores.npy")
).astype(np.float64)

raw_valid = {
    "positional_lightgbm": tree_valid,
    "within_user_bpr": mf_valid,
    "positive_transition": transition_valid,
}
raw_test = {
    "positional_lightgbm": tree_test,
    "within_user_bpr": mf_test,
    "positive_transition": transition_test,
}

rank_valid = {
    name: within_user_rank(valid.user_id, score)
    for name, score in raw_valid.items()
}
rank_test = {
    name: within_user_rank(test.user_id, score)
    for name, score in raw_test.items()
}
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

candidate_valid = {"trusted_incumbent": inc_valid}
candidate_test = {"trusted_incumbent": inc_test}
candidate_raw = {"trusted_incumbent": tree_valid}
candidate_uses_incumbent = {"trusted_incumbent": True}
candidate_scores = {}

for name in raw_valid:
    candidate_valid[name] = raw_valid[name]
    candidate_test[name] = raw_test[name]
    candidate_raw[name] = raw_valid[name]
    candidate_uses_incumbent[name] = False

    for alpha in [0.15, 0.25, 0.40, 0.55]:
        candidate_name = name + "_inc_blend_" + str(alpha)
        candidate_valid[candidate_name] = (
            (1.0 - alpha) * inc_valid_rank +
            alpha * rank_valid[name]
        )
        candidate_test[candidate_name] = (
            (1.0 - alpha) * inc_test_rank +
            alpha * rank_test[name]
        )
        candidate_raw[candidate_name] = raw_valid[name]
        candidate_uses_incumbent[candidate_name] = True

# A structurally diverse ensemble before adding the incumbent.
new_ensemble_valid = (
    0.55 * rank_valid["positional_lightgbm"] +
    0.30 * rank_valid["within_user_bpr"] +
    0.15 * rank_valid["positive_transition"]
)
new_ensemble_test = (
    0.55 * rank_test["positional_lightgbm"] +
    0.30 * rank_test["within_user_bpr"] +
    0.15 * rank_test["positive_transition"]
)
candidate_valid["new_family_ensemble"] = new_ensemble_valid
candidate_test["new_family_ensemble"] = new_ensemble_test
candidate_raw["new_family_ensemble"] = new_ensemble_valid
candidate_uses_incumbent["new_family_ensemble"] = False

for alpha in [0.15, 0.25, 0.40, 0.55]:
    name = "new_ensemble_inc_blend_" + str(alpha)
    candidate_valid[name] = (
        (1.0 - alpha) * inc_valid_rank +
        alpha * new_ensemble_valid
    )
    candidate_test[name] = (
        (1.0 - alpha) * inc_test_rank +
        alpha * new_ensemble_test
    )
    candidate_raw[name] = new_ensemble_valid
    candidate_uses_incumbent[name] = True

best_name = None
best_metrics = None
for name, scores in candidate_valid.items():
    metrics = evaluate(valid.user_id, y_valid, scores)
    candidate_scores[name] = float(metrics["primary"])
    if (
        best_metrics is None or
        metrics["primary"] > best_metrics["primary"]
    ):
        best_name = name
        best_metrics = metrics

chosen_valid = candidate_valid[best_name]
chosen_test = candidate_test[best_name]

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(chosen_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(chosen_test, dtype=np.float64),
    )
    if candidate_uses_incumbent[best_name]:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[best_name], dtype=np.float64),
        )

print("FINDINGS winner=" + best_name)
print(
    "CANDIDATES " +
    json.dumps(
        {k: round(v, 6) for k, v in candidate_scores.items()},
        sort_keys=True,
    )
)
elapsed = time.time() - START
print(
    "METRICS " +
    json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)