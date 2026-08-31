import os
import time
import json
import gc
import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 18437
THREADS = max(1, min(8, os.cpu_count() or 1))
DIM = 64

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

CAT_FIELDS = [
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "user_active_degree",
]


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int64)
    unique = np.unique(dates)
    idx = np.searchsorted(unique, dates)
    age = (len(unique) - 1 - idx).astype(np.float32)
    w = np.exp2(-age / float(half_life))
    return (w / np.mean(w)).astype(np.float32)


def within_user_ranks(user_ids, scores):
    users = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        users,
    ))
    su = users[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = su[1:] != su[:-1]
    starts = np.flatnonzero(first)
    group_id = np.cumsum(first) - 1
    local = np.arange(n, dtype=np.int64) - starts[group_id]
    sizes = np.diff(np.r_[starts, n])

    rank = (local.astype(np.float64) + 0.5) / sizes[group_id]
    result = np.empty(n, dtype=np.float64)
    result[order] = rank
    return result


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    return x / np.maximum(norm, 1e-6)


def graph_block(
    hist_users,
    hist_entities,
    hist_y,
    hist_dates,
    query_users,
    query_entities,
    n_users,
    n_entities,
    seed,
):
    hist_users = np.asarray(hist_users, dtype=np.int32)
    hist_entities = np.asarray(hist_entities, dtype=np.int32)
    hist_y = np.asarray(hist_y, dtype=np.float32)
    hist_dates = np.asarray(hist_dates)
    query_users = np.asarray(query_users, dtype=np.int32)
    query_entities = np.asarray(query_entities, dtype=np.int32)

    rw = recency_weights(hist_dates, half_life=4.0)
    base_rate = float(np.average(hist_y, weights=rw))

    positive_values = rw * hist_y
    signed_values = rw * (hist_y - base_rate)

    shape = (n_users, n_entities)
    positive = sp.csr_matrix(
        (positive_values, (hist_users, hist_entities)),
        shape=shape,
        dtype=np.float32,
    )
    positive.eliminate_zeros()

    signed = sp.csr_matrix(
        (signed_values, (hist_users, hist_entities)),
        shape=shape,
        dtype=np.float32,
    )
    signed.eliminate_zeros()

    rng = np.random.default_rng(seed)
    omega = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(n_users, DIM),
    )
    omega /= np.sqrt(float(DIM))

    # Random projections approximate similarities between entity columns
    # of the user-entity interaction graph without a dense Gram matrix.
    entity_positive = positive.T @ omega
    entity_signed = signed.T @ omega
    entity_positive = normalize_rows(entity_positive)
    entity_signed = normalize_rows(entity_signed)

    # A user's positive-history profile is diffused through those entity
    # signatures, yielding a two-hop user->entity->users->candidate signal.
    profile_positive = positive @ entity_positive
    profile_signed = positive @ entity_signed
    profile_positive = normalize_rows(profile_positive)
    profile_signed = normalize_rows(profile_signed)

    q_pos = np.sum(
        profile_positive[query_users] * entity_positive[query_entities],
        axis=1,
    ).astype(np.float32)
    q_signed = np.sum(
        profile_signed[query_users] * entity_signed[query_entities],
        axis=1,
    ).astype(np.float32)

    exposure_count = np.bincount(
        hist_entities,
        weights=rw,
        minlength=n_entities,
    ).astype(np.float32)
    positive_count = np.bincount(
        hist_entities,
        weights=rw * hist_y,
        minlength=n_entities,
    ).astype(np.float32)

    smooth = 20.0
    rate = (
        positive_count + smooth * base_rate
    ) / (exposure_count + smooth)

    q_count = np.log1p(exposure_count[query_entities]).astype(np.float32)
    q_rate = rate[query_entities].astype(np.float32)

    del positive, signed, omega
    del entity_positive, entity_signed
    del profile_positive, profile_signed
    gc.collect()

    return np.column_stack([
        q_pos,
        q_signed,
        q_count,
        q_rate,
    ]).astype(np.float32)


def graph_features(hist, query):
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    video_card = int(FEATURE_CARDINALITIES["video_id"])
    author_card = int(FEATURE_CARDINALITIES["author_id"])

    video = graph_block(
        hist.user_id,
        hist.video_id,
        hist.y,
        hist.date,
        query.user_id,
        query.video_id,
        n_users,
        video_card,
        SEED + 101,
    )
    author = graph_block(
        hist.user_id,
        hist.X["author_id"],
        hist.y,
        hist.date,
        query.user_id,
        query.X["author_id"],
        n_users,
        author_card,
        SEED + 211,
    )
    return np.concatenate([video, author], axis=1)


class ArraySplit:
    pass


def subset_split(split, mask):
    out = ArraySplit()
    out.user_id = np.asarray(split.user_id)[mask]
    out.video_id = np.asarray(split.video_id)[mask]
    out.date = np.asarray(split.date)[mask]
    out.y = np.asarray(split.y)[mask]
    out.X = {
        name: np.asarray(values)[mask]
        for name, values in split.X.items()
    }
    return out


def combined_split(a, b):
    out = ArraySplit()
    out.user_id = np.concatenate([
        np.asarray(a.user_id),
        np.asarray(b.user_id),
    ])
    out.video_id = np.concatenate([
        np.asarray(a.video_id),
        np.asarray(b.video_id),
    ])
    out.date = np.concatenate([
        np.asarray(a.date),
        np.asarray(b.date),
    ])
    out.y = np.concatenate([
        np.asarray(a.y),
        np.asarray(b.y),
    ])
    out.X = {}
    for name in a.X:
        out.X[name] = np.concatenate([
            np.asarray(a.X[name]),
            np.asarray(b.X[name]),
        ])
    return out


def model_matrix(split, graph):
    cats = np.column_stack([
        np.asarray(split.X[name], dtype=np.float32)
        for name in CAT_FIELDS
    ])
    return np.concatenate([graph, cats], axis=1).astype(np.float32)


def fit_gbdt(x, y, dates, rounds=150):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 120,
        "max_bin": 127,
        "feature_fraction": 0.92,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.03,
        "lambda_l2": 2.0,
        "min_gain_to_split": 0.002,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": THREADS,
        "force_col_wise": True,
        "verbose": -1,
    }
    dset = lgb.Dataset(
        x,
        label=np.asarray(y, dtype=np.float32),
        weight=recency_weights(dates, half_life=4.0),
        categorical_feature=list(
            range(8, 8 + len(CAT_FIELDS))
        ),
        free_raw_data=False,
    )
    return lgb.train(params, dset, num_boost_round=rounds)


class LinearGraph(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def fit_linear(x, y, dates):
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4)
    z = ((x - mean) / std).astype(np.float32)

    torch.manual_seed(SEED + 301)
    rng = np.random.default_rng(SEED + 302)
    model = LinearGraph(z.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.025, weight_decay=2e-4
    )

    tx = torch.from_numpy(z)
    ty = torch.from_numpy(np.asarray(y, dtype=np.float32))
    weights = recency_weights(dates, half_life=4.0)
    batch_size = 32768

    for _ in range(10):
        order = rng.permutation(len(y))
        model.train()
        for begin in range(0, len(order), batch_size):
            idx_np = order[begin:begin + batch_size]
            idx = torch.from_numpy(idx_np)
            logits = model(tx[idx])
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits, ty[idx], reduction="none"
                ) * torch.from_numpy(weights[idx_np])
            ).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model, mean, std


def predict_linear(model, mean, std, x):
    result = np.empty(len(x), dtype=np.float32)
    batch_size = 65536
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(x), batch_size):
            end = min(begin + batch_size, len(x))
            z = ((x[begin:end] - mean) / std).astype(np.float32)
            result[begin:end] = model(
                torch.from_numpy(z)
            ).cpu().numpy()
    return result


train = load("train")
valid = load("valid")
y_valid = np.asarray(valid.y, dtype=np.int8)

# Temporal stacking: graph representations are built on the first nine
# training days, and prediction heads are learned on the final four days.
unique_train_dates = np.unique(np.asarray(train.date))
cut_date = unique_train_dates[-4]
early_mask = np.asarray(train.date) < cut_date
tail_mask = ~early_mask

early = subset_split(train, early_mask)
tail = subset_split(train, tail_mask)

graph_tail = graph_features(early, tail)
graph_valid = graph_features(train, valid)

x_tail = model_matrix(tail, graph_tail)
x_valid = model_matrix(valid, graph_valid)

linear, linear_mean, linear_std = fit_linear(
    graph_tail,
    tail.y,
    tail.date,
)
linear_valid = predict_linear(
    linear, linear_mean, linear_std, graph_valid
)

gbdt = fit_gbdt(
    x_tail,
    tail.y,
    tail.date,
    rounds=150,
)
gbdt_valid = gbdt.predict(
    x_valid, num_iteration=gbdt.current_iteration()
).astype(np.float32)

raw_predictions = {
    "graph_diffusion_positive": (
        graph_valid[:, 0] + graph_valid[:, 4]
    ),
    "graph_diffusion_signed": (
        graph_valid[:, 1] + graph_valid[:, 5]
    ),
    "graph_linear_head": linear_valid,
    "graph_gbdt_head": gbdt_valid,
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_rank = within_user_ranks(valid.user_id, inc_valid)

candidates = {}
records = {}
alphas = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)

for family, raw in raw_predictions.items():
    raw = np.asarray(raw, dtype=np.float64)
    raw_metrics = evaluate(valid.user_id, y_valid, raw)
    candidates[family + "_raw"] = float(raw_metrics["primary"])
    raw_rank = within_user_ranks(valid.user_id, raw)

    for alpha in alphas:
        scores = (1.0 - alpha) * inc_rank + alpha * raw_rank
        metrics = evaluate(valid.user_id, y_valid, scores)
        name = family + "_blend_" + str(alpha)
        candidates[name] = float(metrics["primary"])
        records[name] = {
            "family": family,
            "alpha": float(alpha),
            "scores": scores,
            "raw": raw,
            "metrics": metrics,
        }

winner_name = max(
    records,
    key=lambda name: records[name]["metrics"]["primary"],
)
winner = records[winner_name]

print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": winner_name,
    "winner_family": winner["family"],
    "winner_alpha": winner["alpha"],
    "incumbent_primary": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    ),
    "tail_rows": int(len(tail.y)),
    "early_rows": int(len(early.y)),
    "video_positive_graph_std": float(np.std(graph_valid[:, 0])),
    "author_positive_graph_std": float(np.std(graph_valid[:, 4])),
    "video_signed_graph_std": float(np.std(graph_valid[:, 1])),
    "author_signed_graph_std": float(np.std(graph_valid[:, 5])),
}, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(winner["scores"], dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(winner["raw"], dtype=np.float64),
    )

# Test recipe: validation becomes the temporally later head-training window,
# while graph features for it are produced from train only. The test graph
# itself is built from all permitted train+validation outcomes.
test = load("test")
combined = combined_split(train, valid)
graph_test = graph_features(combined, test)

selected_family = winner["family"]

if selected_family == "graph_diffusion_positive":
    raw_test = graph_test[:, 0] + graph_test[:, 4]
elif selected_family == "graph_diffusion_signed":
    raw_test = graph_test[:, 1] + graph_test[:, 5]
elif selected_family == "graph_linear_head":
    final_linear, final_mean, final_std = fit_linear(
        graph_valid,
        y_valid,
        valid.date,
    )
    raw_test = predict_linear(
        final_linear,
        final_mean,
        final_std,
        graph_test,
    )
else:
    x_final_train = model_matrix(valid, graph_valid)
    x_test = model_matrix(test, graph_test)
    final_gbdt = fit_gbdt(
        x_final_train,
        y_valid,
        valid.date,
        rounds=150,
    )
    raw_test = final_gbdt.predict(
        x_test,
        num_iteration=final_gbdt.current_iteration(),
    ).astype(np.float32)

inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)
inc_test_rank = within_user_ranks(test.user_id, inc_test)
raw_test_rank = within_user_ranks(test.user_id, raw_test)

test_scores = (
    (1.0 - winner["alpha"]) * inc_test_rank
    + winner["alpha"] * raw_test_rank
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

metrics = winner["metrics"]
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(time.time() - START),
}))