import os
import gc
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 29417
EPOCHS = 2
N_PARTITIONS = 64
PRED_PARTITIONS = 64
HALF_LIFE = 8.0

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tag",
    "tab", "duration_bucket", "upload_type", "hour",
]
HISTORY_FIELDS = [
    ("video_id", 18.0),
    ("author_id", 28.0),
    ("tag", 45.0),
    ("duration_bucket", 60.0),
    ("upload_type", 60.0),
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_all(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def get_field(obj, field):
    if field == "user_id":
        return np.asarray(obj.user_id, dtype=np.int64)
    return np.asarray(obj.X[field], dtype=np.int64)


def make_cat(split):
    n = len(split.user_id)
    out = np.empty((n, len(CAT_FIELDS)), dtype=np.int64)
    for j, field in enumerate(CAT_FIELDS):
        out[:, j] = get_field(split, field) + offsets[j]
    return out


def concat_cat(a, b):
    n1 = len(a.user_id)
    n2 = len(b.user_id)
    out = np.empty((n1 + n2, len(CAT_FIELDS)), dtype=np.int64)
    for j, field in enumerate(CAT_FIELDS):
        out[:n1, j] = get_field(a, field) + offsets[j]
        out[n1:, j] = get_field(b, field) + offsets[j]
    return out


def date_weights(dates, half_life=HALF_LIFE):
    day = np.asarray(dates, dtype=np.int64) % 100
    age = int(day.max()) - day
    w = np.exp2(-age.astype(np.float64) / half_life)
    return w


def safe_logit(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def numeric_matrix(split):
    cols = []
    for field in NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(x, 0.0)))
    return np.column_stack(cols).astype(np.float32)


def combined_numeric(a, b):
    return np.concatenate([numeric_matrix(a), numeric_matrix(b)], axis=0)


def build_dense_features(source, query, source_is_query=False):
    """
    Entity histories are recency weighted. For fitting rows, the current row's
    weighted label and exposure are removed. Query rows receive source-only
    histories, so validation/test labels never enter their own features.
    """
    y = np.asarray(source.y, dtype=np.float64)
    weights = date_weights(source.date)
    global_rate = float(np.sum(weights * y) / np.sum(weights))

    history_columns = []
    for field, smoothing in HISTORY_FIELDS:
        source_ids = get_field(source, field)
        query_ids = get_field(query, field)
        size = max(
            int(FEATURE_CARDINALITIES[field]),
            int(source_ids.max(initial=0)) + 1,
            int(query_ids.max(initial=0)) + 1,
        )
        den = np.bincount(source_ids, weights=weights, minlength=size)
        num = np.bincount(source_ids, weights=weights * y, minlength=size)

        if source_is_query:
            row_den = np.maximum(den[source_ids] - weights, 0.0)
            row_num = np.maximum(num[source_ids] - weights * y, 0.0)
            rate = (row_num + smoothing * global_rate) / (
                row_den + smoothing
            )
            count = np.log1p(row_den)
        else:
            q_den = den[query_ids]
            q_num = num[query_ids]
            rate = (q_num + smoothing * global_rate) / (
                q_den + smoothing
            )
            count = np.log1p(q_den)

        history_columns.append(count.astype(np.float32))
        history_columns.append(
            (safe_logit(rate) - safe_logit(global_rate)).astype(np.float32)
        )

    query_num = numeric_matrix(query)
    source_num = numeric_matrix(source)
    mean = source_num.mean(axis=0, dtype=np.float64)
    std = source_num.std(axis=0, dtype=np.float64)
    std = np.maximum(std, 0.15)
    query_num = ((query_num - mean) / std).astype(np.float32)

    return np.column_stack(history_columns + [
        query_num[:, j] for j in range(query_num.shape[1])
    ]).astype(np.float32)


def build_combined_dense(a, b):
    class Combined:
        pass

    combined = Combined()
    combined.user_id = np.concatenate([
        np.asarray(a.user_id, dtype=np.int64),
        np.asarray(b.user_id, dtype=np.int64),
    ])
    combined.date = np.concatenate([
        np.asarray(a.date, dtype=np.int64),
        np.asarray(b.date, dtype=np.int64),
    ])
    combined.y = np.concatenate([
        np.asarray(a.y, dtype=np.int8),
        np.asarray(b.y, dtype=np.int8),
    ])
    combined.X = {}
    for field in set(CAT_FIELDS + [x[0] for x in HISTORY_FIELDS]):
        if field == "user_id":
            continue
        combined.X[field] = np.concatenate([
            np.asarray(a.X[field], dtype=np.int64),
            np.asarray(b.X[field], dtype=np.int64),
        ])
    combined.num = {}
    for field in NUM_FIELDS:
        combined.num[field] = np.concatenate([
            np.asarray(a.num[field], dtype=np.float32),
            np.asarray(b.num[field], dtype=np.float32),
        ])
    return combined


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(starts_flag, np.arange(n, dtype=np.int64), 0)
    )
    ranks = np.arange(n, dtype=np.int64) - starts

    _, counts = np.unique(sorted_users, return_counts=True)
    sizes = np.repeat(counts, counts)
    rank_values = ranks.astype(np.float64) / np.maximum(sizes - 1, 1)
    rank_values[sizes == 1] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = rank_values
    return result


def make_partitions(users, n_partitions=N_PARTITIONS):
    users = np.asarray(users, dtype=np.int64)
    bucket = np.mod(users, n_partitions)
    order = np.argsort(bucket, kind="stable")
    counts = np.bincount(bucket, minlength=n_partitions)
    starts = np.cumsum(np.r_[0, counts[:-1]], dtype=np.int64)
    return [
        order[starts[k]:starts[k] + counts[k]]
        for k in range(n_partitions) if counts[k] > 0
    ]


def group_inverse(users_tensor):
    _, inverse = torch.unique(users_tensor, sorted=True, return_inverse=True)
    return inverse


def segment_logsumexp(scores, group, n_groups):
    maxima = torch.full(
        (n_groups,), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    maxima.scatter_reduce_(
        0, group, scores, reduce="amax", include_self=True
    )
    exp_sum = torch.zeros(
        n_groups, dtype=scores.dtype, device=scores.device
    )
    exp_sum.scatter_add_(0, group, torch.exp(scores - maxima[group]))
    return maxima + torch.log(exp_sum.clamp_min(1e-12))


class AdditiveListwise(nn.Module):
    def __init__(self, dense_dim):
        super().__init__()
        self.bias = nn.Embedding(total_cardinality, 1)
        self.dense = nn.Linear(dense_dim, 1)
        nn.init.zeros_(self.bias.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, xcat, dense, group):
        return (
            self.bias(xcat).sum(dim=1).squeeze(-1)
            + self.dense(dense).squeeze(-1)
        )


class LatentListwise(nn.Module):
    def __init__(self, dense_dim, dim=16):
        super().__init__()
        self.bias = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, dim)
        self.dense = nn.Sequential(
            nn.Linear(dense_dim, 24),
            nn.Tanh(),
            nn.Linear(24, 1),
        )
        nn.init.zeros_(self.bias.weight)
        nn.init.normal_(self.embedding.weight, std=0.025)

    def forward(self, xcat, dense, group):
        emb = self.embedding(xcat)
        user = emb[:, 0]
        candidate = emb[:, 1:].mean(dim=1)
        interaction = (user * candidate).sum(dim=1)
        return (
            self.bias(xcat).sum(dim=1).squeeze(-1)
            + interaction
            + self.dense(dense).squeeze(-1)
        )


class DeepSetListwise(nn.Module):
    def __init__(self, dense_dim, dim=6):
        super().__init__()
        self.bias = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, dim)
        row_dim = len(CAT_FIELDS) * dim + dense_dim
        self.row_tower = nn.Sequential(
            nn.Linear(row_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.ReLU(),
        )
        self.base_head = nn.Linear(24, 1)
        self.context_head = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        nn.init.zeros_(self.bias.weight)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, xcat, dense, group):
        z = torch.cat(
            [self.embedding(xcat).reshape(xcat.shape[0], -1), dense], dim=1
        )
        h = self.row_tower(z)
        n_groups = int(group.max().item()) + 1

        context_sum = torch.zeros(
            (n_groups, h.shape[1]), dtype=h.dtype, device=h.device
        )
        context_sum.index_add_(0, group, h)
        counts = torch.bincount(
            group, minlength=n_groups
        ).to(h.dtype).clamp_min(1.0)
        context = context_sum / counts[:, None]

        relative = torch.cat([h, h - context[group]], dim=1)
        return (
            self.bias(xcat).sum(dim=1).squeeze(-1)
            + self.base_head(h).squeeze(-1)
            + self.context_head(relative).squeeze(-1)
        )


def make_model(name, dense_dim):
    if name == "additive_listnet":
        return AdditiveListwise(dense_dim)
    if name == "latent_listnet":
        return LatentListwise(dense_dim)
    if name == "deepset_listnet":
        return DeepSetListwise(dense_dim)
    raise ValueError(name)


def learning_rate(name):
    if name == "additive_listnet":
        return 0.018
    if name == "latent_listnet":
        return 0.006
    return 0.003


def train_model(name, xcat, dense, users, labels, dates, epochs=EPOCHS):
    seed_all(SEED + {
        "additive_listnet": 11,
        "latent_listnet": 37,
        "deepset_listnet": 71,
    }[name])

    model = make_model(name, dense.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate(name),
        weight_decay=2e-6,
    )

    xt = torch.from_numpy(xcat)
    dt = torch.from_numpy(dense)
    ut = torch.from_numpy(np.asarray(users, dtype=np.int64))
    yt = torch.from_numpy(np.asarray(labels, dtype=np.float32))

    w_np = date_weights(dates).astype(np.float32)
    w_np /= max(float(w_np.mean()), 1e-6)
    wt = torch.from_numpy(w_np)

    partitions = make_partitions(users)
    rng = np.random.default_rng(SEED + 991)

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(partitions))
        for part_id in order:
            idx_np = partitions[int(part_id)]
            idx = torch.from_numpy(idx_np)

            bx = xt[idx]
            bd = dt[idx]
            bu = ut[idx]
            by = yt[idx]
            bw = wt[idx]
            group = group_inverse(bu)
            n_groups = int(group.max().item()) + 1

            scores = model(bx, bd, group)
            lse = segment_logsumexp(scores, group, n_groups)

            positive_weight = bw * by
            pos_mass = torch.zeros(n_groups, dtype=scores.dtype)
            pos_score = torch.zeros(n_groups, dtype=scores.dtype)
            pos_mass.scatter_add_(0, group, positive_weight)
            pos_score.scatter_add_(0, group, positive_weight * scores)

            eligible = pos_mass > 0
            if not bool(eligible.any()):
                continue

            list_loss = (
                lse[eligible]
                - pos_score[eligible] / pos_mass[eligible].clamp_min(1e-8)
            )
            group_weight = torch.sqrt(pos_mass[eligible].clamp_min(1.0))
            list_loss = (list_loss * group_weight).sum() / group_weight.sum()

            point_loss = nn.functional.binary_cross_entropy_with_logits(
                scores, by, weight=bw
            )
            loss = list_loss + 0.08 * point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_model(model, xcat, dense, users):
    model.eval()
    xt = torch.from_numpy(xcat)
    dt = torch.from_numpy(dense)
    ut = torch.from_numpy(np.asarray(users, dtype=np.int64))
    result = np.empty(len(users), dtype=np.float32)

    partitions = make_partitions(users, PRED_PARTITIONS)
    for idx_np in partitions:
        idx = torch.from_numpy(idx_np)
        group = group_inverse(ut[idx])
        result[idx_np] = model(
            xt[idx], dt[idx], group
        ).cpu().numpy()
    return result


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64, copy=False)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

x_train = make_cat(train)
x_valid = make_cat(valid)
dense_train = build_dense_features(train, train, source_is_query=True)
dense_valid = build_dense_features(train, valid, source_is_query=False)

family_names = [
    "additive_listnet",
    "latent_listnet",
    "deepset_listnet",
]

predictions = {}
candidate_scores = {}
best_score = float(evaluate(valid.user_id, y_valid, inc_valid)["primary"])
best_selection = ("incumbent", None, 0.0)
best_valid_scores = inc_valid.copy()
candidate_scores["incumbent"] = best_score

for family in family_names:
    model = train_model(
        family, x_train, dense_train,
        train.user_id, y_train, train.date, EPOCHS
    )
    pred = predict_model(model, x_valid, dense_valid, valid.user_id)
    predictions[family] = pred

    standalone = float(
        evaluate(valid.user_id, y_valid, pred)["primary"]
    )
    candidate_scores[family] = standalone
    if standalone > best_score:
        best_score = standalone
        best_selection = (family, "standalone", 0.0)
        best_valid_scores = pred.astype(np.float64)

    pred_rank = within_user_rank(valid.user_id, pred)
    for alpha in (0.25, 0.50, 0.75):
        blended = alpha * pred_rank + (1.0 - alpha) * inc_valid_rank
        name = family + "_blend_" + str(alpha)
        score = float(
            evaluate(valid.user_id, y_valid, blended)["primary"]
        )
        candidate_scores[name] = score
        if score > best_score:
            best_score = score
            best_selection = (family, "blend", alpha)
            best_valid_scores = blended.copy()

    del model
    gc.collect()

metrics = evaluate(valid.user_id, y_valid, best_valid_scores)

print("CANDIDATES " + json.dumps(
    {k: round(v, 6) for k, v in candidate_scores.items()},
    sort_keys=True
))
print("FINDINGS " + json.dumps({
    "selected": best_selection,
    "dense_dimension": int(dense_train.shape[1]),
    "history_half_life_days": HALF_LIFE,
    "epochs": EPOCHS,
}))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64)
    )

# Produce test scores. If the incumbent remains best, reuse its trusted test
# predictions. Otherwise refit the selected recipe on train+validation.
if best_selection[0] == "incumbent":
    test_scores = np.load(inc_test_path).astype(np.float64, copy=False)
else:
    selected_family, mode, alpha = best_selection
    test = load("test")

    combined = build_combined_dense(train, valid)
    x_combined = concat_cat(train, valid)
    dense_combined = build_dense_features(
        combined, combined, source_is_query=True
    )
    x_test = make_cat(test)
    dense_test = build_dense_features(
        combined, test, source_is_query=False
    )

    combined_labels = np.asarray(combined.y, dtype=np.int8)
    model = train_model(
        selected_family,
        x_combined,
        dense_combined,
        combined.user_id,
        combined_labels,
        combined.date,
        EPOCHS,
    )
    new_test_scores = predict_model(
        model, x_test, dense_test, test.user_id
    ).astype(np.float64)

    if mode == "blend":
        incumbent_test = np.load(inc_test_path).astype(
            np.float64, copy=False
        )
        test_scores = (
            alpha * within_user_rank(test.user_id, new_test_scores)
            + (1.0 - alpha)
            * within_user_rank(test.user_id, incumbent_test)
        )
    else:
        test_scores = new_test_scores

    del model, x_combined, dense_combined, x_test, dense_test, combined
    gc.collect()

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))