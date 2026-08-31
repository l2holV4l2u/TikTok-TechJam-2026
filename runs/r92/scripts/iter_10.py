import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
THREADS = max(1, min(8, os.cpu_count() or 1))
HASH_SIZE = 1 << 20

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int64)
    age = dates.max() - dates
    w = np.exp2(-age.astype(np.float32) / float(half_life))
    return (w / w.mean()).astype(np.float32)


def within_user_ranks(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        users,
    ))
    sorted_users = users[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(first)
    group_id = np.cumsum(first) - 1
    local_rank = np.arange(n, dtype=np.int64) - starts[group_id]
    sizes = np.diff(np.r_[starts, n])

    ranked = (
        local_rank.astype(np.float64) + 0.5
    ) / sizes[group_id].astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def hashed_pair(a, b, salt):
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    z = (
        a * np.uint64(11400714819323198485)
        + b * np.uint64(14029467366897019727)
        + np.uint64(salt)
    )
    z ^= z >> np.uint64(29)
    return np.asarray(z & np.uint64(HASH_SIZE - 1), dtype=np.int64)


def wide_arrays(split):
    user = np.asarray(split.X["user_id"], dtype=np.int64)
    video = np.asarray(split.X["video_id"], dtype=np.int64)
    author = np.asarray(split.X["author_id"], dtype=np.int64)
    tag = np.asarray(split.X["tag"], dtype=np.int64)
    duration = np.asarray(split.X["duration_bucket"], dtype=np.int64)
    tab = np.asarray(split.X["tab"], dtype=np.int64)
    feat3 = np.asarray(split.X["onehot_feat3"], dtype=np.int64)

    return (
        video,
        author,
        tag,
        duration,
        tab,
        feat3,
        hashed_pair(user, video, 11),
        hashed_pair(user, author, 23),
        hashed_pair(user, tag, 37),
        hashed_pair(user, duration, 51),
        hashed_pair(user, tab, 71),
        hashed_pair(user, feat3, 89),
    )


class WideCrossModel(nn.Module):
    def __init__(self):
        super().__init__()

        cardinalities = [
            FEATURE_CARDINALITIES["video_id"],
            FEATURE_CARDINALITIES["author_id"],
            FEATURE_CARDINALITIES["tag"],
            FEATURE_CARDINALITIES["duration_bucket"],
            FEATURE_CARDINALITIES["tab"],
            FEATURE_CARDINALITIES["onehot_feat3"],
        ]

        self.main = nn.ModuleList([
            nn.Embedding(int(c), 1, sparse=True)
            for c in cardinalities
        ])
        self.cross = nn.ModuleList([
            nn.Embedding(HASH_SIZE, 1, sparse=True)
            for _ in range(6)
        ])
        self.intercept = nn.Parameter(torch.zeros(1))

        with torch.no_grad():
            for emb in self.main:
                emb.weight.zero_()
            for emb in self.cross:
                emb.weight.zero_()

    def forward(self, arrays):
        score = self.intercept.expand(arrays[0].shape[0])
        for emb, x in zip(self.main, arrays[:6]):
            score = score + emb(x).squeeze(1)
        for emb, x in zip(self.cross, arrays[6:]):
            score = score + emb(x).squeeze(1)
        return score


def fit_wide(arrays, labels, dates, epochs=3):
    torch.manual_seed(SEED + 1)
    rng = np.random.default_rng(SEED + 2)

    model = WideCrossModel()
    sparse_parameters = []
    for emb in list(model.main) + list(model.cross):
        sparse_parameters.extend(list(emb.parameters()))

    sparse_optimizer = torch.optim.SparseAdam(
        sparse_parameters, lr=0.045
    )
    dense_optimizer = torch.optim.Adam(
        [model.intercept], lr=0.025
    )

    tensors = tuple(torch.from_numpy(x) for x in arrays)
    labels_tensor = torch.from_numpy(
        np.asarray(labels, dtype=np.float32)
    )
    weights_tensor = torch.from_numpy(recency_weights(dates, 4.0))

    n = len(labels)
    batch_size = 16384

    model.train()
    for _ in range(epochs):
        permutation = rng.permutation(n).astype(np.int64, copy=False)
        for begin in range(0, n, batch_size):
            idx_np = permutation[begin:begin + batch_size]
            idx = torch.from_numpy(idx_np)

            logits = model(tuple(x[idx] for x in tensors))
            losses = F.binary_cross_entropy_with_logits(
                logits,
                labels_tensor[idx],
                reduction="none",
            )
            loss = (losses * weights_tensor[idx]).mean()

            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            sparse_optimizer.step()
            dense_optimizer.step()

    return model


def predict_wide(model, arrays, batch_size=32768):
    tensors = tuple(torch.from_numpy(x) for x in arrays)
    result = np.empty(len(arrays[0]), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for begin in range(0, len(result), batch_size):
            end = min(begin + batch_size, len(result))
            result[begin:end] = model(
                tuple(x[begin:end] for x in tensors)
            ).cpu().numpy()

    return result


def fit_residual_svd(split, labels, rank=24):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    y = np.asarray(labels, dtype=np.float32)
    w = recency_weights(split.date, 4.0)

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])

    user_weight = np.bincount(
        users, weights=w, minlength=n_users
    ).astype(np.float64)
    user_positive = np.bincount(
        users, weights=w * y, minlength=n_users
    ).astype(np.float64)

    global_mean = float(np.sum(w * y) / np.sum(w))
    user_mean = np.full(n_users, global_mean, dtype=np.float32)
    seen = user_weight > 0
    user_mean[seen] = (
        (user_positive[seen] + 8.0 * global_mean)
        / (user_weight[seen] + 8.0)
    ).astype(np.float32)

    residual = (y - user_mean[users]) * np.sqrt(w)

    matrix = sparse.coo_matrix(
        (residual, (users, videos)),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    row_degree = np.asarray(
        (matrix != 0).sum(axis=1)
    ).ravel().astype(np.float32)
    col_degree = np.asarray(
        (matrix != 0).sum(axis=0)
    ).ravel().astype(np.float32)

    row_scale = 1.0 / np.sqrt(np.maximum(row_degree, 1.0))
    col_scale = 1.0 / np.sqrt(np.maximum(col_degree, 1.0))

    normalized = sparse.diags(row_scale).dot(matrix).dot(
        sparse.diags(col_scale)
    ).tocsr()

    u, singular, vt = svds(
        normalized,
        k=rank,
        which="LM",
        random_state=SEED + 10,
        tol=2e-3,
        maxiter=500,
    )

    order = np.argsort(singular)[::-1]
    singular = singular[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)

    root_s = np.sqrt(np.maximum(singular, 0.0))
    user_factors = u * root_s[None, :]
    video_factors = vt.T * root_s[None, :]

    return user_factors, video_factors


def predict_svd(factors, split):
    user_factors, video_factors = factors
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    return np.einsum(
        "ij,ij->i",
        user_factors[users],
        video_factors[videos],
        optimize=True,
    ).astype(np.float32)


def concatenate_wide_arrays(a, b):
    return tuple(
        np.concatenate([x, y], axis=0)
        for x, y in zip(a, b)
    )


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

train_wide = wide_arrays(train)
valid_wide = wide_arrays(valid)

wide_model = fit_wide(
    train_wide, y_train, train.date, epochs=3
)
wide_valid = predict_wide(wide_model, valid_wide)
del wide_model
gc.collect()

svd_factors = fit_residual_svd(train, y_train, rank=24)
svd_valid = predict_svd(svd_factors, valid)
del svd_factors
gc.collect()

wide_rank = within_user_ranks(valid.user_id, wide_valid)
svd_rank = within_user_ranks(valid.user_id, svd_valid)
hybrid_rank = 0.55 * wide_rank + 0.45 * svd_rank

own_scores = {
    "wide_cross": wide_valid.astype(np.float64),
    "residual_svd": svd_valid.astype(np.float64),
    "wide_svd_rank_hybrid": hybrid_rank,
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_rank = within_user_ranks(valid.user_id, inc_valid)

candidate_log = {}
records = {}

for family, raw_score in own_scores.items():
    raw_metric = evaluate(valid.user_id, y_valid, raw_score)
    candidate_log[family + "_raw"] = float(raw_metric["primary"])

    own_rank = within_user_ranks(valid.user_id, raw_score)
    for alpha in (0.10, 0.20, 0.35, 0.50, 0.70):
        blended = (1.0 - alpha) * inc_rank + alpha * own_rank
        metric = evaluate(valid.user_id, y_valid, blended)
        name = family + "_inc_blend_" + str(alpha)
        candidate_log[name] = float(metric["primary"])
        records[name] = {
            "family": family,
            "alpha": alpha,
            "scores": blended,
            "raw": np.asarray(raw_score, dtype=np.float64),
            "metrics": metric,
        }

winner_name = max(
    records, key=lambda name: records[name]["metrics"]["primary"]
)
winner = records[winner_name]

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": winner_name,
    "winner_family": winner["family"],
    "winner_alpha": winner["alpha"],
    "wide_raw_primary": candidate_log["wide_cross_raw"],
    "svd_raw_primary": candidate_log["residual_svd_raw"],
    "hybrid_raw_primary": candidate_log["wide_svd_rank_hybrid_raw"],
    "incumbent_primary": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    ),
    "wide_svd_rank_correlation": float(
        np.corrcoef(wide_rank, svd_rank)[0, 1]
    ),
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

# Refit the selected identical recipe on train + validation.
# Test labels are never accessed.
test = load("test")
test_wide = wide_arrays(test)

combined_labels = np.concatenate([
    y_train,
    y_valid.astype(np.float32, copy=False),
])
combined_dates = np.concatenate([
    np.asarray(train.date),
    np.asarray(valid.date),
])
combined_wide = concatenate_wide_arrays(train_wide, valid_wide)

selected_family = winner["family"]

wide_test = None
svd_test = None

if selected_family in ("wide_cross", "wide_svd_rank_hybrid"):
    final_wide_model = fit_wide(
        combined_wide,
        combined_labels,
        combined_dates,
        epochs=3,
    )
    wide_test = predict_wide(final_wide_model, test_wide)
    del final_wide_model
    gc.collect()

if selected_family in ("residual_svd", "wide_svd_rank_hybrid"):
    class CombinedSplit:
        pass

    combined_split = CombinedSplit()
    combined_split.X = {
        "user_id": np.concatenate([
            np.asarray(train.X["user_id"], dtype=np.int64),
            np.asarray(valid.X["user_id"], dtype=np.int64),
        ]),
        "video_id": np.concatenate([
            np.asarray(train.X["video_id"], dtype=np.int64),
            np.asarray(valid.X["video_id"], dtype=np.int64),
        ]),
    }
    combined_split.date = combined_dates

    final_svd = fit_residual_svd(
        combined_split, combined_labels, rank=24
    )
    svd_test = predict_svd(final_svd, test)
    del final_svd
    gc.collect()

if selected_family == "wide_cross":
    own_test = wide_test.astype(np.float64)
elif selected_family == "residual_svd":
    own_test = svd_test.astype(np.float64)
else:
    own_test = (
        0.55 * within_user_ranks(test.user_id, wide_test)
        + 0.45 * within_user_ranks(test.user_id, svd_test)
    )

inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)
test_scores = (
    (1.0 - winner["alpha"])
    * within_user_ranks(test.user_id, inc_test)
    + winner["alpha"]
    * within_user_ranks(test.user_id, own_test)
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

metrics = winner["metrics"]
elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))