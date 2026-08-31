import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 19427
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "hour", "tag", "upload_type", "music_type", "user_active_degree",
    "is_live_streamer", "is_video_author", "follow_user_num_range",
    "fans_user_num_range", "friend_user_num_range", "register_days_range",
    "onehot_feat3", "onehot_feat7", "onehot_feat8", "video_type",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
EMBED_DIM = 8
TRAIN_BATCH = 4096
PRED_BATCH = 16384
EPOCHS = 2
HALF_LIFE = 4.0


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    user_start = np.empty(n, dtype=bool)
    user_start[0] = True
    user_start[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(user_start, np.arange(n, dtype=np.int64), 0)
    )

    user_end = np.empty(n, dtype=bool)
    user_end[-1] = True
    user_end[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_indices = np.flatnonzero(user_end)
    sizes = np.diff(np.concatenate((np.array([-1]), end_indices)))
    row_sizes = np.repeat(sizes, sizes)

    positions = np.arange(n, dtype=np.int64) - starts
    ranked = (positions.astype(np.float64) + 0.5) / row_sizes
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def recency_weights(dates, half_life):
    d = np.asarray(dates, dtype=np.int32)
    age = d.max() - d
    w = np.power(0.5, age.astype(np.float32) / half_life).astype(np.float32)
    return w / max(float(w.mean()), 1e-8)


def categorical_matrix(split):
    cards = np.asarray(
        [FEATURE_CARDINALITIES[f] for f in FIELDS], dtype=np.int64
    )
    offsets = np.cumsum(
        np.concatenate((np.array([0], dtype=np.int64), cards[:-1]))
    )
    columns = [
        np.asarray(split.X[f], dtype=np.int64) + offsets[j]
        for j, f in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.int64)


def raw_dense_matrix(split, video_hist, author_hist):
    columns = []
    for field in NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    for hist in (video_hist, author_hist):
        for key in sorted(hist.keys()):
            x = np.asarray(hist[key], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if "count" in key:
                x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
            columns.append(x)

    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float32)


def standardize_dense(train_x, valid_x, test_x):
    lo = np.quantile(train_x, 0.001, axis=0).astype(np.float32)
    hi = np.quantile(train_x, 0.999, axis=0).astype(np.float32)

    train_clip = np.clip(train_x, lo, hi)
    mean = train_clip.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_clip.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-3)

    def transform(x):
        x = np.clip(x, lo, hi)
        return np.ascontiguousarray((x - mean) / std, dtype=np.float32)

    return transform(train_x), transform(valid_x), transform(test_x)


class XDeepFM(nn.Module):
    def __init__(self, total_features, n_fields, dense_dim, initial_bias):
        super().__init__()
        self.n_fields = n_fields
        self.embedding = nn.Embedding(total_features, EMBED_DIM)
        self.linear = nn.Embedding(total_features, 1)

        self.cin_sizes = [12, 12]
        previous = n_fields
        self.cin_layers = nn.ModuleList()
        for size in self.cin_sizes:
            self.cin_layers.append(
                nn.Conv1d(previous * n_fields, size, kernel_size=1)
            )
            previous = size

        deep_input = n_fields * EMBED_DIM + dense_dim
        self.deep = nn.Sequential(
            nn.Linear(deep_input, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.cin_output = nn.Linear(sum(self.cin_sizes), 1, bias=False)
        self.dense_linear = nn.Linear(dense_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32))

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, categorical, dense):
        x0 = self.embedding(categorical)
        xk = x0
        cin_outputs = []

        for layer in self.cin_layers:
            interaction = torch.einsum("bhd,bmd->bhmd", xk, x0)
            interaction = interaction.reshape(
                interaction.shape[0], -1, EMBED_DIM
            )
            xk = torch.relu(layer(interaction))
            cin_outputs.append(xk.sum(dim=2))

        cin = self.cin_output(torch.cat(cin_outputs, dim=1)).squeeze(1)
        deep_input = torch.cat((x0.flatten(1), dense), dim=1)
        deep = self.deep(deep_input).squeeze(1)
        wide = self.linear(categorical).sum(dim=1).squeeze(1)
        dense_wide = self.dense_linear(dense).squeeze(1)
        return self.bias + wide + dense_wide + deep + cin


class FiBiNET(nn.Module):
    def __init__(self, total_features, n_fields, dense_dim, initial_bias):
        super().__init__()
        self.n_fields = n_fields
        self.embedding = nn.Embedding(total_features, EMBED_DIM)
        self.linear = nn.Embedding(total_features, 1)

        reduction = max(4, n_fields // 3)
        self.senet = nn.Sequential(
            nn.Linear(n_fields, reduction),
            nn.ReLU(),
            nn.Linear(reduction, n_fields),
            nn.Sigmoid(),
        )
        self.bilinear = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)

        pair_i, pair_j = np.triu_indices(n_fields, k=1)
        self.register_buffer(
            "pair_i", torch.from_numpy(pair_i.astype(np.int64))
        )
        self.register_buffer(
            "pair_j", torch.from_numpy(pair_j.astype(np.int64))
        )
        pair_dim = len(pair_i) * EMBED_DIM
        head_input = n_fields * EMBED_DIM + pair_dim + dense_dim

        self.head = nn.Sequential(
            nn.Linear(head_input, 192),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.dense_linear = nn.Linear(dense_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32))

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, categorical, dense):
        emb = self.embedding(categorical)
        squeeze = emb.mean(dim=2)
        field_weights = self.senet(squeeze).unsqueeze(2)
        recalibrated = emb * field_weights

        left = self.bilinear(recalibrated[:, self.pair_i, :])
        right = recalibrated[:, self.pair_j, :]
        interactions = (left * right).flatten(1)

        head_input = torch.cat(
            (recalibrated.flatten(1), interactions, dense), dim=1
        )
        deep = self.head(head_input).squeeze(1)
        wide = self.linear(categorical).sum(dim=1).squeeze(1)
        dense_wide = self.dense_linear(dense).squeeze(1)
        return self.bias + wide + dense_wide + deep


def train_neural(model, categorical, dense, labels, weights, seed_offset):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-6
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    n = len(labels)
    generator = torch.Generator().manual_seed(SEED + seed_offset)

    cat_tensor = torch.from_numpy(categorical)
    dense_tensor = torch.from_numpy(dense)
    label_tensor = torch.from_numpy(labels)
    weight_tensor = torch.from_numpy(weights)

    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, TRAIN_BATCH):
            idx = permutation[start:start + TRAIN_BATCH]
            logits = model(cat_tensor[idx], dense_tensor[idx])
            loss = (
                criterion(logits, label_tensor[idx]) * weight_tensor[idx]
            ).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict_neural(model, categorical, dense):
    result = np.empty(len(categorical), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(categorical), PRED_BATCH):
            end = min(start + PRED_BATCH, len(categorical))
            cat = torch.from_numpy(categorical[start:end])
            den = torch.from_numpy(dense[start:end])
            result[start:end] = model(cat, den).cpu().numpy()
    return result


def signed_svd_predictions(train, valid, test, rank=32):
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.float32)

    n_users = FEATURE_CARDINALITIES["user_id"]
    n_videos = FEATURE_CARDINALITIES["video_id"]
    global_rate = float(labels.mean())

    values = labels - global_rate
    matrix = sp.coo_matrix(
        (values, (users, videos)),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    row_nnz = np.diff(matrix.indptr).astype(np.float32)
    row_scale = 1.0 / np.sqrt(np.maximum(row_nnz, 1.0))
    matrix = sp.diags(row_scale).dot(matrix).tocsr()

    try:
        u, singular, vt = svds(
            matrix,
            k=rank,
            which="LM",
            return_singular_vectors=True,
            random_state=SEED,
        )
    except TypeError:
        u, singular, vt = svds(
            matrix,
            k=rank,
            which="LM",
            return_singular_vectors=True,
        )

    order = np.argsort(singular)[::-1]
    singular = singular[order]
    u = u[:, order]
    vt = vt[order, :]

    user_factors = (
        u * singular.reshape(1, -1)
    ) / np.maximum(row_scale.reshape(-1, 1), 1e-8)
    video_factors = vt.T

    def score(split):
        su = np.asarray(split.X["user_id"], dtype=np.int64)
        sv = np.asarray(split.X["video_id"], dtype=np.int64)
        return np.einsum(
            "ij,ij->i",
            user_factors[su],
            video_factors[sv],
            optimize=True,
        ).astype(np.float32)

    return score(valid), score(test), singular


train = load("train")
valid = load("valid")
test = load("test")

labels = np.asarray(train.y, dtype=np.float32)
positive_rate = float(labels.mean())
initial_bias = float(np.log(positive_rate / (1.0 - positive_rate)))
weights = recency_weights(train.date, HALF_LIFE)

# All feature statistics below are derived solely from train. Historical
# features for valid/test are supplied train-only histories by the pipeline.
vh_train = historical_features("train", key="video_id")
ah_train = historical_features("train", key="author_id")
vh_valid = historical_features("valid", key="video_id")
ah_valid = historical_features("valid", key="author_id")
vh_test = historical_features("test", key="video_id")
ah_test = historical_features("test", key="author_id")

cat_train = categorical_matrix(train)
cat_valid = categorical_matrix(valid)
cat_test = categorical_matrix(test)

dense_train_raw = raw_dense_matrix(train, vh_train, ah_train)
dense_valid_raw = raw_dense_matrix(valid, vh_valid, ah_valid)
dense_test_raw = raw_dense_matrix(test, vh_test, ah_test)
dense_train, dense_valid, dense_test = standardize_dense(
    dense_train_raw, dense_valid_raw, dense_test_raw
)
del dense_train_raw, dense_valid_raw, dense_test_raw
gc.collect()

total_features = int(sum(FEATURE_CARDINALITIES[f] for f in FIELDS))
dense_dim = dense_train.shape[1]

# Family 1: explicit bounded-degree high-order feature interactions.
xdeep = XDeepFM(
    total_features, len(FIELDS), dense_dim, initial_bias
)
xdeep = train_neural(
    xdeep, cat_train, dense_train, labels, weights, seed_offset=101
)
xdeep_valid = predict_neural(xdeep, cat_valid, dense_valid)
xdeep_test = predict_neural(xdeep, cat_test, dense_test)
del xdeep
gc.collect()

# Family 2: field recalibration followed by bilinear interaction formation.
torch.manual_seed(SEED + 1)
fibinet = FiBiNET(
    total_features, len(FIELDS), dense_dim, initial_bias
)
fibinet = train_neural(
    fibinet, cat_train, dense_train, labels, weights, seed_offset=307
)
fibinet_valid = predict_neural(fibinet, cat_valid, dense_valid)
fibinet_test = predict_neural(fibinet, cat_test, dense_test)
del fibinet
gc.collect()

# Family 3: non-neural collaborative latent reconstruction.
svd_valid, svd_test, svd_singular = signed_svd_predictions(
    train, valid, test, rank=32
)

own_valid = {
    "xdeepfm": np.asarray(xdeep_valid, dtype=np.float64),
    "fibinet": np.asarray(fibinet_valid, dtype=np.float64),
    "signed_svd": np.asarray(svd_valid, dtype=np.float64),
}
own_test = {
    "xdeepfm": np.asarray(xdeep_test, dtype=np.float64),
    "fibinet": np.asarray(fibinet_test, dtype=np.float64),
    "signed_svd": np.asarray(svd_test, dtype=np.float64),
}

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

candidate_scores = {"incumbent_reference": inc_valid}
candidate_metrics = {
    "incumbent_reference": evaluate(valid.user_id, valid.y, inc_valid)
}
candidate_specs = {
    "incumbent_reference": ("incumbent", 0.0)
}

for family in ("xdeepfm", "fibinet", "signed_svd"):
    standalone = family + "_standalone"
    candidate_scores[standalone] = own_valid[family]
    candidate_metrics[standalone] = evaluate(
        valid.user_id, valid.y, own_valid[family]
    )
    candidate_specs[standalone] = (family, None)

    valid_rank = rank_percentile(valid.user_id, own_valid[family])
    for alpha in (0.15, 0.30, 0.50, 0.70):
        name = f"{family}_blend_{alpha:.2f}"
        blended = alpha * valid_rank + (1.0 - alpha) * inc_valid_rank
        candidate_scores[name] = blended
        candidate_metrics[name] = evaluate(
            valid.user_id, valid.y, blended
        )
        candidate_specs[name] = (family, alpha)

# Add a Borda-style consensus of the two new supervised interaction models.
xdeep_rank_valid = rank_percentile(valid.user_id, own_valid["xdeepfm"])
fibinet_rank_valid = rank_percentile(valid.user_id, own_valid["fibinet"])
xdeep_rank_test = rank_percentile(test.user_id, own_test["xdeepfm"])
fibinet_rank_test = rank_percentile(test.user_id, own_test["fibinet"])

supervised_consensus_valid = 0.5 * (
    xdeep_rank_valid + fibinet_rank_valid
)
supervised_consensus_test = 0.5 * (
    xdeep_rank_test + fibinet_rank_test
)
candidate_scores["supervised_consensus_standalone"] = (
    supervised_consensus_valid
)
candidate_metrics["supervised_consensus_standalone"] = evaluate(
    valid.user_id, valid.y, supervised_consensus_valid
)
candidate_specs["supervised_consensus_standalone"] = (
    "supervised_consensus", None
)

for alpha in (0.20, 0.40, 0.60):
    name = f"supervised_consensus_blend_{alpha:.2f}"
    blended = (
        alpha * supervised_consensus_valid
        + (1.0 - alpha) * inc_valid_rank
    )
    candidate_scores[name] = blended
    candidate_metrics[name] = evaluate(
        valid.user_id, valid.y, blended
    )
    candidate_specs[name] = ("supervised_consensus", alpha)

best_key = max(
    candidate_metrics,
    key=lambda key: float(candidate_metrics[key]["primary"])
)
best_metrics = candidate_metrics[best_key]
best_valid = candidate_scores[best_key]
best_family, best_alpha = candidate_specs[best_key]

if best_family == "incumbent":
    best_test = inc_test
    raw_valid = None
elif best_family == "supervised_consensus":
    raw_valid = supervised_consensus_valid
    if best_alpha is None:
        best_test = supervised_consensus_test
    else:
        best_test = (
            best_alpha * supervised_consensus_test
            + (1.0 - best_alpha) * inc_test_rank
        )
else:
    raw_valid = own_valid[best_family]
    if best_alpha is None:
        best_test = own_test[best_family]
    else:
        own_test_rank = rank_percentile(
            test.user_id, own_test[best_family]
        )
        best_test = (
            best_alpha * own_test_rank
            + (1.0 - best_alpha) * inc_test_rank
        )

candidate_summary = {
    key: float(value["primary"])
    for key, value in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(candidate_summary, sort_keys=True))

rank_correlations = {}
family_ranks = {
    "xdeepfm": xdeep_rank_valid,
    "fibinet": fibinet_rank_valid,
    "signed_svd": rank_percentile(valid.user_id, own_valid["signed_svd"]),
    "incumbent": inc_valid_rank,
}
names = list(family_ranks.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a = family_ranks[names[i]]
        b = family_ranks[names[j]]
        corr = float(np.corrcoef(a, b)[0, 1])
        rank_correlations[names[i] + "__" + names[j]] = corr

print("FINDINGS " + json.dumps({
    "best_candidate": best_key,
    "recency_half_life_days": HALF_LIFE,
    "neural_epochs": EPOCHS,
    "dense_feature_count": int(dense_dim),
    "svd_rank": 32,
    "svd_leading_singular_value": float(svd_singular[0]),
    "within_user_rank_correlations": rank_correlations,
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if raw_valid is not None and best_alpha is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))