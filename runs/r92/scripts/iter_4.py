import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 731
DEVICE = "cpu"
MF_RANK = 20
MF_EPOCHS = 4
MF_BATCH = 8192
MF_LR = 0.012

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def safe_logit(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def entity_rate(ids, y, cardinality, prior, strength=30.0):
    counts = np.bincount(ids, minlength=cardinality).astype(np.float64)
    positives = np.bincount(
        ids, weights=y.astype(np.float64, copy=False),
        minlength=cardinality
    )
    return ((positives + strength * prior) /
            (counts + strength)).astype(np.float32)


class PairResidual:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    @staticmethod
    def fit(user, content, y, parent_rate, content_cardinality, strength):
        key = (user.astype(np.int64, copy=False) *
               np.int64(content_cardinality) +
               content.astype(np.int64, copy=False))
        order = np.argsort(key, kind="mergesort")
        sorted_key = key[order]
        residual = (
            y.astype(np.float64, copy=False) -
            parent_rate[content].astype(np.float64, copy=False)
        )[order]

        starts = np.empty(len(sorted_key), dtype=bool)
        starts[0] = True
        starts[1:] = sorted_key[1:] != sorted_key[:-1]
        positions = np.flatnonzero(starts)
        unique_keys = sorted_key[positions]
        sums = np.add.reduceat(residual, positions)
        ends = np.r_[positions[1:], len(sorted_key)]
        counts = ends - positions
        values = (sums / (counts + strength)).astype(np.float32)
        return PairResidual(unique_keys, values)

    def predict(self, user, content, content_cardinality):
        key = (user.astype(np.int64, copy=False) *
               np.int64(content_cardinality) +
               content.astype(np.int64, copy=False))
        loc = np.searchsorted(self.keys, key)
        out = np.zeros(len(key), dtype=np.float32)
        good = loc < len(self.keys)
        good_idx = np.flatnonzero(good)
        if len(good_idx):
            matched = self.keys[loc[good_idx]] == key[good_idx]
            matched_idx = good_idx[matched]
            out[matched_idx] = self.values[loc[matched_idx]]
        return out


class StatisticalModels:
    def __init__(self, source, y):
        self.prior = float(np.mean(y))
        self.specs = [
            ("video_id", 35.0),
            ("author_id", 35.0),
            ("tag", 45.0),
            ("onehot_feat3", 45.0),
            ("duration_bucket", 50.0),
        ]
        self.rates = {}
        for name, strength in self.specs:
            self.rates[name] = entity_rate(
                source.X[name], y, int(FEATURE_CARDINALITIES[name]),
                self.prior, strength
            )

        user = source.X["user_id"]
        pair_specs = [
            ("author_id", 10.0),
            ("tag", 15.0),
            ("onehot_feat3", 14.0),
            ("duration_bucket", 16.0),
        ]
        self.pairs = {}
        for name, strength in pair_specs:
            self.pairs[name] = PairResidual.fit(
                user, source.X[name], y, self.rates[name],
                int(FEATURE_CARDINALITIES[name]), strength
            )

    def predict(self, target):
        logits = {}
        for name, _ in self.specs:
            logits[name] = safe_logit(
                self.rates[name][target.X[name]]
            ).astype(np.float32)

        entity = (
            0.34 * logits["video_id"] +
            0.34 * logits["author_id"] +
            0.12 * logits["tag"] +
            0.10 * logits["onehot_feat3"] +
            0.10 * logits["duration_bucket"]
        ).astype(np.float32)

        user = target.X["user_id"]
        residuals = {}
        for name in self.pairs:
            residuals[name] = self.pairs[name].predict(
                user, target.X[name], int(FEATURE_CARDINALITIES[name])
            )

        personalized = (
            entity +
            3.4 * (
                0.48 * residuals["author_id"] +
                0.20 * residuals["tag"] +
                0.20 * residuals["onehot_feat3"] +
                0.12 * residuals["duration_bucket"]
            )
        ).astype(np.float32)
        return entity, personalized


class MatrixFactorization(nn.Module):
    def __init__(self, mean_rate):
        super().__init__()
        nu = int(FEATURE_CARDINALITIES["user_id"])
        nv = int(FEATURE_CARDINALITIES["video_id"])
        na = int(FEATURE_CARDINALITIES["author_id"])
        nt = int(FEATURE_CARDINALITIES["tag"])
        nd = int(FEATURE_CARDINALITIES["duration_bucket"])

        self.user = nn.Embedding(nu, MF_RANK)
        self.video = nn.Embedding(nv, MF_RANK)
        self.author = nn.Embedding(na, MF_RANK)
        self.user_bias = nn.Embedding(nu, 1)
        self.video_bias = nn.Embedding(nv, 1)
        self.author_bias = nn.Embedding(na, 1)
        self.tag_bias = nn.Embedding(nt, 1)
        self.duration_bias = nn.Embedding(nd, 1)
        self.global_bias = nn.Parameter(torch.tensor(
            math.log(mean_rate / (1.0 - mean_rate)), dtype=torch.float32
        ))

        with torch.no_grad():
            for emb in (self.user, self.video, self.author):
                emb.weight.normal_(0.0, 0.035)
            for emb in (
                self.user_bias, self.video_bias, self.author_bias,
                self.tag_bias, self.duration_bias
            ):
                emb.weight.zero_()

    def forward(self, u, v, a, tag, duration):
        uv = (self.user(u) * self.video(v)).sum(dim=1)
        ua = (self.user(u) * self.author(a)).sum(dim=1)
        bias = (
            self.user_bias(u).squeeze(1) +
            self.video_bias(v).squeeze(1) +
            self.author_bias(a).squeeze(1) +
            self.tag_bias(tag).squeeze(1) +
            self.duration_bias(duration).squeeze(1)
        )
        return self.global_bias + uv + 0.65 * ua + bias


def extract_mf_arrays(split):
    return (
        np.asarray(split.X["user_id"], dtype=np.int64),
        np.asarray(split.X["video_id"], dtype=np.int64),
        np.asarray(split.X["author_id"], dtype=np.int64),
        np.asarray(split.X["tag"], dtype=np.int64),
        np.asarray(split.X["duration_bucket"], dtype=np.int64),
    )


def fit_mf(arrays, y, epochs=MF_EPOCHS):
    torch.manual_seed(SEED)
    model = MatrixFactorization(float(np.mean(y))).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MF_LR, weight_decay=2e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    tensors = tuple(torch.from_numpy(x) for x in arrays)
    target = torch.from_numpy(y.astype(np.float32, copy=False))
    n = len(y)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, MF_BATCH):
            idx = permutation[start:min(start + MF_BATCH, n)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(x[idx] for x in tensors))
            loss = criterion(logits, target[idx])
            loss.backward()
            optimizer.step()
    return model


def predict_mf(model, arrays, batch=32768):
    model.eval()
    tensors = tuple(torch.from_numpy(x) for x in arrays)
    result = np.empty(len(arrays[0]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(result), batch):
            end = min(start + batch, len(result))
            result[start:end] = model(
                *(x[start:end] for x in tensors)
            ).cpu().numpy()
    return result


def within_user_ranks(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)

    # Stable row index is a deterministic tiebreaker. The blend always
    # contains the continuous incumbent score, so ties in a statistical
    # component do not determine the final ordering by themselves.
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    group_start = np.empty(n, dtype=bool)
    group_start[0] = True
    group_start[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(group_start)
    group_index = np.cumsum(group_start) - 1
    local_rank = np.arange(n, dtype=np.int64) - starts[group_index]
    sizes = np.diff(np.r_[starts, n])

    percentile = (
        (local_rank.astype(np.float64) + 0.5) /
        sizes[group_index].astype(np.float64)
    )
    result = np.empty(n, dtype=np.float64)
    result[order] = percentile
    return result


def combine_sources(a, b):
    class Combined:
        pass

    c = Combined()
    c.X = {}
    for name in a.X:
        c.X[name] = np.concatenate([a.X[name], b.X[name]])
    c.user_id = c.X["user_id"]
    c.video_id = c.X["video_id"]
    return c


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_valid_rank = within_user_ranks(valid.user_id, inc_valid)

# Family 1: hierarchical entity empirical Bayes.
# Family 2: personalized user-content empirical-Bayes residuals.
stats = StatisticalModels(train, y_train)
entity_valid, personal_valid = stats.predict(valid)

# Family 3: low-rank collaborative matrix factorization.
train_mf_arrays = extract_mf_arrays(train)
valid_mf_arrays = extract_mf_arrays(valid)
mf_model = fit_mf(train_mf_arrays, y_train)
mf_valid = predict_mf(mf_model, valid_mf_arrays)

families = {
    "hierarchical_empirical_bayes": entity_valid,
    "personalized_pair_residuals": personal_valid,
    "latent_matrix_factorization": mf_valid,
}

candidate_scores = {}
best_primary = -np.inf
best_name = None
best_family = None
best_alpha = None
best_valid_scores = None
best_raw_scores = None
best_metrics = None

alphas = [0.15, 0.30, 0.50, 0.70]

for family_name, raw_scores in families.items():
    raw_metrics = evaluate(valid.user_id, y_valid, raw_scores)
    candidate_scores[family_name + "_raw"] = float(raw_metrics["primary"])

    family_rank = within_user_ranks(valid.user_id, raw_scores)
    for alpha in alphas:
        blended = (
            (1.0 - alpha) * inc_valid_rank + alpha * family_rank
        )
        metrics = evaluate(valid.user_id, y_valid, blended)
        name = family_name + "_blend_" + str(alpha)
        candidate_scores[name] = float(metrics["primary"])
        if metrics["primary"] > best_primary:
            best_primary = float(metrics["primary"])
            best_name = name
            best_family = family_name
            best_alpha = float(alpha)
            best_valid_scores = blended.copy()
            best_raw_scores = np.asarray(raw_scores).copy()
            best_metrics = metrics

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": best_name,
    "blend_alpha": best_alpha,
    "winner_raw_primary": candidate_scores[best_family + "_raw"],
    "incumbent_primary_recomputed": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    )
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_scores, dtype=np.float64)
    )

# Refit only the selected family on train+validation. Selection was made
# exclusively using the train-only validation experiment above.
combined = combine_sources(train, valid)
y_combined = np.concatenate([
    y_train,
    y_valid.astype(np.float32, copy=False)
])
test = load("test")

if best_family in (
    "hierarchical_empirical_bayes",
    "personalized_pair_residuals"
):
    combined_stats = StatisticalModels(combined, y_combined)
    entity_test, personal_test = combined_stats.predict(test)
    if best_family == "hierarchical_empirical_bayes":
        raw_test = entity_test
    else:
        raw_test = personal_test
else:
    del mf_model
    combined_arrays = tuple(
        np.concatenate([a, b])
        for a, b in zip(train_mf_arrays, valid_mf_arrays)
    )
    refit_mf = fit_mf(combined_arrays, y_combined)
    raw_test = predict_mf(refit_mf, extract_mf_arrays(test))

inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)
inc_test_rank = within_user_ranks(test.user_id, inc_test)
raw_test_rank = within_user_ranks(test.user_id, raw_test)
test_scores = (
    (1.0 - best_alpha) * inc_test_rank +
    best_alpha * raw_test_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed)
}))