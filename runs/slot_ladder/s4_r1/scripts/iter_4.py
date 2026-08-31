import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
n_train = y_train.size

# A four-day half-life emphasizes behavior near the train/evaluation boundary.
train_day = np.asarray(train.date, dtype=np.int64) % 100
recency_weight = np.exp2(-(21.0 - train_day.astype(np.float64)) / 4.0)
global_rate = float(
    np.sum(recency_weight * y_train) / np.sum(recency_weight)
)
EPS = 1e-5


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def weighted_marginal(field, alpha):
    card = int(FEATURE_CARDINALITIES[field])
    x = np.asarray(train.X[field], dtype=np.int64)
    cnt = np.bincount(x, weights=recency_weight, minlength=card)
    pos = np.bincount(
        x, weights=recency_weight * y_train, minlength=card
    )
    rate = (pos + alpha * global_rate) / (cnt + alpha)
    return rate.astype(np.float64), cnt.astype(np.float64)


marginals = {}
marginal_counts = {}
for field, alpha in [
    ("video_id", 18.0),
    ("author_id", 24.0),
    ("tab", 80.0),
    ("tag", 50.0),
    ("duration_bucket", 80.0),
]:
    marginals[field], marginal_counts[field] = weighted_marginal(
        field, alpha
    )


def marginal_score(split):
    # Identity signals get most weight; context terms resolve ties and shifts.
    terms = [
        (0.42, "video_id"),
        (0.30, "author_id"),
        (0.12, "tab"),
        (0.10, "tag"),
        (0.06, "duration_bucket"),
    ]
    out = np.zeros(len(split.user_id), dtype=np.float64)
    for weight, field in terms:
        ids = np.asarray(split.X[field], dtype=np.int64)
        out += weight * logit(marginals[field][ids])
    return out


def build_pair_table(entity_field, alpha):
    card = int(FEATURE_CARDINALITIES[entity_field])
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    entities = np.asarray(train.X[entity_field], dtype=np.int64)
    keys = users * np.int64(card) + entities

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inverse, weights=recency_weight)
    pos = np.bincount(
        inverse, weights=recency_weight * y_train
    )

    prior = marginals[entity_field][entities]
    prior_sum = np.bincount(
        inverse, weights=recency_weight * prior
    )
    pair_prior = prior_sum / np.maximum(cnt, 1e-12)
    rate = (pos + alpha * pair_prior) / (cnt + alpha)

    return (
        unique_keys.astype(np.int64),
        rate.astype(np.float64),
        card,
    )


def lookup_pair(split, entity_field, table):
    unique_keys, rates, card = table
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    entities = np.asarray(split.X[entity_field], dtype=np.int64)
    keys = users * np.int64(card) + entities
    idx = np.searchsorted(unique_keys, keys)
    found = idx < unique_keys.size
    safe = np.minimum(idx, unique_keys.size - 1)
    found &= unique_keys[safe] == keys

    fallback = marginals[entity_field][entities]
    result = fallback.astype(np.float64, copy=True)
    result[found] = rates[safe[found]]
    return result


pair_video = build_pair_table("video_id", alpha=5.0)
pair_author = build_pair_table("author_id", alpha=7.0)
pair_tag = build_pair_table("tag", alpha=10.0)


def personalized_video_score(split):
    pair_rate = lookup_pair(split, "video_id", pair_video)
    base = marginal_score(split)
    return 0.72 * logit(pair_rate) + 0.28 * base


def content_profile_score(split):
    author_rate = lookup_pair(split, "author_id", pair_author)
    tag_rate = lookup_pair(split, "tag", pair_tag)
    base = marginal_score(split)
    return (
        0.48 * logit(author_rate)
        + 0.30 * logit(tag_rate)
        + 0.22 * base
    )


# Causal sequence family: time since the user's most recent positive event
# involving the candidate entity. Only training outcomes are used.
def build_latest_positive_table(field):
    card = int(FEATURE_CARDINALITIES[field])
    mask = y_train > 0.5
    users = np.asarray(train.X["user_id"], dtype=np.int64)[mask]
    entities = np.asarray(train.X[field], dtype=np.int64)[mask]
    times = np.asarray(train.time_ms, dtype=np.int64)[mask]
    keys = users * np.int64(card) + entities

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    latest = np.full(unique_keys.size, np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(latest, inverse, times)
    return unique_keys, latest, card


latest_video = build_latest_positive_table("video_id")
latest_author = build_latest_positive_table("author_id")
latest_tag = build_latest_positive_table("tag")


def latest_affinity(split, field, table, half_life_days):
    unique_keys, latest, card = table
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    entities = np.asarray(split.X[field], dtype=np.int64)
    now = np.asarray(split.time_ms, dtype=np.int64)
    keys = users * np.int64(card) + entities

    idx = np.searchsorted(unique_keys, keys)
    found = idx < unique_keys.size
    safe = np.minimum(idx, unique_keys.size - 1)
    found &= unique_keys[safe] == keys

    out = np.zeros(keys.size, dtype=np.float64)
    if np.any(found):
        age_days = (
            now[found].astype(np.float64)
            - latest[safe[found]].astype(np.float64)
        ) / 86400000.0
        age_days = np.maximum(age_days, 0.0)
        out[found] = np.exp2(-age_days / half_life_days)
    return out


def sequence_score(split):
    v = latest_affinity(split, "video_id", latest_video, 8.0)
    a = latest_affinity(split, "author_id", latest_author, 10.0)
    t = latest_affinity(split, "tag", latest_tag, 12.0)
    # Marginal component makes unseen-history candidates rank meaningfully.
    return 1.35 * v + 0.75 * a + 0.35 * t + 0.20 * marginal_score(split)


# Latent collaborative family: user-to-video and user-to-author dot products.
class LatentCF(nn.Module):
    def __init__(self, rank=24):
        super().__init__()
        nu = int(FEATURE_CARDINALITIES["user_id"])
        nv = int(FEATURE_CARDINALITIES["video_id"])
        na = int(FEATURE_CARDINALITIES["author_id"])
        nt = int(FEATURE_CARDINALITIES["tab"])
        nd = int(FEATURE_CARDINALITIES["duration_bucket"])

        self.user = nn.Embedding(nu, rank)
        self.video = nn.Embedding(nv, rank)
        self.author = nn.Embedding(na, rank)
        self.user_bias = nn.Embedding(nu, 1)
        self.video_bias = nn.Embedding(nv, 1)
        self.author_bias = nn.Embedding(na, 1)
        self.tab_bias = nn.Embedding(nt, 1)
        self.duration_bias = nn.Embedding(nd, 1)
        self.global_bias = nn.Parameter(
            torch.tensor([np.log(global_rate / (1.0 - global_rate))],
                         dtype=torch.float32)
        )

        nn.init.normal_(self.user.weight, std=0.035)
        nn.init.normal_(self.video.weight, std=0.035)
        nn.init.normal_(self.author.weight, std=0.035)
        for emb in [
            self.user_bias,
            self.video_bias,
            self.author_bias,
            self.tab_bias,
            self.duration_bias,
        ]:
            nn.init.zeros_(emb.weight)

    def forward(self, u, v, a, tab, duration):
        ue = self.user(u)
        ve = self.video(v)
        ae = self.author(a)
        interaction = (ue * ve).sum(dim=1)
        interaction += 0.55 * (ue * ae).sum(dim=1)
        bias = (
            self.user_bias(u).squeeze(1)
            + self.video_bias(v).squeeze(1)
            + self.author_bias(a).squeeze(1)
            + self.tab_bias(tab).squeeze(1)
            + self.duration_bias(duration).squeeze(1)
        )
        return self.global_bias + interaction + bias


def tensor_columns(split):
    return [
        np.asarray(split.X[f], dtype=np.int64)
        for f in [
            "user_id",
            "video_id",
            "author_id",
            "tab",
            "duration_bucket",
        ]
    ]


cf = LatentCF(rank=24)
optimizer = torch.optim.AdamW(cf.parameters(), lr=0.004, weight_decay=2e-6)
train_cols = tensor_columns(train)
rng = np.random.default_rng(SEED)
batch_size = 4096

cf.train()
for epoch in range(4):
    order = rng.permutation(n_train)
    for start in range(0, n_train, batch_size):
        idx = order[start:start + batch_size]
        tensors = [torch.from_numpy(col[idx]) for col in train_cols]
        target = torch.from_numpy(y_train[idx])
        logits = cf(*tensors)
        loss = F.binary_cross_entropy_with_logits(logits, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


@torch.inference_mode()
def predict_cf(split):
    cf.eval()
    cols = tensor_columns(split)
    result = np.empty(len(split.user_id), dtype=np.float64)
    for start in range(0, result.size, 32768):
        end = min(start + 32768, result.size)
        tensors = [
            torch.from_numpy(col[start:end]) for col in cols
        ]
        result[start:end] = (
            cf(*tensors).cpu().numpy().astype(np.float64)
        )
    return result


valid_predictions = {
    "recency_empirical_bayes": marginal_score(valid),
    "personalized_video_eb": personalized_video_score(valid),
    "content_profile_eb": content_profile_score(valid),
    "causal_sequence_recency": sequence_score(valid),
    "latent_cf": predict_cf(valid),
}

test_predictions = {
    "recency_empirical_bayes": marginal_score(test),
    "personalized_video_eb": personalized_video_score(test),
    "content_profile_eb": content_profile_score(test),
    "causal_sequence_recency": sequence_score(test),
    "latent_cf": predict_cf(test),
}


def within_user_standardize(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    size = int(users.max()) + 1 if users.size else 0
    count = np.bincount(users, minlength=size).astype(np.float64)
    total = np.bincount(users, weights=scores, minlength=size)
    total2 = np.bincount(users, weights=scores * scores, minlength=size)
    mean = total / np.maximum(count, 1.0)
    var = total2 / np.maximum(count, 1.0) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-8))
    return (scores - mean[users]) / std[users]


shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required for incumbent blending")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_z = within_user_standardize(valid.user_id, inc_valid)
inc_test_z = within_user_standardize(test.user_id, inc_test)

candidate_scores = {}
best_primary = -np.inf
best_name = None
best_valid_scores = None
best_test_scores = None
best_raw_valid = None
best_is_blend = False

for name, vpred in valid_predictions.items():
    metric = evaluate(valid.user_id, valid.y, vpred)
    candidate_scores[name] = float(metric["primary"])
    if metric["primary"] > best_primary:
        best_primary = float(metric["primary"])
        best_name = name
        best_valid_scores = vpred
        best_test_scores = test_predictions[name]
        best_raw_valid = vpred
        best_is_blend = False

    vz = within_user_standardize(valid.user_id, vpred)
    tz = within_user_standardize(test.user_id, test_predictions[name])

    # Weight is the contribution from the new family.
    for own_weight in [0.20, 0.35, 0.50, 0.65]:
        blend_valid = (
            own_weight * vz + (1.0 - own_weight) * inc_valid_z
        )
        blend_test = (
            own_weight * tz + (1.0 - own_weight) * inc_test_z
        )
        blend_name = "%s_blend_%.2f" % (name, own_weight)
        blend_metric = evaluate(
            valid.user_id, valid.y, blend_valid
        )
        candidate_scores[blend_name] = float(blend_metric["primary"])

        if blend_metric["primary"] > best_primary:
            best_primary = float(blend_metric["primary"])
            best_name = blend_name
            best_valid_scores = blend_valid
            best_test_scores = blend_test
            best_raw_valid = vpred
            best_is_blend = True

final_metrics = evaluate(
    valid.user_id, valid.y, best_valid_scores
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )
    if best_is_blend:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

print(
    "FINDINGS selected=%s recency_global_rate=%.6f pair_sizes=%d/%d/%d"
    % (
        best_name,
        global_rate,
        pair_video[0].size,
        pair_author[0].size,
        pair_tag[0].size,
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.3f}'
    % (
        float(final_metrics["primary"]),
        float(final_metrics["gauc"]),
        float(final_metrics["ndcg@5"]),
        elapsed,
    )
)