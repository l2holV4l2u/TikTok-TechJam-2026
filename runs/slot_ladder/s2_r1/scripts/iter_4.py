import os
import time
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 18427
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, values, users))
    sorted_users = users[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(new_group)
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    positions = np.arange(n, dtype=np.int64) - np.repeat(starts, counts)
    ranked = (positions.astype(np.float64) + 0.5) / np.repeat(counts, counts)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


class MarginalRate:
    def __init__(self, values, labels, cardinality, prior_strength):
        values = np.asarray(values, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float64)
        self.global_rate = float(labels.mean())
        count = np.bincount(values, minlength=cardinality).astype(np.float64)
        positive = np.bincount(
            values, weights=labels, minlength=cardinality
        ).astype(np.float64)
        self.rate = (
            positive + prior_strength * self.global_rate
        ) / (count + prior_strength)

    def predict(self, values):
        values = np.asarray(values, dtype=np.int64)
        valid = (values >= 0) & (values < len(self.rate))
        result = np.full(values.shape, self.global_rate, dtype=np.float64)
        result[valid] = self.rate[values[valid]]
        return result


class PairRate:
    def __init__(
        self, left, right, labels, right_cardinality,
        prior_rate, prior_strength
    ):
        left = np.asarray(left, dtype=np.int64)
        right = np.asarray(right, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float64)

        keys = left * np.int64(right_cardinality) + right
        order = np.argsort(keys, kind="mergesort")
        sorted_keys = keys[order]
        sorted_y = labels[order]

        starts = np.r_[0, np.flatnonzero(
            sorted_keys[1:] != sorted_keys[:-1]
        ) + 1]
        self.keys = sorted_keys[starts]
        counts = np.diff(np.r_[starts, len(sorted_keys)]).astype(np.float64)
        positives = np.add.reduceat(sorted_y, starts)

        right_prior = np.asarray(prior_rate, dtype=np.float64)[right[order][starts]]
        self.rates = (
            positives + prior_strength * right_prior
        ) / (counts + prior_strength)
        self.right_cardinality = int(right_cardinality)
        self.default_rate = np.asarray(prior_rate, dtype=np.float64)

    def predict(self, left, right):
        left = np.asarray(left, dtype=np.int64)
        right = np.asarray(right, dtype=np.int64)
        keys = left * np.int64(self.right_cardinality) + right
        positions = np.searchsorted(self.keys, keys)
        positions_safe = np.minimum(positions, len(self.keys) - 1)
        found = (
            (positions < len(self.keys))
            & (self.keys[positions_safe] == keys)
        )
        result = self.default_rate[
            np.clip(right, 0, len(self.default_rate) - 1)
        ].copy()
        result[found] = self.rates[positions_safe[found]]
        return result


class LastPositivePair:
    def __init__(self, left, right, labels, times, right_cardinality):
        left = np.asarray(left, dtype=np.int64)
        right = np.asarray(right, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int8)
        times = np.asarray(times, dtype=np.int64)

        positive = labels == 1
        left = left[positive]
        right = right[positive]
        times = times[positive]

        keys = left * np.int64(right_cardinality) + right
        order = np.argsort(keys, kind="mergesort")
        keys = keys[order]
        times = times[order]

        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        self.keys = keys[starts]
        self.latest = np.maximum.reduceat(times, starts)
        self.right_cardinality = int(right_cardinality)

    def freshness(self, left, right, query_time, half_life_days):
        left = np.asarray(left, dtype=np.int64)
        right = np.asarray(right, dtype=np.int64)
        query_time = np.asarray(query_time, dtype=np.int64)

        keys = left * np.int64(self.right_cardinality) + right
        positions = np.searchsorted(self.keys, keys)
        positions_safe = np.minimum(positions, len(self.keys) - 1)
        found = (
            (positions < len(self.keys))
            & (self.keys[positions_safe] == keys)
        )

        result = np.zeros(len(keys), dtype=np.float64)
        if np.any(found):
            age_days = np.maximum(
                0.0,
                (
                    query_time[found].astype(np.float64)
                    - self.latest[positions_safe[found]].astype(np.float64)
                ) / 86400000.0
            )
            result[found] = np.exp(
                -np.log(2.0) * age_days / half_life_days
            )
        return result


def get_history_matrix(split_name, split):
    blocks = []
    names = []

    for key in ("video_id", "author_id"):
        data = historical_features(split_name, key=key)
        for name in sorted(data):
            value = np.asarray(data[name], dtype=np.float64)
            if value.ndim != 1 or len(value) != len(split.user_id):
                continue
            value = np.nan_to_num(
                value, nan=0.0, posinf=0.0, neginf=0.0
            )
            if np.nanpercentile(np.abs(value), 95) > 20.0:
                value = np.sign(value) * np.log1p(np.abs(value))
            blocks.append(value)
            names.append(key + ":" + name)

    duration = np.asarray(split.num["duration_ms"], dtype=np.float64)
    duration = np.nan_to_num(duration, nan=0.0, posinf=0.0, neginf=0.0)
    blocks.append(np.log1p(np.maximum(duration, 0.0)))
    names.append("numeric:log_duration")

    for field in ("tab", "tag", "duration_bucket", "upload_type", "hour"):
        blocks.append(np.asarray(split.X[field], dtype=np.float64))
        names.append("categorical_numeric:" + field)

    return np.column_stack(blocks).astype(np.float32), names


def fit_linear_history(x, y):
    n, d = x.shape
    model = torch.nn.Linear(d, 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.constant_(
        model.bias,
        float(np.log(y.mean() / (1.0 - y.mean())))
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.015, weight_decay=2e-4
    )
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.float32))
    generator = torch.Generator().manual_seed(SEED)

    batch_size = 16384
    for epoch in range(5):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            logits = model(xt[idx]).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def predict_linear(model, x):
    result = np.empty(len(x), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 32768):
            end = min(start + 32768, len(x))
            result[start:end] = model(
                torch.from_numpy(x[start:end])
            ).squeeze(1).numpy().astype(np.float64)
    return result


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float64)
global_rate = float(y_train.mean())

cards = {}
for field in ("video_id", "author_id", "tag", "tab", "duration_bucket"):
    cards[field] = int(max(
        np.asarray(train.X[field]).max(),
        np.asarray(valid.X[field]).max()
    )) + 1

marginals = {
    "video_id": MarginalRate(
        train.X["video_id"], y_train, cards["video_id"], 20.0
    ),
    "author_id": MarginalRate(
        train.X["author_id"], y_train, cards["author_id"], 30.0
    ),
    "tag": MarginalRate(
        train.X["tag"], y_train, cards["tag"], 80.0
    ),
    "tab": MarginalRate(
        train.X["tab"], y_train, cards["tab"], 200.0
    ),
    "duration_bucket": MarginalRate(
        train.X["duration_bucket"], y_train,
        cards["duration_bucket"], 150.0
    ),
}

user_author = PairRate(
    train.user_id,
    train.X["author_id"],
    y_train,
    cards["author_id"],
    marginals["author_id"].rate,
    8.0,
)
user_tag = PairRate(
    train.user_id,
    train.X["tag"],
    y_train,
    cards["tag"],
    marginals["tag"].rate,
    15.0,
)
user_video = PairRate(
    train.user_id,
    train.X["video_id"],
    y_train,
    cards["video_id"],
    marginals["video_id"].rate,
    5.0,
)

last_author = LastPositivePair(
    train.user_id, train.X["author_id"], train.y,
    train.time_ms, cards["author_id"]
)
last_tag = LastPositivePair(
    train.user_id, train.X["tag"], train.y,
    train.time_ms, cards["tag"]
)
last_video = LastPositivePair(
    train.user_id, train.X["video_id"], train.y,
    train.time_ms, cards["video_id"]
)


def empirical_bayes_scores(split):
    video = marginals["video_id"].predict(split.X["video_id"])
    author = marginals["author_id"].predict(split.X["author_id"])
    tag = marginals["tag"].predict(split.X["tag"])
    tab = marginals["tab"].predict(split.X["tab"])
    duration = marginals["duration_bucket"].predict(
        split.X["duration_bucket"]
    )
    return (
        0.42 * safe_logit(video)
        + 0.30 * safe_logit(author)
        + 0.13 * safe_logit(tag)
        + 0.10 * safe_logit(tab)
        + 0.05 * safe_logit(duration)
    )


def conditional_pair_scores(split):
    uv = user_video.predict(split.user_id, split.X["video_id"])
    ua = user_author.predict(split.user_id, split.X["author_id"])
    ut = user_tag.predict(split.user_id, split.X["tag"])
    video = marginals["video_id"].predict(split.X["video_id"])
    author = marginals["author_id"].predict(split.X["author_id"])
    return (
        0.34 * safe_logit(uv)
        + 0.28 * safe_logit(ua)
        + 0.18 * safe_logit(ut)
        + 0.12 * safe_logit(video)
        + 0.08 * safe_logit(author)
    )


def sequence_scores(split):
    base = empirical_bayes_scores(split)
    fresh_video = last_video.freshness(
        split.user_id, split.X["video_id"], split.time_ms, 12.0
    )
    fresh_author = last_author.freshness(
        split.user_id, split.X["author_id"], split.time_ms, 10.0
    )
    fresh_tag = last_tag.freshness(
        split.user_id, split.X["tag"], split.time_ms, 8.0
    )
    return (
        base
        + 1.10 * fresh_video
        + 0.70 * fresh_author
        + 0.35 * fresh_tag
    )


history_train, history_names = get_history_matrix("train", train)
history_valid, valid_history_names = get_history_matrix("valid", valid)
if history_names != valid_history_names:
    raise ValueError("Historical feature schema differs by split")

mean = history_train.mean(axis=0, dtype=np.float64)
std = history_train.std(axis=0, dtype=np.float64)
std = np.where(std < 1e-5, 1.0, std)
history_train = np.clip(
    (history_train - mean) / std, -8.0, 8.0
).astype(np.float32)
history_valid = np.clip(
    (history_valid - mean) / std, -8.0, 8.0
).astype(np.float32)

history_model = fit_linear_history(
    history_train, np.asarray(train.y, dtype=np.float32)
)

valid_predictions = {
    "empirical_bayes": empirical_bayes_scores(valid),
    "conditional_pair_eb": conditional_pair_scores(valid),
    "last_positive_sequence": sequence_scores(valid),
    "historical_linear": predict_linear(history_model, history_valid),
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")

candidate_scores = {}
candidate_specs = {}

for name, prediction in valid_predictions.items():
    score = evaluate(valid.user_id, valid.y, prediction)["primary"]
    candidate_scores[name] = float(score)
    candidate_specs[name] = ("raw", name, 1.0)

inc_metric = evaluate(valid.user_id, valid.y, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metric["primary"])
candidate_specs["trusted_incumbent"] = ("incumbent", None, 0.0)

inc_rank = within_user_rank(valid.user_id, inc_valid)
blend_weights = (0.15, 0.25, 0.35, 0.50, 0.65)

for name, prediction in valid_predictions.items():
    own_rank = within_user_rank(valid.user_id, prediction)
    for alpha in blend_weights:
        blended = alpha * own_rank + (1.0 - alpha) * inc_rank
        blend_name = "%s_rankblend_%.2f" % (name, alpha)
        score = evaluate(
            valid.user_id, valid.y, blended
        )["primary"]
        candidate_scores[blend_name] = float(score)
        candidate_specs[blend_name] = ("blend", name, alpha)

winner = max(candidate_scores, key=candidate_scores.get)
winner_kind, winner_name, winner_alpha = candidate_specs[winner]

if winner_kind == "raw":
    valid_scores = valid_predictions[winner_name]
elif winner_kind == "blend":
    valid_scores = (
        winner_alpha
        * within_user_rank(valid.user_id, valid_predictions[winner_name])
        + (1.0 - winner_alpha) * inc_rank
    )
else:
    valid_scores = inc_valid.copy()

metrics = evaluate(valid.user_id, valid.y, valid_scores)
best_raw_name = max(
    valid_predictions,
    key=lambda name: candidate_scores[name]
)

print("FINDINGS " + json.dumps({
    "winner": winner,
    "best_raw_family": best_raw_name,
    "best_raw_primary": candidate_scores[best_raw_name],
    "incumbent_primary": candidate_scores["trusted_incumbent"],
    "historical_feature_count": len(history_names),
    "historical_feature_names": history_names,
}, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64)
    )
    if winner_kind in ("blend", "incumbent"):
        raw_name = winner_name if winner_name is not None else best_raw_name
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(valid_predictions[raw_name], dtype=np.float64)
        )

test = load("test")
history_test, test_history_names = get_history_matrix("test", test)
if history_names != test_history_names:
    raise ValueError("Historical test feature schema mismatch")
history_test = np.clip(
    (history_test - mean) / std, -8.0, 8.0
).astype(np.float32)

test_predictions = {
    "empirical_bayes": empirical_bayes_scores(test),
    "conditional_pair_eb": conditional_pair_scores(test),
    "last_positive_sequence": sequence_scores(test),
    "historical_linear": predict_linear(history_model, history_test),
}

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")

if winner_kind == "raw":
    test_scores = test_predictions[winner_name]
elif winner_kind == "blend":
    test_scores = (
        winner_alpha
        * within_user_rank(test.user_id, test_predictions[winner_name])
        + (1.0 - winner_alpha)
        * within_user_rank(test.user_id, inc_test)
    )
else:
    test_scores = inc_test.copy()

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