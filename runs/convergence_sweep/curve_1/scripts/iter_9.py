import os
import time
import json
import math
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
torch.set_num_threads(min(8, os.cpu_count() or 8))

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
CARDINALITIES = [FEATURE_CARDINALITIES[name] for name in FIELDS]
BATCH_SIZE = 32768
DIM = 12
EPOCHS = 2


def stack_fields(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int64)
        for name in FIELDS
    ])


def date_ordinals(dates):
    dates = np.asarray(dates, dtype=np.int32)
    unique = np.unique(dates)
    mapping = {}
    for value in unique:
        text = str(int(value))
        day = np.datetime64(
            "{}-{}-{}".format(text[:4], text[4:6], text[6:8]), "D"
        ).astype(np.int64)
        mapping[int(value)] = int(day)
    return np.asarray([mapping[int(x)] for x in dates], dtype=np.int64)


def recency_weights(dates, half_life):
    ordinal = date_ordinals(dates)
    age = ordinal.max() - ordinal
    weights = np.exp2(-age.astype(np.float32) / np.float32(half_life))
    weights /= weights.mean()
    return weights.astype(np.float32)


def init_embedding(module, std=0.025):
    nn.init.normal_(module.weight, mean=0.0, std=std)


class NeuralFM(nn.Module):
    """
    NFM forms its prediction from the summed pairwise embedding interaction
    vector, then learns nonlinear combinations of those interaction channels.
    """

    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, DIM)
            for cardinality in CARDINALITIES
        ])
        self.linear = nn.ModuleList([
            nn.Embedding(cardinality, 1)
            for cardinality in CARDINALITIES
        ])
        self.bias = nn.Parameter(torch.zeros(()))
        self.interaction_net = nn.Sequential(
            nn.BatchNorm1d(DIM),
            nn.Linear(DIM, 48),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

        for embedding in self.embeddings:
            init_embedding(embedding)
        for embedding in self.linear:
            init_embedding(embedding)

    def forward(self, x):
        stacked = torch.stack([
            embedding(x[:, index])
            for index, embedding in enumerate(self.embeddings)
        ], dim=1)
        summed = stacked.sum(dim=1)
        bi_interaction = 0.5 * (
            summed.square() - stacked.square().sum(dim=1)
        )

        score = self.bias.expand(x.shape[0])
        for index, linear in enumerate(self.linear):
            score = score + linear(x[:, index]).squeeze(1)
        score = score + self.interaction_net(bi_interaction).squeeze(1)
        return score


class ESMM(nn.Module):
    """
    The relevance score is formed as P(click) * P(long_view | engagement).
    Click is used only as an auxiliary training target, never as an input.
    """

    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, DIM)
            for cardinality in CARDINALITIES
        ])
        input_dim = len(FIELDS) * DIM

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 80),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(80, 40),
            nn.ReLU(),
        )
        self.click_tower = nn.Sequential(
            nn.Linear(40, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        self.conditional_tower = nn.Sequential(
            nn.Linear(40, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

        for embedding in self.embeddings:
            init_embedding(embedding)

    def probabilities(self, x):
        embedded = torch.cat([
            embedding(x[:, index])
            for index, embedding in enumerate(self.embeddings)
        ], dim=1)
        shared = self.shared(embedded)
        click_probability = torch.sigmoid(
            self.click_tower(shared).squeeze(1)
        )
        conditional_probability = torch.sigmoid(
            self.conditional_tower(shared).squeeze(1)
        )
        long_probability = click_probability * conditional_probability
        return click_probability, conditional_probability, long_probability

    def forward(self, x):
        click_probability, conditional_probability, _ = self.probabilities(x)
        return (
            torch.log(click_probability.clamp_min(1e-7))
            + torch.log(conditional_probability.clamp_min(1e-7))
        )


def train_nfm(model, x, labels, weights):
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(labels.astype(np.float32, copy=False))
    w_tensor = torch.from_numpy(weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0018, weight_decay=2e-6
    )
    rng = np.random.RandomState(SEED + 10)

    for epoch in range(EPOCHS):
        permutation = rng.permutation(x.shape[0])
        total_loss = 0.0
        total_weight = 0.0
        model.train()

        for start in range(0, x.shape[0], BATCH_SIZE):
            indices = torch.from_numpy(
                permutation[start:start + BATCH_SIZE]
            )
            xb = x_tensor.index_select(0, indices)
            yb = y_tensor.index_select(0, indices)
            wb = w_tensor.index_select(0, indices)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(row_loss * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(torch.sum(row_loss * wb).detach())
            total_weight += float(torch.sum(wb))

        print(
            "FINDINGS family=nfm epoch={} weighted_loss={:.6f}".format(
                epoch + 1, total_loss / max(total_weight, 1e-9)
            ),
            flush=True,
        )
    return model


def train_esmm(model, x, long_labels, click_labels, weights):
    x_tensor = torch.from_numpy(x)
    long_tensor = torch.from_numpy(
        long_labels.astype(np.float32, copy=False)
    )
    click_tensor = torch.from_numpy(
        click_labels.astype(np.float32, copy=False)
    )
    w_tensor = torch.from_numpy(weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0015, weight_decay=2e-6
    )
    rng = np.random.RandomState(SEED + 20)

    for epoch in range(EPOCHS):
        permutation = rng.permutation(x.shape[0])
        total_long = 0.0
        total_click = 0.0
        total_weight = 0.0
        model.train()

        for start in range(0, x.shape[0], BATCH_SIZE):
            indices = torch.from_numpy(
                permutation[start:start + BATCH_SIZE]
            )
            xb = x_tensor.index_select(0, indices)
            y_long = long_tensor.index_select(0, indices)
            y_click = click_tensor.index_select(0, indices)
            wb = w_tensor.index_select(0, indices)

            optimizer.zero_grad(set_to_none=True)
            p_click, _, p_long = model.probabilities(xb)
            long_loss = F.binary_cross_entropy(
                p_long.clamp(1e-6, 1.0 - 1e-6),
                y_long,
                reduction="none",
            )
            click_loss = F.binary_cross_entropy(
                p_click.clamp(1e-6, 1.0 - 1e-6),
                y_click,
                reduction="none",
            )
            row_loss = long_loss + 0.35 * click_loss
            loss = torch.sum(row_loss * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_long += float(torch.sum(long_loss * wb).detach())
            total_click += float(torch.sum(click_loss * wb).detach())
            total_weight += float(torch.sum(wb))

        print(
            "FINDINGS family=esmm epoch={} long_loss={:.6f} "
            "click_loss={:.6f}".format(
                epoch + 1,
                total_long / max(total_weight, 1e-9),
                total_click / max(total_weight, 1e-9),
            ),
            flush=True,
        )
    return model


@torch.no_grad()
def predict_torch(model, x):
    model.eval()
    x_tensor = torch.from_numpy(x)
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, x.shape[0])
        result[start:end] = (
            model(x_tensor[start:end]).cpu().numpy().astype(np.float64)
        )
    return result


class TemporalEmpiricalBayes:
    """
    Each entity receives a recency-weighted base logit and a count-shrunk
    linear logit trend across training days. The trend is extrapolated to the
    date of each validation/test impression.
    """

    def __init__(self):
        self.tables = {}
        self.last_train_day = None
        self.global_logit = 0.0
        self.fields = ["video_id", "author_id", "tab", "duration_bucket"]
        self.coefficients = {
            "video_id": 0.90,
            "author_id": 0.65,
            "tab": 0.35,
            "duration_bucket": 0.40,
        }
        self.priors = {
            "video_id": 35.0,
            "author_id": 45.0,
            "tab": 100.0,
            "duration_bucket": 80.0,
        }

    def fit(self, split):
        labels = np.asarray(split.y, dtype=np.float64)
        day_values = date_ordinals(split.date)
        unique_days = np.unique(day_values)
        self.last_train_day = int(unique_days.max())

        recency = np.exp2(
            -(self.last_train_day - day_values).astype(np.float64) / 4.0
        )
        global_rate = np.sum(recency * labels) / np.sum(recency)
        global_rate = np.clip(global_rate, 1e-5, 1.0 - 1e-5)
        self.global_logit = float(
            np.log(global_rate / (1.0 - global_rate))
        )

        for name in self.fields:
            ids = np.asarray(split.X[name], dtype=np.int64)
            cardinality = FEATURE_CARDINALITIES[name]
            prior = self.priors[name]

            total_count = np.bincount(
                ids, weights=recency, minlength=cardinality
            )
            total_sum = np.bincount(
                ids, weights=recency * labels, minlength=cardinality
            )
            base_rate = (
                total_sum + prior * global_rate
            ) / (total_count + prior)
            base_rate = np.clip(base_rate, 1e-5, 1.0 - 1e-5)
            base_logit = np.log(base_rate / (1.0 - base_rate))

            trend_num = np.zeros(cardinality, dtype=np.float64)
            trend_den = np.zeros(cardinality, dtype=np.float64)
            center = float(np.mean(unique_days))

            for day in unique_days:
                mask = day_values == day
                day_ids = ids[mask]
                count = np.bincount(
                    day_ids, minlength=cardinality
                ).astype(np.float64)
                positive = np.bincount(
                    day_ids,
                    weights=labels[mask],
                    minlength=cardinality,
                )
                day_rate = (
                    positive + 25.0 * global_rate
                ) / (count + 25.0)
                day_rate = np.clip(day_rate, 1e-5, 1.0 - 1e-5)
                day_logit = np.log(day_rate / (1.0 - day_rate))

                x_day = float(day) - center
                reliability = count / (count + 25.0)
                trend_num += reliability * x_day * (
                    day_logit - base_logit
                )
                trend_den += reliability * x_day * x_day

            slope = trend_num / np.maximum(trend_den, 1e-8)
            slope *= total_count / (total_count + 300.0)
            slope = np.clip(slope, -0.04, 0.04)

            self.tables[name] = (
                base_logit.astype(np.float32),
                slope.astype(np.float32),
            )
        return self

    def predict(self, split):
        target_days = date_ordinals(split.date)
        horizon = np.clip(
            target_days - self.last_train_day, 1, 20
        ).astype(np.float64)
        score = np.full(
            target_days.shape[0], self.global_logit, dtype=np.float64
        )

        for name in self.fields:
            ids = np.asarray(split.X[name], dtype=np.int64)
            base, slope = self.tables[name]
            score += self.coefficients[name] * (
                base[ids].astype(np.float64)
                - self.global_logit
                + slope[ids].astype(np.float64) * horizon
            )
        return score


def within_user_rank(user_ids, scores):
    """
    Convert each candidate to a common within-user percentile scale. This
    preserves every standalone ordering and makes blend weights comparable.
    """
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = values.size

    order = np.lexsort((np.arange(n, dtype=np.int64), values, users))
    sorted_users = users[order]
    group_start_mask = np.empty(n, dtype=bool)
    group_start_mask[0] = True
    group_start_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(group_start_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    normalized = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    normalized[multi] = (
        positions[multi] / (repeated_lengths[multi] - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = normalized
    return result


tr = load("train")
va = load("valid")
te = load("test")

x_train = stack_fields(tr)
x_valid = stack_fields(va)
x_test = stack_fields(te)
labels = np.asarray(tr.y, dtype=np.int8)

nfm_weights = recency_weights(tr.date, half_life=4.0)
esmm_weights = recency_weights(tr.date, half_life=1.75)

nfm = train_nfm(NeuralFM(), x_train, labels, nfm_weights)
nfm_valid_raw = predict_torch(nfm, x_valid)
nfm_test_raw = predict_torch(nfm, x_test)
del nfm

click_labels = np.asarray(tr.aux["is_click"], dtype=np.float32)
print(
    "FINDINGS auxiliary click_rate={:.6f} long_rate={:.6f}".format(
        float(click_labels.mean()), float(labels.mean())
    ),
    flush=True,
)

esmm = train_esmm(
    ESMM(), x_train, labels, click_labels, esmm_weights
)
esmm_valid_raw = predict_torch(esmm, x_valid)
esmm_test_raw = predict_torch(esmm, x_test)
del esmm

temporal = TemporalEmpiricalBayes().fit(tr)
temporal_valid_raw = temporal.predict(va)
temporal_test_raw = temporal.predict(te)

shared = os.environ.get("SHARED_ARTIFACTS")
incumbent_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
incumbent_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = within_user_rank(va.user_id, incumbent_valid)
inc_test_rank = within_user_rank(te.user_id, incumbent_test)

families = {
    "nfm": (
        within_user_rank(va.user_id, nfm_valid_raw),
        within_user_rank(te.user_id, nfm_test_raw),
    ),
    "esmm": (
        within_user_rank(va.user_id, esmm_valid_raw),
        within_user_rank(te.user_id, esmm_test_raw),
    ),
    "temporal_empirical_bayes": (
        within_user_rank(va.user_id, temporal_valid_raw),
        within_user_rank(te.user_id, temporal_test_raw),
    ),
}

candidate_scores = {}
best_primary = -np.inf
best_valid = None
best_test = None
best_raw_valid = None
best_name = None
best_metrics = None
best_alpha = None

alphas = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0]

for family_name, (valid_rank, test_rank) in families.items():
    standalone_metrics = evaluate(va.user_id, va.y, valid_rank)
    candidate_scores[family_name + "_standalone"] = float(
        standalone_metrics["primary"]
    )

    family_best = -np.inf
    family_best_alpha = None

    for alpha in alphas:
        blended_valid = (
            (1.0 - alpha) * inc_valid_rank + alpha * valid_rank
        )
        metrics = evaluate(va.user_id, va.y, blended_valid)
        primary = float(metrics["primary"])

        if primary > family_best:
            family_best = primary
            family_best_alpha = alpha

        if primary > best_primary:
            best_primary = primary
            best_valid = blended_valid.copy()
            best_test = (
                (1.0 - alpha) * inc_test_rank + alpha * test_rank
            )
            best_raw_valid = valid_rank.copy()
            best_name = family_name
            best_metrics = metrics
            best_alpha = alpha

    candidate_scores[
        family_name + "_best_blend"
    ] = float(family_best)
    print(
        "FINDINGS family={} best_blend_alpha={:.2f} "
        "best_primary={:.6f}".format(
            family_name, family_best_alpha, family_best
        ),
        flush=True,
    )

print(
    "FINDINGS selected_family={} selected_alpha={:.2f}".format(
        best_name, best_alpha
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }),
    flush=True,
)