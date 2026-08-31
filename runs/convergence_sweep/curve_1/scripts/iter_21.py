import os
import time
import json
import random

import numpy as np
import torch
from torch import nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
THREADS = min(8, os.cpu_count() or 8)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)


PAIR_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat7",
    "onehot_feat1",
    "upload_type",
    "music_type",
    "hour",
]

BACKOFF_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat7",
    "upload_type",
    "music_type",
]


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    positions = np.searchsorted(unique_dates, dates)
    age = unique_dates.size - 1 - positions
    weight = np.exp2(-age.astype(np.float64) / float(half_life))
    weight /= np.mean(weight)
    return weight.astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked[multi] = (
        positions[multi]
        / (repeated_lengths[multi].astype(np.float64) - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def rank_blend(user_ids, left, right, alpha):
    left_rank = within_user_rank(user_ids, left)
    right_rank = within_user_rank(user_ids, right)
    return (1.0 - alpha) * left_rank + alpha * right_rank


def make_pair_indices(split, labels, maximum_pairs=2200000):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    rows = np.arange(users.size, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    sorted_users = users[order]
    sorted_labels = labels[order]

    positive_parts = []
    negative_parts = []

    # Multiple deterministic offsets provide local and longer-range
    # comparisons without constructing every quadratic user pair.
    for shift in (1, 2, 4, 8, 16, 32, 64):
        if shift >= order.size:
            continue

        left = np.arange(0, order.size - shift, dtype=np.int64)
        right = left + shift
        eligible = (
            (sorted_users[left] == sorted_users[right])
            & (sorted_labels[left] != sorted_labels[right])
        )
        left = left[eligible]
        right = right[eligible]

        left_positive = sorted_labels[left] == 1
        positive_parts.append(
            np.where(left_positive, order[left], order[right])
        )
        negative_parts.append(
            np.where(left_positive, order[right], order[left])
        )

    positive = np.concatenate(positive_parts).astype(np.int64, copy=False)
    negative = np.concatenate(negative_parts).astype(np.int64, copy=False)

    rng = np.random.default_rng(SEED)
    if positive.size > maximum_pairs:
        selected = rng.choice(
            positive.size, size=maximum_pairs, replace=False
        )
        positive = positive[selected]
        negative = negative[selected]

    permutation = rng.permutation(positive.size)
    return positive[permutation], negative[permutation]


class ConditionalBradleyTerry(nn.Module):
    def __init__(self, fields):
        super().__init__()
        self.fields = list(fields)
        self.effects = nn.ModuleDict()
        for field in self.fields:
            cardinality = int(FEATURE_CARDINALITIES[field])
            embedding = nn.Embedding(cardinality, 1)
            nn.init.zeros_(embedding.weight)
            self.effects[field] = embedding

        # Gates let the conditional likelihood suppress unstable fields,
        # but prediction remains an additive random-utility model.
        self.gates = nn.Parameter(torch.ones(len(self.fields)))

    def score(self, field_tensors, indices):
        result = torch.zeros(
            indices.shape[0], dtype=torch.float32, device=indices.device
        )
        gates = torch.tanh(self.gates)
        for j, field in enumerate(self.fields):
            values = field_tensors[field][indices]
            result = result + gates[j] * self.effects[field](values).squeeze(1)
        return result

    def score_matrix(self, field_arrays, batch_size=131072):
        n = len(next(iter(field_arrays.values())))
        output = np.empty(n, dtype=np.float32)
        self.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                score = torch.zeros(end - start, dtype=torch.float32)
                gates = torch.tanh(self.gates).cpu()
                for j, field in enumerate(self.fields):
                    ids = torch.from_numpy(
                        np.asarray(field_arrays[field][start:end], dtype=np.int64)
                    )
                    score += (
                        gates[j]
                        * self.effects[field](ids).squeeze(1).cpu()
                    )
                output[start:end] = score.numpy()
        return output.astype(np.float64)


def fit_bradley_terry(train, labels, positive, negative, row_weights):
    field_tensors = {
        field: torch.from_numpy(
            np.asarray(train.X[field], dtype=np.int64)
        )
        for field in PAIR_FIELDS
    }
    positive_tensor = torch.from_numpy(positive)
    negative_tensor = torch.from_numpy(negative)
    pair_weight = torch.from_numpy(
        np.sqrt(row_weights[positive] * row_weights[negative]).astype(
            np.float32
        )
    )

    model = ConditionalBradleyTerry(PAIR_FIELDS)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.035, weight_decay=2e-5
    )

    n_pairs = positive.size
    batch_size = 65536
    generator = torch.Generator()
    generator.manual_seed(SEED + 17)

    model.train()
    for epoch in range(6):
        permutation = torch.randperm(n_pairs, generator=generator)
        epoch_loss = 0.0
        epoch_weight = 0.0

        for start in range(0, n_pairs, batch_size):
            selected = permutation[start:start + batch_size]
            pos_idx = positive_tensor[selected]
            neg_idx = negative_tensor[selected]
            weights = pair_weight[selected]

            pos_score = model.score(field_tensors, pos_idx)
            neg_score = model.score(field_tensors, neg_idx)
            margin = pos_score - neg_score

            loss_vector = torch.nn.functional.softplus(-margin)
            loss = torch.sum(loss_vector * weights) / torch.sum(weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(torch.sum(loss_vector * weights))
            epoch_weight += float(torch.sum(weights))

        print(
            "FINDINGS bradley_epoch=%d weighted_pair_loss=%.6f"
            % (epoch + 1, epoch_loss / max(epoch_weight, 1e-12))
        )

    return model


def fit_backoff_tables(train, labels, weights):
    labels = np.asarray(labels, dtype=np.float64)
    weights64 = np.asarray(weights, dtype=np.float64)
    global_rate = np.sum(weights64 * labels) / np.sum(weights64)
    global_rate = float(np.clip(global_rate, 1e-5, 1.0 - 1e-5))
    global_logit = np.log(global_rate / (1.0 - global_rate))

    # Stronger priors for finer identities; coarser content attributes
    # are permitted to move farther from the global rate.
    prior_by_field = {
        "video_id": 180.0,
        "author_id": 220.0,
        "tag": 300.0,
        "duration_bucket": 600.0,
        "tab": 900.0,
        "onehot_feat3": 260.0,
        "onehot_feat8": 320.0,
        "onehot_feat7": 450.0,
        "upload_type": 700.0,
        "music_type": 700.0,
    }

    tables = {}
    reliabilities = {}
    for field in BACKOFF_FIELDS:
        ids = np.asarray(train.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])
        total = np.bincount(
            ids, weights=weights64, minlength=cardinality
        )
        positive = np.bincount(
            ids, weights=weights64 * labels, minlength=cardinality
        )

        prior = prior_by_field[field]
        rate = (
            positive + prior * global_rate
        ) / (total + prior)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        effect = np.log(rate / (1.0 - rate)) - global_logit

        # Reliability keeps rare categories close to their parent/global
        # estimate while allowing repeated entities to contribute fully.
        reliability = total / (total + prior)
        tables[field] = effect.astype(np.float32)
        reliabilities[field] = reliability.astype(np.float32)

    return tables, reliabilities, global_logit


def predict_backoff(split, tables, reliabilities):
    score = np.zeros(len(split.user_id), dtype=np.float64)

    field_scale = {
        "video_id": 1.00,
        "author_id": 0.65,
        "tag": 0.35,
        "duration_bucket": 0.40,
        "tab": 0.50,
        "onehot_feat3": 0.30,
        "onehot_feat8": 0.30,
        "onehot_feat7": 0.20,
        "upload_type": 0.20,
        "music_type": 0.15,
    }

    for field in BACKOFF_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        table = tables[field]
        reliability = reliabilities[field]
        safe_ids = np.minimum(ids, table.size - 1)
        score += (
            field_scale[field]
            * table[safe_ids].astype(np.float64)
            * reliability[safe_ids].astype(np.float64)
        )
    return score


def predict_bradley(model, split):
    arrays = {
        field: np.asarray(split.X[field], dtype=np.int64)
        for field in PAIR_FIELDS
    }
    return model.score_matrix(arrays)


train = load("train")
valid = load("valid")

train_labels = np.asarray(train.y, dtype=np.int8)
valid_labels = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_weights = recency_weights(train.date, half_life=4.0)
positive_pairs, negative_pairs = make_pair_indices(
    train, train_labels
)
print(
    "FINDINGS conditional_pairs=%d positive_row_weight_mean=%.5f"
    % (
        positive_pairs.size,
        float(np.mean(train_weights[positive_pairs])),
    )
)

bradley_model = fit_bradley_terry(
    train,
    train_labels,
    positive_pairs,
    negative_pairs,
    train_weights,
)
bradley_valid = predict_bradley(bradley_model, valid)

backoff_tables, backoff_reliabilities, _ = fit_backoff_tables(
    train, train_labels, train_weights
)
backoff_valid = predict_backoff(
    valid, backoff_tables, backoff_reliabilities
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)

candidate_scores = {}
candidate_metrics = {}
candidate_specs = {}
candidate_raw = {}


def register(name, scores, spec, raw_scores):
    scores = np.asarray(scores, dtype=np.float64)
    result = evaluate(valid_users, valid_labels, scores)
    candidate_scores[name] = scores
    candidate_metrics[name] = result
    candidate_specs[name] = spec
    candidate_raw[name] = np.asarray(raw_scores, dtype=np.float64)


register(
    "incumbent",
    incumbent_valid,
    ("incumbent",),
    bradley_valid,
)
register(
    "conditional_bradley_terry",
    bradley_valid,
    ("bradley",),
    bradley_valid,
)
register(
    "hierarchical_content_backoff",
    backoff_valid,
    ("backoff",),
    backoff_valid,
)

own_ensemble_valid = rank_blend(
    valid_users, bradley_valid, backoff_valid, 0.35
)
register(
    "own_family_ensemble",
    own_ensemble_valid,
    ("own_ensemble", 0.35),
    own_ensemble_valid,
)

for family_name, family_scores in (
    ("bradley", bradley_valid),
    ("backoff", backoff_valid),
    ("own_ensemble", own_ensemble_valid),
):
    for alpha in (0.05, 0.10, 0.20, 0.35, 0.50, 0.70):
        blended = rank_blend(
            valid_users, incumbent_valid, family_scores, alpha
        )
        name = "%s_incumbent_blend_%.2f" % (family_name, alpha)
        register(
            name,
            blended,
            ("incumbent_blend", family_name, alpha),
            family_scores,
        )

winner = max(
    candidate_metrics,
    key=lambda key: candidate_metrics[key]["primary"],
)
winner_spec = candidate_specs[winner]
valid_scores = candidate_scores[winner]
metrics = candidate_metrics[winner]

compact = {
    name: round(float(value["primary"]), 6)
    for name, value in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(compact, sort_keys=True))
print(
    "FINDINGS winner=%s spec=%s bradley=%.6f backoff=%.6f ensemble=%.6f incumbent=%.6f"
    % (
        winner,
        repr(winner_spec),
        candidate_metrics["conditional_bradley_terry"]["primary"],
        candidate_metrics["hierarchical_content_backoff"]["primary"],
        candidate_metrics["own_family_ensemble"]["primary"],
        candidate_metrics["incumbent"]["primary"],
    )
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(candidate_raw[winner], dtype=np.float64),
    )

# The winning validation specification is now fixed. Test labels are never
# read, and every fitted parameter above used only the train split.
test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)
bradley_test = predict_bradley(bradley_model, test)
backoff_test = predict_backoff(
    test, backoff_tables, backoff_reliabilities
)
incumbent_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)
own_ensemble_test = rank_blend(
    test_users, bradley_test, backoff_test, 0.35
)

if winner_spec[0] == "incumbent":
    test_scores = incumbent_test
elif winner_spec[0] == "bradley":
    test_scores = bradley_test
elif winner_spec[0] == "backoff":
    test_scores = backoff_test
elif winner_spec[0] == "own_ensemble":
    test_scores = own_ensemble_test
elif winner_spec[0] == "incumbent_blend":
    family_name = winner_spec[1]
    alpha = float(winner_spec[2])
    if family_name == "bradley":
        family_test = bradley_test
    elif family_name == "backoff":
        family_test = backoff_test
    else:
        family_test = own_ensemble_test
    test_scores = rank_blend(
        test_users, incumbent_test, family_test, alpha
    )
else:
    raise RuntimeError("Unknown winning specification: %r" % (winner_spec,))

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)