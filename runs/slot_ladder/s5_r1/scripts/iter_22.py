import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
torch.manual_seed(2026)
np.random.seed(2026)

# Identity/content fields have extremely high train-to-validation overlap. User
# properties are represented by relatively stable coarse segments rather than
# user_id, reducing dependence on user activity volume across the date split.
ADDITIVE_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "tab",
    "hour",
    "user_active_degree",
    "register_days_range",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "is_live_streamer",
]

# Exact crosses implement a wide polynomial model. They let the same content
# receive different scores for user segments and feed contexts without relying
# on a high-capacity user embedding.
CROSS_FIELDS = [
    ("video_id", "user_active_degree"),
    ("author_id", "user_active_degree"),
    ("tag", "user_active_degree"),
    ("duration_bucket", "user_active_degree"),
    ("upload_type", "user_active_degree"),
    ("onehot_feat3", "user_active_degree"),
    ("tag", "tab"),
    ("duration_bucket", "tab"),
    ("upload_type", "tab"),
    ("onehot_feat3", "tab"),
    ("tag", "register_days_range"),
    ("duration_bucket", "register_days_range"),
    ("tag", "fans_user_num_range"),
    ("duration_bucket", "fans_user_num_range"),
    ("video_id", "tab"),
    ("author_id", "tab"),
]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    # Stable row-position tie breaking makes the transformation reproducible.
    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    sizes = np.diff(np.r_[starts, n])

    ranks = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


class WideEncoder:
    def __init__(self, use_crosses):
        self.use_crosses = bool(use_crosses)
        self.terms = []
        offset = 0

        for field in ADDITIVE_FIELDS:
            card = int(FEATURE_CARDINALITIES[field])
            self.terms.append(("single", field, None, card, offset))
            offset += card

        if self.use_crosses:
            for left, right in CROSS_FIELDS:
                left_card = int(FEATURE_CARDINALITIES[left])
                right_card = int(FEATURE_CARDINALITIES[right])
                card = left_card * right_card
                self.terms.append(
                    ("cross", left, right, right_card, offset)
                )
                offset += card

        self.cardinality = int(offset)

    def encode_batch(self, split, rows):
        columns = []
        for kind, left, right, multiplier, offset in self.terms:
            left_ids = np.asarray(split.X[left][rows], dtype=np.int64)

            if kind == "single":
                ids = left_ids
            else:
                right_ids = np.asarray(split.X[right][rows], dtype=np.int64)
                ids = left_ids * multiplier + right_ids

            columns.append(ids + offset)

        matrix = np.stack(columns, axis=1)
        return torch.from_numpy(matrix).long()


class SparseWideModel(nn.Module):
    def __init__(self, cardinality):
        super().__init__()
        self.weight = nn.Embedding(cardinality, 1, sparse=True)
        nn.init.zeros_(self.weight.weight)

    def forward(self, encoded):
        return self.weight(encoded).sum(dim=1).squeeze(-1)


def fit_wide(train, use_crosses, epochs=3, batch_size=65536):
    encoder = WideEncoder(use_crosses)
    model = SparseWideModel(encoder.cardinality)
    optimizer = torch.optim.SparseAdam(
        model.parameters(), lr=0.035, betas=(0.9, 0.995), eps=1e-8
    )

    y = np.asarray(train.y, dtype=np.float32)
    n = len(y)

    # Shuffle batches each epoch. The estimator remains uniformly weighted in
    # date, making the experiment distinct from the parallel recency sweeps.
    for epoch in range(epochs):
        rng = np.random.default_rng(9100 + epoch + 100 * int(use_crosses))
        permutation = rng.permutation(n)

        model.train()
        for start in range(0, n, batch_size):
            rows = permutation[start:start + batch_size]
            encoded = encoder.encode_batch(train, rows)
            target = torch.from_numpy(y[rows])

            optimizer.zero_grad(set_to_none=True)
            logits = model(encoded)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            optimizer.step()

    return encoder, model


def predict_wide(encoder, model, split, batch_size=131072):
    n = len(split.user_id)
    result = np.empty(n, dtype=np.float64)
    model.eval()

    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            rows = np.arange(start, stop, dtype=np.int64)
            encoded = encoder.encode_batch(split, rows)
            result[start:stop] = (
                model(encoded).cpu().numpy().astype(np.float64)
            )

    return result


train = load("train")
valid = load("valid")

# Same convex estimator and labels, but two structurally different prediction
# formations: additive evidence and additive plus exact contextual crosses.
add_encoder, add_model = fit_wide(train, use_crosses=False)
cross_encoder, cross_model = fit_wide(train, use_crosses=True)

valid_add_raw = predict_wide(add_encoder, add_model, valid)
valid_cross_raw = predict_wide(cross_encoder, cross_model, valid)

valid_own = {
    "wide_additive": within_user_rank(valid.user_id, valid_add_raw),
    "wide_exact_cross": within_user_rank(valid.user_id, valid_cross_raw),
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores are missing")
if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent test scores are missing")

inc_valid_raw = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid_raw) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")

inc_valid = within_user_rank(valid.user_id, inc_valid_raw)

candidate_scores = {"incumbent": inc_valid}
candidate_primary = {
    "incumbent": float(
        evaluate(valid.user_id, valid.y, inc_valid)["primary"]
    )
}
recipes = {"incumbent": ("incumbent", "", 0.0)}

# Validation is used only for public feedback and selection among complete
# train-only models, as explicitly permitted for trusted-incumbent blends.
blend_weights = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)

for family, own_score in valid_own.items():
    standalone = family + "_standalone"
    candidate_scores[standalone] = own_score
    candidate_primary[standalone] = float(
        evaluate(valid.user_id, valid.y, own_score)["primary"]
    )
    recipes[standalone] = ("standalone", family, 1.0)

    for weight in blend_weights:
        name = f"{family}_blend_{weight:.2f}"
        blended = (1.0 - weight) * inc_valid + weight * own_score
        candidate_scores[name] = blended
        candidate_primary[name] = float(
            evaluate(valid.user_id, valid.y, blended)["primary"]
        )
        recipes[name] = ("blend", family, weight)

winner = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)
recipe_type, winner_family, winner_weight = recipes[winner]

best_own_family = max(
    valid_own,
    key=lambda family: candidate_primary[family + "_standalone"],
)
raw_for_audit = valid_own[
    winner_family if winner_family in valid_own else best_own_family
]

# Quantify whether crosses actually changed order rather than merely calibration.
rank_difference = np.abs(
    valid_own["wide_exact_cross"] - valid_own["wide_additive"]
)

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "best_own_family": best_own_family,
            "incumbent_primary": candidate_primary["incumbent"],
            "additive_primary": candidate_primary[
                "wide_additive_standalone"
            ],
            "cross_primary": candidate_primary[
                "wide_exact_cross_standalone"
            ],
            "mean_abs_additive_cross_rank_change": float(
                rank_difference.mean()
            ),
            "additive_parameters": int(add_encoder.cardinality),
            "cross_parameters": int(cross_encoder.cardinality),
        },
        separators=(",", ":"),
    )
)

print(
    "CANDIDATES "
    + json.dumps(
        {name: float(value) for name, value in candidate_primary.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
)

test = load("test")
test_own = {
    "wide_additive": within_user_rank(
        test.user_id, predict_wide(add_encoder, add_model, test)
    ),
    "wide_exact_cross": within_user_rank(
        test.user_id, predict_wide(cross_encoder, cross_model, test)
    ),
}

inc_test_raw = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test_raw) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")
inc_test = within_user_rank(test.user_id, inc_test_raw)

if recipe_type == "incumbent":
    test_scores = inc_test
elif recipe_type == "standalone":
    test_scores = test_own[winner_family]
else:
    test_scores = (
        (1.0 - winner_weight) * inc_test
        + winner_weight * test_own[winner_family]
    )

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(raw_for_audit, dtype=np.float64),
    )
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
        },
        separators=(",", ":"),
    )
)