import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 28471
BATCH = 8192

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
train_dates = np.asarray(train.date, dtype=np.int64)
max_train_date = int(train_dates.max())

# Dates are consecutive inside this split, but YYYYMMDD subtraction is not
# generally a valid day difference. Mapping through the observed train dates
# avoids relying on that representation.
unique_train_dates = np.sort(np.unique(train_dates))
date_to_index = {
    int(date): index for index, date in enumerate(unique_train_dates)
}
train_day = np.asarray(
    [date_to_index[int(date)] for date in train_dates],
    dtype=np.int64,
)
last_train_day = int(train_day.max())

recency_weight = np.exp2(
    (train_day.astype(np.float32) - last_train_day) / 4.0
)
recency_weight /= recency_weight.mean()
recency_weight = recency_weight.astype(np.float32)


def calendar_day_number(yyyymmdd):
    values = np.asarray(yyyymmdd, dtype=np.int64)
    text = values.astype("U8")
    return np.asarray(
        [
            np.datetime64(
                value[:4] + "-" + value[4:6] + "-" + value[6:8],
                "D",
            ).astype(np.int64)
            for value in text
        ],
        dtype=np.int64,
    )


train_calendar = calendar_day_number(train.date)
last_train_calendar = int(train_calendar.max())
train_time_offset = (
    train_calendar - last_train_calendar
).astype(np.float64)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort(
        (
            np.arange(n, dtype=np.int64),
            scores,
            user_ids,
        )
    )
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.int64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked[multi] = (
        positions[multi] / (repeated_lengths[multi] - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


# ---------------------------------------------------------------------
# Family 1: domain-adversarial neural predictor.
#
# The label tower learns long_view with a four-day recency weight. A second
# head tries to recover the training day from the shared representation.
# The shared encoder receives the reversed domain gradient, discouraging
# representations that exploit features useful only for identifying a
# particular training date.
# ---------------------------------------------------------------------
DANN_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "user_active_degree",
    "register_days_bucket",
    "hour",
    "onehot_feat3",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

dann_cards = [
    int(FEATURE_CARDINALITIES[field]) for field in DANN_FIELDS
]


def make_categorical_matrix(split):
    return np.ascontiguousarray(
        np.stack(
            [
                np.asarray(split.X[field], dtype=np.int64)
                for field in DANN_FIELDS
            ],
            axis=1,
        )
    )


def raw_numeric_matrix(split):
    columns = []
    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(value, 0.0)))
    return np.stack(columns, axis=1).astype(np.float32)


xcat_train = make_categorical_matrix(train)
xcat_valid = make_categorical_matrix(valid)
xcat_test = make_categorical_matrix(test)

xnum_train = raw_numeric_matrix(train)
num_mean = xnum_train.mean(axis=0, dtype=np.float64).astype(np.float32)
num_std = xnum_train.std(axis=0, dtype=np.float64).astype(np.float32)
num_std = np.maximum(num_std, 1e-3)

xnum_train = np.ascontiguousarray(
    (xnum_train - num_mean) / num_std
)
xnum_valid = np.ascontiguousarray(
    (raw_numeric_matrix(valid) - num_mean) / num_std
)
xnum_test = np.ascontiguousarray(
    (raw_numeric_matrix(test) - num_mean) / num_std
)


class DomainAdversarialNet(nn.Module):
    def __init__(self, cards, n_days):
        super().__init__()
        self.embeddings = nn.ModuleList()
        embedding_dims = []

        for card in cards:
            dim = 10 if card >= 1000 else 6
            embedding_dims.append(dim)
            embedding = nn.Embedding(card, dim)
            nn.init.normal_(embedding.weight, std=0.02)
            self.embeddings.append(embedding)

        input_dim = int(sum(embedding_dims) + len(NUM_FIELDS))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.label_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(64, 48),
            nn.SiLU(),
            nn.Linear(48, n_days),
        )

    def encode(self, categorical, numeric):
        parts = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        parts.append(numeric)
        return self.encoder(torch.cat(parts, dim=1))

    def label_logits(self, representation):
        return self.label_head(representation).squeeze(1)

    def domain_logits(self, representation):
        return self.domain_head(representation)


dann_model = DomainAdversarialNet(
    dann_cards, len(unique_train_dates)
)

encoder_parameters = (
    list(dann_model.embeddings.parameters())
    + list(dann_model.encoder.parameters())
    + list(dann_model.label_head.parameters())
)
label_optimizer = torch.optim.AdamW(
    encoder_parameters,
    lr=0.0015,
    weight_decay=2e-6,
)
domain_optimizer = torch.optim.AdamW(
    dann_model.domain_head.parameters(),
    lr=0.0015,
    weight_decay=2e-6,
)

rng = np.random.default_rng(SEED)
adversarial_strength = 0.06

for epoch in range(3):
    order = rng.permutation(len(ytr))
    dann_model.train()

    for lo in range(0, len(order), BATCH):
        idx = order[lo:lo + BATCH]

        categorical = torch.from_numpy(xcat_train[idx])
        numeric = torch.from_numpy(xnum_train[idx])
        target = torch.from_numpy(ytr[idx])
        weight = torch.from_numpy(recency_weight[idx])
        domain_target = torch.from_numpy(train_day[idx])

        # First train the domain classifier against a detached encoder.
        with torch.no_grad():
            detached_representation = dann_model.encode(
                categorical, numeric
            )
        domain_optimizer.zero_grad(set_to_none=True)
        domain_loss = F.cross_entropy(
            dann_model.domain_logits(detached_representation),
            domain_target,
        )
        domain_loss.backward()
        domain_optimizer.step()

        # Then update the encoder to predict the label while confusing the
        # now-trained domain head. The domain head is not in this optimizer.
        label_optimizer.zero_grad(set_to_none=True)
        representation = dann_model.encode(categorical, numeric)
        logits = dann_model.label_logits(representation)
        row_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        supervised_loss = (row_loss * weight).sum() / weight.sum()

        adversarial_loss = F.cross_entropy(
            dann_model.domain_logits(representation),
            domain_target,
        )
        total_loss = (
            supervised_loss
            - adversarial_strength * adversarial_loss
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder_parameters, 5.0)
        label_optimizer.step()


@torch.inference_mode()
def predict_dann(categorical, numeric):
    dann_model.eval()
    result = np.empty(len(categorical), dtype=np.float64)

    for lo in range(0, len(categorical), 32768):
        hi = min(lo + 32768, len(categorical))
        representation = dann_model.encode(
            torch.from_numpy(categorical[lo:hi]),
            torch.from_numpy(numeric[lo:hi]),
        )
        logits = dann_model.label_logits(representation)
        result[lo:hi] = logits.numpy().astype(np.float64)

    return result


dann_valid = predict_dann(xcat_valid, xnum_valid)
dann_test = predict_dann(xcat_test, xnum_test)

# Measure whether the adversary learned a nontrivial date discriminator.
probe_idx = np.linspace(
    0, len(xcat_train) - 1, min(50000, len(xcat_train)),
    dtype=np.int64,
)
with torch.inference_mode():
    probe_representation = dann_model.encode(
        torch.from_numpy(xcat_train[probe_idx]),
        torch.from_numpy(xnum_train[probe_idx]),
    )
    probe_prediction = dann_model.domain_logits(
        probe_representation
    ).argmax(dim=1).numpy()
domain_accuracy = float(
    np.mean(probe_prediction == train_day[probe_idx])
)

del xcat_train, xcat_valid, xcat_test
del xnum_train, xnum_valid, xnum_test
del dann_model, label_optimizer, domain_optimizer


# ---------------------------------------------------------------------
# Family 2: recency-weighted randomized tree ensemble.
#
# Random-forest LightGBM forms predictions by averaging independently
# bagged, feature-subsampled trees rather than by sequentially correcting
# residuals. It can model hard content/context partitions while bagging
# limits reliance on a few unstable identity splits.
# ---------------------------------------------------------------------
TREE_CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "user_active_degree",
    "register_days_bucket",
    "hour",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
]


def make_tree_matrix(split):
    columns = [
        np.asarray(split.X[field], dtype=np.float32)
        for field in TREE_CAT_FIELDS
    ]

    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(value, 0.0)))

    return np.ascontiguousarray(np.stack(columns, axis=1))


xtree_train = make_tree_matrix(train)
xtree_valid = make_tree_matrix(valid)
xtree_test = make_tree_matrix(test)

tree_dataset = lgb.Dataset(
    xtree_train,
    label=ytr,
    weight=recency_weight,
    categorical_feature=list(range(len(TREE_CAT_FIELDS))),
    free_raw_data=True,
)

tree_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "rf",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "max_depth": 10,
    "min_data_in_leaf": 120,
    "feature_fraction": 0.72,
    "bagging_fraction": 0.68,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "seed": SEED + 17,
    "feature_fraction_seed": SEED + 18,
    "bagging_seed": SEED + 19,
    "num_threads": min(8, os.cpu_count() or 1),
    "verbose": -1,
}

tree_model = lgb.train(
    tree_params,
    tree_dataset,
    num_boost_round=180,
)

tree_valid = tree_model.predict(
    xtree_valid, num_iteration=tree_model.current_iteration()
).astype(np.float64)
tree_test = tree_model.predict(
    xtree_test, num_iteration=tree_model.current_iteration()
).astype(np.float64)

del xtree_train, xtree_valid, xtree_test
del tree_dataset, tree_model


# ---------------------------------------------------------------------
# Family 3: time-varying empirical-Bayes extrapolator.
#
# For each entity, fit a recency-weighted local linear trend in long-view
# probability. Sparse means and slopes shrink strongly to stable priors.
# Predictions are projected to each evaluation row's actual calendar day,
# directly representing drift instead of assuming a stationary entity rate.
# ---------------------------------------------------------------------
DYNAMIC_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
]

global_rate = float(
    np.sum(recency_weight * ytr) / np.sum(recency_weight)
)
dynamic_parameters = {}

for field in DYNAMIC_FIELDS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])
    w = recency_weight.astype(np.float64)
    t = train_time_offset
    target = ytr.astype(np.float64)

    sw = np.bincount(ids, weights=w, minlength=card)
    sy = np.bincount(ids, weights=w * target, minlength=card)
    st = np.bincount(ids, weights=w * t, minlength=card)
    stt = np.bincount(ids, weights=w * t * t, minlength=card)
    sty = np.bincount(ids, weights=w * t * target, minlength=card)

    prior_strength = 30.0
    mean_rate = (
        sy + prior_strength * global_rate
    ) / np.maximum(sw + prior_strength, 1e-12)

    safe_sw = np.maximum(sw, 1e-12)
    centered_covariance = sty - st * sy / safe_sw
    centered_variance = stt - st * st / safe_sw

    # The ridge term is in weighted observation units and makes trend
    # extrapolation conservative for rare entities.
    slope = centered_covariance / (
        centered_variance + 6.0 * sw + 20.0
    )
    slope *= sw / (sw + 100.0)
    slope = np.clip(slope, -0.012, 0.012)

    dynamic_parameters[field] = (
        mean_rate.astype(np.float64),
        slope.astype(np.float64),
    )


def predict_dynamic(split):
    future_offset = (
        calendar_day_number(split.date) - last_train_calendar
    ).astype(np.float64)

    evidence = np.zeros(len(split.user_id), dtype=np.float64)
    total_weight = 0.0

    field_weights = {
        "video_id": 1.25,
        "author_id": 1.10,
        "tag": 0.75,
        "tab": 0.65,
        "duration_bucket": 0.55,
        "upload_type": 0.45,
    }

    for field in DYNAMIC_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        mean_rate, slope = dynamic_parameters[field]
        projected = mean_rate[ids] + slope[ids] * future_offset
        projected = np.clip(projected, 0.015, 0.985)
        logit = np.log(projected) - np.log1p(-projected)

        weight = field_weights[field]
        evidence += weight * logit
        total_weight += weight

    return evidence / total_weight


dynamic_valid = predict_dynamic(valid)
dynamic_test = predict_dynamic(test)


# ---------------------------------------------------------------------
# Compare raw models and rank-space blends with the trusted incumbent.
# Rank-space aggregation prevents arbitrary score calibration from deciding
# the blend and preserves the within-user nature of both metrics.
# ---------------------------------------------------------------------
shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

families = {
    "domain_adversarial_neural": (dann_valid, dann_test),
    "recency_random_forest": (tree_valid, tree_test),
    "temporal_bayes_extrapolator": (
        dynamic_valid, dynamic_test
    ),
}

candidate_primary = {}
candidate_valid = {}
candidate_test = {}
candidate_own_raw = {}
candidate_is_blend = {}

inc_metrics = evaluate(valid.user_id, valid.y, inc_valid_rank)
candidate_primary["trusted_incumbent"] = float(
    inc_metrics["primary"]
)
candidate_valid["trusted_incumbent"] = inc_valid_rank
candidate_test["trusted_incumbent"] = inc_test_rank
candidate_own_raw["trusted_incumbent"] = None
candidate_is_blend["trusted_incumbent"] = False

blend_alphas = [0.15, 0.30, 0.50, 0.70]

for family_name, (raw_valid, raw_test) in families.items():
    own_valid_rank = within_user_rank(valid.user_id, raw_valid)
    own_test_rank = within_user_rank(test.user_id, raw_test)

    raw_metrics = evaluate(
        valid.user_id, valid.y, own_valid_rank
    )
    raw_name = family_name + "_raw"
    candidate_primary[raw_name] = float(raw_metrics["primary"])
    candidate_valid[raw_name] = own_valid_rank
    candidate_test[raw_name] = own_test_rank
    candidate_own_raw[raw_name] = raw_valid
    candidate_is_blend[raw_name] = False

    for alpha in blend_alphas:
        name = family_name + "_blend_" + str(alpha)
        blended_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * own_valid_rank
        )
        blended_test = (
            (1.0 - alpha) * inc_test_rank
            + alpha * own_test_rank
        )

        metrics = evaluate(
            valid.user_id, valid.y, blended_valid
        )
        candidate_primary[name] = float(metrics["primary"])
        candidate_valid[name] = blended_valid
        candidate_test[name] = blended_test
        candidate_own_raw[name] = raw_valid
        candidate_is_blend[name] = True


# Prefer a genuinely new candidate on an exact floating-point tie, while
# still selecting strictly by the public validation aggregate.
best_name = max(
    candidate_primary,
    key=lambda name: (
        candidate_primary[name],
        name != "trusted_incumbent",
    ),
)
best_valid = np.asarray(
    candidate_valid[best_name], dtype=np.float64
)
best_test = np.asarray(
    candidate_test[best_name], dtype=np.float64
)
best_metrics = evaluate(valid.user_id, valid.y, best_valid)

print(
    "FINDINGS "
    + json.dumps(
        {
            "domain_adversary_day_accuracy": domain_accuracy,
            "chance_day_accuracy": 1.0 / len(unique_train_dates),
            "selected": best_name,
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(candidate_primary, sort_keys=True)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        best_valid,
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        best_test,
    )

    if candidate_is_blend[best_name]:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(
                candidate_own_raw[best_name],
                dtype=np.float64,
            ),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)