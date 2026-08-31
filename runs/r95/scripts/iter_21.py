import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 8))

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)
ute = np.asarray(test.user_id, dtype=np.int64)

# The propensity model asks whether an observed user/context-item pairing looks
# like a pairing produced by the logging policy rather than a shuffled pairing.
CONTEXT_FIELDS = [
    "user_id",
    "tab",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "is_live_streamer",
]
ITEM_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]
PROP_FIELDS = CONTEXT_FIELDS + ITEM_FIELDS

OUTCOME_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "music_type",
    "video_type",
    "is_live_streamer",
    "is_video_author",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(scores), dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts

    positions = (
        np.arange(len(order), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    sorted_ranks = positions / denominators

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = sorted_ranks
    return result


# -------------------------------------------------------------------------
# Train-only temporal weights.
# -------------------------------------------------------------------------
dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(dates)
date_lookup = {int(d): i for i, d in enumerate(unique_dates)}
day_index = np.fromiter(
    (date_lookup[int(d)] for d in dates),
    dtype=np.int16,
    count=len(dates),
)
age = (len(unique_dates) - 1 - day_index).astype(np.float32)
recency_weight = np.exp2(-age / 4.0).astype(np.float32)
recency_weight /= recency_weight.mean()


# -------------------------------------------------------------------------
# Exposure propensity model.
#
# Positive examples are the logged context-item pairings. Negative examples
# retain every logged context but attach an item-side feature block from a
# random different row. The classifier therefore learns logging-policy
# support using train features only and never touches auxiliary outcomes.
# -------------------------------------------------------------------------
def categorical_matrix(sample, fields):
    return np.column_stack([
        np.asarray(sample.X[field], dtype=np.float32)
        for field in fields
    ]).astype(np.float32, copy=False)


prop_real = categorical_matrix(train, PROP_FIELDS)
n_train = len(ytr)

rng = np.random.default_rng(SEED)
permutation = rng.permutation(n_train)
prop_fake = prop_real.copy()
item_start = len(CONTEXT_FIELDS)
prop_fake[:, item_start:] = prop_real[permutation, item_start:]

prop_x = np.concatenate([prop_real, prop_fake], axis=0)
prop_y = np.concatenate([
    np.ones(n_train, dtype=np.float32),
    np.zeros(n_train, dtype=np.float32),
])
prop_dset = lgb.Dataset(
    prop_x,
    label=prop_y,
    categorical_feature=list(range(len(PROP_FIELDS))),
    free_raw_data=False,
)

prop_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": 7,
    "min_data_in_leaf": 1200,
    "lambda_l2": 5.0,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 30.0,
    "verbosity": -1,
    "num_threads": min(8, os.cpu_count() or 8),
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
}

prop_model = lgb.train(
    prop_params,
    prop_dset,
    num_boost_round=110,
)

prop_train = np.clip(
    prop_model.predict(prop_real),
    0.05,
    0.98,
).astype(np.float64)

prop_valid_x = categorical_matrix(valid, PROP_FIELDS)
prop_test_x = categorical_matrix(test, PROP_FIELDS)
prop_valid = np.clip(
    prop_model.predict(prop_valid_x),
    0.05,
    0.98,
).astype(np.float64)
prop_test = np.clip(
    prop_model.predict(prop_test_x),
    0.05,
    0.98,
).astype(np.float64)

# Tempered inverse odds are less variable than raw inverse propensity. The
# clipping prevents a tiny set of synthetic-looking rows dominating training.
inverse_exposure = np.sqrt((1.0 - prop_train) / prop_train)
inverse_exposure /= max(float(np.mean(inverse_exposure)), 1.0e-8)
inverse_exposure = np.clip(inverse_exposure, 0.25, 4.0)

outcome_weight = (
    recency_weight.astype(np.float64) * inverse_exposure
)
outcome_weight /= max(float(outcome_weight.mean()), 1.0e-8)
outcome_weight = np.clip(outcome_weight, 0.10, 6.0).astype(np.float32)
outcome_weight /= outcome_weight.mean()

ess = (
    float(outcome_weight.sum()) ** 2
    / float(np.square(outcome_weight).sum())
)
print(
    "FINDINGS propensity_mean=%.6f propensity_q10=%.6f "
    "propensity_q90=%.6f weight_q10=%.6f weight_q90=%.6f "
    "effective_sample_fraction=%.6f"
    % (
        float(prop_train.mean()),
        float(np.quantile(prop_train, 0.10)),
        float(np.quantile(prop_train, 0.90)),
        float(np.quantile(outcome_weight, 0.10)),
        float(np.quantile(outcome_weight, 0.90)),
        ess / n_train,
    )
)

del prop_x, prop_y, prop_fake, prop_dset, prop_valid_x, prop_test_x
del prop_model
gc.collect()


# -------------------------------------------------------------------------
# Family 1: inverse-propensity-weighted categorical GBDT.
# -------------------------------------------------------------------------
num_location = {}
num_scale = {}
for field in NUM_FIELDS:
    raw = np.asarray(train.num[field], dtype=np.float64)
    transformed = np.log1p(
        np.maximum(np.nan_to_num(raw, nan=0.0), 0.0)
    )
    median = float(np.median(transformed))
    q25, q75 = np.quantile(transformed, [0.25, 0.75])
    num_location[field] = median
    num_scale[field] = max(float(q75 - q25), 1.0e-3)


def outcome_matrix(sample):
    columns = [
        np.asarray(sample.X[field], dtype=np.float32)
        for field in OUTCOME_FIELDS
    ]
    for field in NUM_FIELDS:
        raw = np.asarray(sample.num[field], dtype=np.float64)
        values = np.log1p(
            np.maximum(np.nan_to_num(raw, nan=0.0), 0.0)
        )
        values = (
            (values - num_location[field]) / num_scale[field]
        )
        columns.append(
            np.clip(values, -8.0, 8.0).astype(np.float32)
        )
    return np.column_stack(columns).astype(np.float32, copy=False)


xtr = outcome_matrix(train)
xva = outcome_matrix(valid)
xte = outcome_matrix(test)

dtrain = lgb.Dataset(
    xtr,
    label=ytr,
    weight=outcome_weight,
    categorical_feature=list(range(len(OUTCOME_FIELDS))),
    free_raw_data=False,
)

gbdt_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.045,
    "num_leaves": 31,
    "max_depth": 8,
    "min_data_in_leaf": 900,
    "lambda_l2": 4.0,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 25.0,
    "verbosity": -1,
    "num_threads": min(8, os.cpu_count() or 8),
    "seed": SEED + 10,
    "feature_fraction_seed": SEED + 11,
    "bagging_seed": SEED + 12,
}

gbdt = lgb.train(
    gbdt_params,
    dtrain,
    num_boost_round=260,
)
gbdt_valid = gbdt.predict(xva).astype(np.float64)
gbdt_test = gbdt.predict(xte).astype(np.float64)

del gbdt, dtrain, xtr, xva, xte
gc.collect()


# -------------------------------------------------------------------------
# Family 2: inverse-propensity-weighted factorization machine.
#
# This forms predictions through global pairwise latent interactions rather
# than tree partitions. Propensity weighting changes which logged edges drive
# the interaction geometry.
# -------------------------------------------------------------------------
FM_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat8",
]

offsets = []
running = 0
for field in FM_FIELDS:
    offsets.append(running)
    running += int(FEATURE_CARDINALITIES[field])
total_cardinality = running
offset_array = np.asarray(offsets, dtype=np.int64)


def fm_matrix(sample):
    result = np.column_stack([
        np.asarray(sample.X[field], dtype=np.int64)
        for field in FM_FIELDS
    ])
    result += offset_array[None, :]
    return result


fm_xtr = fm_matrix(train)
fm_xva = fm_matrix(valid)
fm_xte = fm_matrix(test)


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, dimension):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, dimension)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, indices):
        linear_term = self.linear(indices).sum(dim=1).squeeze(-1)
        embeddings = self.embedding(indices)
        summed = embeddings.sum(dim=1)
        interactions = 0.5 * (
            summed.square().sum(dim=1)
            - embeddings.square().sum(dim=(1, 2))
        )
        return self.bias + linear_term + interactions


fm_model = FactorizationMachine(total_cardinality, 12)
optimizer = torch.optim.AdamW(
    fm_model.parameters(),
    lr=0.003,
    weight_decay=2.0e-6,
)

batch_size = 16384
train_indices = np.arange(n_train, dtype=np.int64)
fm_model.train()

for epoch in range(3):
    rng.shuffle(train_indices)
    epoch_loss = 0.0
    epoch_weight = 0.0

    for begin in range(0, n_train, batch_size):
        rows = train_indices[begin:begin + batch_size]
        xb = torch.from_numpy(fm_xtr[rows])
        yb = torch.from_numpy(ytr[rows])
        wb = torch.from_numpy(outcome_weight[rows])

        optimizer.zero_grad(set_to_none=True)
        logits = fm_model(xb)
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits,
            yb,
            reduction="none",
        )
        loss = (losses * wb).sum() / wb.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm_model.parameters(), 5.0)
        optimizer.step()

        epoch_loss += float((losses.detach() * wb).sum())
        epoch_weight += float(wb.sum())

    print(
        "FINDINGS fm_epoch=%d weighted_logloss=%.6f"
        % (epoch + 1, epoch_loss / max(epoch_weight, 1.0))
    )


def fm_predict(matrix):
    result = np.empty(len(matrix), dtype=np.float64)
    fm_model.eval()
    with torch.no_grad():
        for begin in range(0, len(matrix), 32768):
            end = min(begin + 32768, len(matrix))
            xb = torch.from_numpy(matrix[begin:end])
            result[begin:end] = (
                fm_model(xb).cpu().numpy().astype(np.float64)
            )
    return result


fm_valid = fm_predict(fm_xva)
fm_test = fm_predict(fm_xte)

del fm_model, optimizer, fm_xtr, fm_xva, fm_xte
gc.collect()


# -------------------------------------------------------------------------
# Family 3: inverse-propensity-weighted empirical-Bayes likelihood ratios.
#
# This pools class-conditional evidence independently per field and crossed
# field, providing a low-variance non-parametric contrast to GBDT and FM.
# -------------------------------------------------------------------------
NB_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "music_type",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]
NB_CROSSES = [
    ("tab", "tag"),
    ("duration_bucket", "tag"),
    ("author_id", "tab"),
    ("user_active_degree", "duration_bucket"),
    ("upload_type", "duration_bucket"),
]


def nb_arrays(sample):
    arrays = []
    cards = []

    for field in NB_FIELDS:
        arrays.append(
            np.asarray(sample.X[field], dtype=np.int64)
        )
        cards.append(int(FEATURE_CARDINALITIES[field]))

    for left, right in NB_CROSSES:
        right_card = int(FEATURE_CARDINALITIES[right])
        values = (
            np.asarray(sample.X[left], dtype=np.int64) * right_card
            + np.asarray(sample.X[right], dtype=np.int64)
        )
        arrays.append(values)
        cards.append(
            int(FEATURE_CARDINALITIES[left]) * right_card
        )

    return arrays, cards


tr_nb, nb_cards = nb_arrays(train)
va_nb, _ = nb_arrays(valid)
te_nb, _ = nb_arrays(test)

positive_weight = outcome_weight.astype(np.float64) * ytr
negative_weight = outcome_weight.astype(np.float64) * (1.0 - ytr)
total_positive = float(positive_weight.sum())
total_negative = float(negative_weight.sum())

nb_tables = []
smoothing = 35.0

for values, cardinality in zip(tr_nb, nb_cards):
    c1 = np.bincount(
        values,
        weights=positive_weight,
        minlength=cardinality,
    ).astype(np.float64)
    c0 = np.bincount(
        values,
        weights=negative_weight,
        minlength=cardinality,
    ).astype(np.float64)

    p1 = (c1 + smoothing) / (
        total_positive + smoothing * cardinality
    )
    p0 = (c0 + smoothing) / (
        total_negative + smoothing * cardinality
    )
    nb_tables.append(np.log(p1) - np.log(p0))


def nb_predict(arrays):
    result = np.zeros(len(arrays[0]), dtype=np.float64)
    for values, table in zip(arrays, nb_tables):
        result += table[values]
    return result


nb_valid = nb_predict(va_nb)
nb_test = nb_predict(te_nb)

del tr_nb, va_nb, te_nb, nb_tables
gc.collect()


# -------------------------------------------------------------------------
# Rank aggregation and trusted-incumbent blending.
# -------------------------------------------------------------------------
families_valid = {
    "ips_recency_gbdt": within_user_rank(uva, gbdt_valid),
    "ips_recency_fm": within_user_rank(uva, fm_valid),
    "ips_recency_empirical_bayes": within_user_rank(uva, nb_valid),
}
families_test = {
    "ips_recency_gbdt": within_user_rank(ute, gbdt_test),
    "ips_recency_fm": within_user_rank(ute, fm_test),
    "ips_recency_empirical_bayes": within_user_rank(ute, nb_test),
}

# Cross-family aggregates test whether local partitions, latent interactions,
# and pooled marginal evidence make complementary errors.
families_valid["ips_gbdt_fm"] = (
    families_valid["ips_recency_gbdt"]
    + families_valid["ips_recency_fm"]
) / 2.0
families_test["ips_gbdt_fm"] = (
    families_test["ips_recency_gbdt"]
    + families_test["ips_recency_fm"]
) / 2.0

families_valid["ips_all_family_ensemble"] = (
    families_valid["ips_recency_gbdt"]
    + families_valid["ips_recency_fm"]
    + families_valid["ips_recency_empirical_bayes"]
) / 3.0
families_test["ips_all_family_ensemble"] = (
    families_test["ips_recency_gbdt"]
    + families_test["ips_recency_fm"]
    + families_test["ips_recency_empirical_bayes"]
) / 3.0

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared_dir, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared_dir, "incumbent_test_scores.npy")),
    dtype=np.float64,
)
inc_rank_valid = within_user_rank(uva, inc_valid)
inc_rank_test = within_user_rank(ute, inc_test)

alphas = [0.0, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50, 0.75, 1.0]

candidate_scores = {}
best_primary = -np.inf
best_metrics = None
best_valid = None
best_test = None
best_raw = None
best_name = None

for name, own_valid in families_valid.items():
    own_test = families_test[name]
    standalone = evaluate(uva, yva, own_valid)
    candidate_scores[name + "_standalone"] = float(
        standalone["primary"]
    )

    correlation = float(
        np.corrcoef(inc_rank_valid, own_valid)[0, 1]
    )
    print(
        "FINDINGS family=%s incumbent_rank_correlation=%.6f "
        "standalone_primary=%.6f"
        % (
            name,
            correlation,
            float(standalone["primary"]),
        )
    )

    for alpha in alphas:
        blend_valid = (
            (1.0 - alpha) * inc_rank_valid
            + alpha * own_valid
        )
        blend_test = (
            (1.0 - alpha) * inc_rank_test
            + alpha * own_test
        )
        metrics = evaluate(uva, yva, blend_valid)
        primary = float(metrics["primary"])
        candidate_name = "%s_blend_%.2f" % (name, alpha)
        candidate_scores[candidate_name] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_valid = blend_valid.copy()
            best_test = blend_test.copy()
            best_raw = own_valid.copy()
            best_name = candidate_name

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s primary=%.6f"
    % (best_name, best_primary)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)