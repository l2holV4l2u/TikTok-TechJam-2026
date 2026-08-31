import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
THREADS = max(1, min(8, os.cpu_count() or 1))

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "user_active_degree",
    "fans_user_num_range",
    "register_days_bucket",
    "hour",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

RATE_SPECS = [
    ("video", ("video_id",), 18.0),
    ("author", ("author_id",), 22.0),
    ("tag", ("tag",), 45.0),
    ("duration", ("duration_bucket",), 55.0),
    ("onehot3", ("onehot_feat3",), 30.0),
    ("upload", ("upload_type",), 45.0),
    ("user_tag", ("user_id", "tag"), 12.0),
    ("user_duration", ("user_id", "duration_bucket"), 12.0),
    ("author_tag", ("author_id", "tag"), 15.0),
    ("video_tab", ("video_id", "tab"), 12.0),
]


def split_arrays(split):
    return {
        name: np.asarray(split.X[name], dtype=np.int64)
        for name in set(CAT_FIELDS + [
            f for _, fields, _ in RATE_SPECS for f in fields
        ])
    }


def concatenate_arrays(a, b):
    return {
        name: np.concatenate([a[name], b[name]])
        for name in a
    }


def encoded_key(arrays, fields):
    if len(fields) == 1:
        return arrays[fields[0]]

    key = arrays[fields[0]].astype(np.int64, copy=True)
    for field in fields[1:]:
        key = key * int(FEATURE_CARDINALITIES[field]) + arrays[field]
    return key


def smoothed_rate_features(fit_arrays, y_fit, target_arrays, leave_one_out):
    y_fit = np.asarray(y_fit, dtype=np.float64)
    prior = float(np.mean(y_fit))
    n_target = len(next(iter(target_arrays.values())))
    result = np.empty((n_target, len(RATE_SPECS) * 2), dtype=np.float32)

    for j, (_, fields, strength) in enumerate(RATE_SPECS):
        fit_key = encoded_key(fit_arrays, fields)
        target_key = encoded_key(target_arrays, fields)
        size = int(max(
            int(fit_key.max(initial=0)),
            int(target_key.max(initial=0))
        )) + 1

        counts = np.bincount(fit_key, minlength=size).astype(np.float64)
        sums = np.bincount(
            fit_key, weights=y_fit, minlength=size
        ).astype(np.float64)

        tc = counts[target_key]
        ts = sums[target_key]

        if leave_one_out:
            tc = tc - 1.0
            ts = ts - y_fit
            tc = np.maximum(tc, 0.0)

        rate = (ts + strength * prior) / (tc + strength)
        result[:, 2 * j] = rate.astype(np.float32)
        result[:, 2 * j + 1] = np.log1p(tc).astype(np.float32)

        del fit_key, target_key, counts, sums, tc, ts, rate

    return result


def make_base_matrix(split):
    n = len(split.user_id)
    x = np.empty(
        (n, len(CAT_FIELDS) + len(NUM_FIELDS)),
        dtype=np.float32
    )
    for j, field in enumerate(CAT_FIELDS):
        x[:, j] = np.asarray(split.X[field], dtype=np.float32)

    for j, field in enumerate(NUM_FIELDS):
        z = np.asarray(split.num[field], dtype=np.float32)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        x[:, len(CAT_FIELDS) + j] = np.log1p(
            np.maximum(z, 0.0)
        )
    return x


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int64)
    age = int(dates.max()) - dates
    w = np.exp2(-age.astype(np.float32) / float(half_life))
    return (w / np.mean(w)).astype(np.float32)


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
    group_index = np.cumsum(first) - 1
    local_rank = np.arange(n, dtype=np.int64) - starts[group_index]
    sizes = np.diff(np.r_[starts, n])

    ranked = (
        local_rank.astype(np.float64) + 0.5
    ) / sizes[group_index].astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def nonparametric_scores(rate_features, style):
    rates = np.clip(
        rate_features[:, 0::2].astype(np.float64),
        1e-4,
        1.0 - 1e-4,
    )
    logits = np.log(rates / (1.0 - rates))

    if style == "content":
        weights = np.array(
            [1.25, 1.10, 0.45, 0.50, 0.55, 0.25,
             0.00, 0.00, 0.45, 0.45],
            dtype=np.float64,
        )
    elif style == "personal":
        weights = np.array(
            [0.85, 0.75, 0.30, 0.35, 0.35, 0.20,
             1.15, 0.90, 0.50, 0.35],
            dtype=np.float64,
        )
    else:
        weights = np.array(
            [1.00, 0.90, 0.35, 0.40, 0.45, 0.20,
             0.65, 0.55, 0.50, 0.40],
            dtype=np.float64,
        )

    return (logits @ weights / weights.sum()).astype(np.float32)


def fit_gbdt(x, y, dates, rounds=240):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 120,
        "max_bin": 127,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.92,
        "bagging_freq": 1,
        "lambda_l1": 0.04,
        "lambda_l2": 1.8,
        "min_gain_to_split": 0.002,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": THREADS,
        "force_col_wise": True,
        "verbose": -1,
    }
    dset = lgb.Dataset(
        x,
        label=np.asarray(y, dtype=np.float32),
        weight=recency_weights(dates, 4.0),
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    return lgb.train(params, dset, num_boost_round=rounds)


class DCN(nn.Module):
    def __init__(self, num_numeric, embedding_dim=8):
        super().__init__()
        cardinalities = [
            int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS
        ]
        offsets = np.cumsum([0] + cardinalities[:-1]).astype(np.int64)
        self.register_buffer(
            "offsets", torch.from_numpy(offsets)
        )
        self.embedding = nn.Embedding(sum(cardinalities), embedding_dim)

        dim = len(CAT_FIELDS) * embedding_dim + num_numeric
        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(dim)) for _ in range(2)
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for _ in range(2)
        ])
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 64, 1)

        nn.init.normal_(self.embedding.weight, std=0.025)
        for w in self.cross_w:
            nn.init.normal_(w, std=0.015)

    def forward(self, cats, numeric):
        indices = cats + self.offsets
        emb = self.embedding(indices).flatten(1)
        x0 = torch.cat([emb, numeric], dim=1)
        x = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = torch.sum(x * w, dim=1, keepdim=True)
            x = x0 * scalar + b + x
        deep = self.deep(x0)
        return self.output(torch.cat([x, deep], dim=1)).squeeze(1)


def cat_matrix(arrays):
    return np.stack(
        [arrays[f] for f in CAT_FIELDS], axis=1
    ).astype(np.int64, copy=False)


def normalize_numeric(train_num, target_num=None):
    mean = train_num.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_num.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-3)

    train_scaled = np.clip(
        (train_num - mean) / std, -6.0, 6.0
    ).astype(np.float32)

    if target_num is None:
        return train_scaled, mean, std

    target_scaled = np.clip(
        (target_num - mean) / std, -6.0, 6.0
    ).astype(np.float32)
    return train_scaled, target_scaled, mean, std


def fit_dcn(cats, numeric, y, dates, epochs=2):
    torch.manual_seed(SEED + 20)
    rng = np.random.default_rng(SEED + 21)

    model = DCN(numeric.shape[1], embedding_dim=8)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=2e-6
    )

    y = np.asarray(y, dtype=np.float32)
    weights = recency_weights(dates, 4.0)
    batch_size = 8192

    model.train()
    for _ in range(epochs):
        permutation = rng.permutation(len(y))
        for begin in range(0, len(y), batch_size):
            idx = permutation[begin:begin + batch_size]
            tc = torch.from_numpy(cats[idx])
            tn = torch.from_numpy(numeric[idx])
            ty = torch.from_numpy(y[idx])
            tw = torch.from_numpy(weights[idx])

            logits = model(tc, tn)
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits, ty, reduction="none"
                ) * tw
            ).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict_dcn(model, cats, numeric, batch_size=32768):
    result = np.empty(len(cats), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(cats), batch_size):
            end = min(begin + batch_size, len(cats))
            result[begin:end] = model(
                torch.from_numpy(cats[begin:end]),
                torch.from_numpy(numeric[begin:end]),
            ).cpu().numpy()
    return result


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

train_arrays = split_arrays(train)
valid_arrays = split_arrays(valid)

train_rates = smoothed_rate_features(
    train_arrays, y_train, train_arrays, leave_one_out=True
)
valid_rates = smoothed_rate_features(
    train_arrays, y_train, valid_arrays, leave_one_out=False
)

base_train = make_base_matrix(train)
base_valid = make_base_matrix(valid)
gbdt_train = np.concatenate([base_train, train_rates], axis=1)
gbdt_valid = np.concatenate([base_valid, valid_rates], axis=1)

raw_predictions = {}
raw_predictions["target_stats_content"] = nonparametric_scores(
    valid_rates, "content"
)
raw_predictions["target_stats_balanced"] = nonparametric_scores(
    valid_rates, "balanced"
)
raw_predictions["target_stats_personal"] = nonparametric_scores(
    valid_rates, "personal"
)

gbdt = fit_gbdt(
    gbdt_train, y_train, np.asarray(train.date), rounds=240
)
raw_predictions["statistics_gbdt"] = gbdt.predict(
    gbdt_valid, num_iteration=gbdt.current_iteration()
).astype(np.float32)
del gbdt
gc.collect()

train_cats = cat_matrix(train_arrays)
valid_cats = cat_matrix(valid_arrays)

# DCN receives the continuous quantities and stable target-statistic channels.
dcn_num_train = np.concatenate([
    base_train[:, len(CAT_FIELDS):],
    train_rates,
], axis=1)
dcn_num_valid = np.concatenate([
    base_valid[:, len(CAT_FIELDS):],
    valid_rates,
], axis=1)
dcn_num_train, dcn_num_valid, _, _ = normalize_numeric(
    dcn_num_train, dcn_num_valid
)

dcn = fit_dcn(
    train_cats,
    dcn_num_train,
    y_train,
    np.asarray(train.date),
    epochs=2,
)
raw_predictions["dcn_cross_statistics"] = predict_dcn(
    dcn, valid_cats, dcn_num_valid
)
del dcn
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_rank = within_user_ranks(valid.user_id, inc_valid)

candidates = {
    "trusted_incumbent": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    )
}
records = [{
    "key": "trusted_incumbent",
    "family": "target_stats_balanced",
    "alpha": 0.0,
    "scores": inc_rank,
    "raw": raw_predictions["target_stats_balanced"],
    "metrics": evaluate(valid.user_id, y_valid, inc_rank),
}]

for family, raw in raw_predictions.items():
    raw = np.asarray(raw, dtype=np.float64)
    raw_metric = evaluate(valid.user_id, y_valid, raw)
    candidates[family + "_raw"] = float(raw_metric["primary"])
    raw_rank = within_user_ranks(valid.user_id, raw)

    for alpha in (0.15, 0.30, 0.50, 0.70, 1.0):
        blend = (1.0 - alpha) * inc_rank + alpha * raw_rank
        metric = evaluate(valid.user_id, y_valid, blend)
        key = family + "_blend_" + str(alpha)
        candidates[key] = float(metric["primary"])
        records.append({
            "key": key,
            "family": family,
            "alpha": float(alpha),
            "scores": blend,
            "raw": raw,
            "metrics": metric,
        })

winner = max(records, key=lambda r: r["metrics"]["primary"])

print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": winner["key"],
    "winner_family": winner["family"],
    "winner_alpha": winner["alpha"],
    "target_stats_best_raw": max(
        candidates["target_stats_content_raw"],
        candidates["target_stats_balanced_raw"],
        candidates["target_stats_personal_raw"],
    ),
    "statistics_gbdt_raw": candidates["statistics_gbdt_raw"],
    "dcn_raw": candidates["dcn_cross_statistics_raw"],
    "incumbent_primary": candidates["trusted_incumbent"],
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(winner["scores"], dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(winner["raw"], dtype=np.float64),
    )

# Refit the selected recipe on train + validation and score test.
# Test labels are never accessed.
test = load("test")
test_arrays = split_arrays(test)
combined_arrays = concatenate_arrays(train_arrays, valid_arrays)
y_combined = np.concatenate([
    y_train,
    y_valid.astype(np.float32, copy=False),
])
dates_combined = np.concatenate([
    np.asarray(train.date),
    np.asarray(valid.date),
])

combined_rates = smoothed_rate_features(
    combined_arrays,
    y_combined,
    combined_arrays,
    leave_one_out=True,
)
test_rates = smoothed_rate_features(
    combined_arrays,
    y_combined,
    test_arrays,
    leave_one_out=False,
)

selected_family = winner["family"]

if selected_family.startswith("target_stats_"):
    style = selected_family.replace("target_stats_", "")
    raw_test = nonparametric_scores(test_rates, style)

elif selected_family == "statistics_gbdt":
    base_combined = np.concatenate([base_train, base_valid], axis=0)
    base_test = make_base_matrix(test)
    x_combined = np.concatenate(
        [base_combined, combined_rates], axis=1
    )
    x_test = np.concatenate([base_test, test_rates], axis=1)

    refit = fit_gbdt(
        x_combined, y_combined, dates_combined, rounds=240
    )
    raw_test = refit.predict(
        x_test, num_iteration=refit.current_iteration()
    ).astype(np.float32)
    del refit, x_combined, x_test, base_test
    gc.collect()

else:
    combined_cats = cat_matrix(combined_arrays)
    test_cats = cat_matrix(test_arrays)

    base_combined = np.concatenate([base_train, base_valid], axis=0)
    base_test = make_base_matrix(test)
    num_combined = np.concatenate([
        base_combined[:, len(CAT_FIELDS):],
        combined_rates,
    ], axis=1)
    num_test = np.concatenate([
        base_test[:, len(CAT_FIELDS):],
        test_rates,
    ], axis=1)

    num_combined, num_test, _, _ = normalize_numeric(
        num_combined, num_test
    )
    refit_dcn = fit_dcn(
        combined_cats,
        num_combined,
        y_combined,
        dates_combined,
        epochs=2,
    )
    raw_test = predict_dcn(
        refit_dcn, test_cats, num_test
    )
    del refit_dcn
    gc.collect()

inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)
test_inc_rank = within_user_ranks(test.user_id, inc_test)
test_raw_rank = within_user_ranks(test.user_id, raw_test)

test_scores = (
    (1.0 - winner["alpha"]) * test_inc_rank
    + winner["alpha"] * test_raw_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

metrics = winner["metrics"]
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(time.time() - START),
}))