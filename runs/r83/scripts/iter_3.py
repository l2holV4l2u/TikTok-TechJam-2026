import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 7319
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

HALF_LIVES = (3.0, 7.0, 20.0)
SMOOTH = 20.0
ENTITY_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "upload_type",
]
PAIR_FIELDS = ["author_id", "tag"]


def lookup_sorted(unique_keys, values, query):
    pos = np.searchsorted(unique_keys, query)
    ok = pos < len(unique_keys)
    clipped = np.minimum(pos, max(len(unique_keys) - 1, 0))
    if len(unique_keys):
        ok &= unique_keys[clipped] == query
    out = np.zeros(len(query), dtype=np.float32)
    if len(unique_keys):
        out[ok] = values[clipped[ok]]
    return out, ok


def make_keys(split, specification):
    kind, field = specification
    entity = np.asarray(split.X[field], dtype=np.int64)
    if kind == "entity":
        return entity
    user = np.asarray(split.user_id, dtype=np.int64)
    cardinality = int(FEATURE_CARDINALITIES[field])
    return user * np.int64(cardinality) + entity


def build_history_features(reference, reference_y, target,
                           leave_one_out_reference=False):
    """
    Every statistic is built only from reference labels. When target is the
    reference table, its own weighted outcome is removed exactly.
    """
    y = np.asarray(reference_y, dtype=np.float64)
    dates = np.asarray(reference.date, dtype=np.int64)
    max_date = int(dates.max())

    specs = [("entity", f) for f in ENTITY_FIELDS]
    specs += [("pair", f) for f in PAIR_FIELDS]

    columns = []
    eb_by_half_life = [[] for _ in HALF_LIVES]

    for spec in specs:
        ref_key = make_keys(reference, spec)
        target_key = make_keys(target, spec)

        unique_key, inverse = np.unique(ref_key, return_inverse=True)
        unweighted_count = np.bincount(inverse).astype(np.float64)

        target_pos = np.searchsorted(unique_key, target_key)
        target_ok = target_pos < len(unique_key)
        target_clip = np.minimum(target_pos, len(unique_key) - 1)
        target_ok &= unique_key[target_clip] == target_key

        count_target = np.zeros(len(target_key), dtype=np.float32)
        count_target[target_ok] = np.log1p(
            unweighted_count[target_clip[target_ok]]
        ).astype(np.float32)
        columns.append(count_target)

        same_rows = (
            leave_one_out_reference
            and target is reference
            and len(target_key) == len(ref_key)
        )

        for hidx, half_life in enumerate(HALF_LIVES):
            age = (max_date - dates).astype(np.float64)
            row_weight = np.exp2(-age / half_life)
            weighted_count = np.bincount(
                inverse, weights=row_weight, minlength=len(unique_key)
            ).astype(np.float64)
            weighted_pos = np.bincount(
                inverse, weights=row_weight * y, minlength=len(unique_key)
            ).astype(np.float64)

            prior = float(np.sum(row_weight * y) / np.sum(row_weight))

            if same_rows:
                numerator = weighted_pos[inverse] - row_weight * y
                denominator = weighted_count[inverse] - row_weight
                rate = (numerator + SMOOTH * prior) / (
                    denominator + SMOOTH
                )
            else:
                numerator = np.zeros(len(target_key), dtype=np.float64)
                denominator = np.zeros(len(target_key), dtype=np.float64)
                numerator[target_ok] = weighted_pos[target_clip[target_ok]]
                denominator[target_ok] = weighted_count[target_clip[target_ok]]
                rate = (numerator + SMOOTH * prior) / (
                    denominator + SMOOTH
                )

            rate = np.clip(rate, 1e-4, 1.0 - 1e-4)
            logit_rate = np.log(rate / (1.0 - rate)).astype(np.float32)
            columns.append(logit_rate)
            eb_by_half_life[hidx].append(logit_rate)

    X = np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)
    eb_scores = [
        np.mean(np.column_stack(parts), axis=1).astype(np.float32)
        for parts in eb_by_half_life
    ]
    return X, eb_scores


def rank_within_user(user_ids, scores):
    """Vectorized ordinal percentile ranks; ranking is the scored object."""
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    # Primary user ordering, secondary score ordering, tertiary row position.
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, users))
    sorted_users = users[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    ordinal = np.arange(n, dtype=np.float64) - repeated_starts

    ranked_sorted = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked_sorted[multi] = ordinal[multi] / (repeated_lengths[multi] - 1.0)

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


class LinearHistory(nn.Module):
    def __init__(self, n_features, initial_rate):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.zeros_(self.linear.weight)
        initial_rate = float(np.clip(initial_rate, 1e-5, 1 - 1e-5))
        nn.init.constant_(
            self.linear.bias,
            np.log(initial_rate / (1.0 - initial_rate))
        )

    def forward(self, x):
        return self.linear(x).squeeze(1)


def fit_linear(X, y, epochs=8):
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0
    Xn = np.ascontiguousarray((X - mean) / std, dtype=np.float32)

    torch.manual_seed(SEED)
    model = LinearHistory(X.shape[1], float(np.mean(y)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.015, weight_decay=2e-4
    )
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(Xn)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    generator = torch.Generator().manual_seed(SEED)
    batch = 16384

    for _ in range(epochs):
        order = torch.randperm(len(Xn), generator=generator)
        model.train()
        for start in range(0, len(Xn), batch):
            idx = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()

    return model, mean, std


@torch.inference_mode()
def predict_linear(model, X, mean, std):
    Xn = np.ascontiguousarray((X - mean) / std, dtype=np.float32)
    result = np.empty(len(Xn), dtype=np.float32)
    xt = torch.from_numpy(Xn)
    model.eval()
    for start in range(0, len(Xn), 32768):
        end = min(start + 32768, len(Xn))
        result[start:end] = model(xt[start:end]).cpu().numpy()
    return result


def fit_gbdt(X_train, y_train, X_valid=None, y_valid=None, rounds=320):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.045,
        "num_leaves": 31,
        "min_data_in_leaf": 800,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "num_threads": min(8, os.cpu_count() or 1),
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

    if X_valid is not None:
        dvalid = lgb.Dataset(
            X_valid, label=y_valid, reference=dtrain, free_raw_data=False
        )
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(35, verbose=False)],
        )
        selected = int(model.best_iteration)
    else:
        model = lgb.train(params, dtrain, num_boost_round=rounds)
        selected = rounds
    return model, selected


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

# Identical leakage-free history inputs for all three prediction families.
X_train, eb_train = build_history_features(
    train, y_train, train, leave_one_out_reference=True
)
X_valid, eb_valid = build_history_features(
    train, y_train, valid, leave_one_out_reference=False
)

linear_model, linear_mean, linear_std = fit_linear(X_train, y_train)
linear_valid = predict_linear(
    linear_model, X_valid, linear_mean, linear_std
).astype(np.float64)

gbdt_model, best_rounds = fit_gbdt(
    X_train, y_train, X_valid, y_valid, rounds=320
)
gbdt_valid = gbdt_model.predict(
    X_valid, num_iteration=best_rounds
).astype(np.float64)

family_valid = {
    "eb_hl3": eb_valid[0].astype(np.float64),
    "eb_hl7": eb_valid[1].astype(np.float64),
    "eb_hl20": eb_valid[2].astype(np.float64),
    "linear_history": linear_valid,
    "gbdt_history": gbdt_valid,
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
have_incumbent = (
    os.path.exists(inc_valid_path) and os.path.exists(inc_test_path)
)

candidate_scores = {}
candidate_predictions = {}

for name, pred in family_valid.items():
    m = evaluate(valid_users, y_valid, pred)
    candidate_scores[name] = float(m["primary"])
    candidate_predictions[name] = pred

if have_incumbent:
    incumbent_valid = np.load(inc_valid_path).astype(np.float64)
    incumbent_rank = rank_within_user(valid_users, incumbent_valid)

    for name, pred in family_valid.items():
        new_rank = rank_within_user(valid_users, pred)
        for new_weight in (0.25, 0.50, 0.75):
            blend_name = "%s_blend%.2f" % (name, new_weight)
            blend = (
                (1.0 - new_weight) * incumbent_rank
                + new_weight * new_rank
            )
            m = evaluate(valid_users, y_valid, blend)
            candidate_scores[blend_name] = float(m["primary"])
            candidate_predictions[blend_name] = blend

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_predictions[winner].astype(np.float64)
metrics = evaluate(valid_users, y_valid, valid_scores)

print("FINDINGS " + json.dumps({
    "history_features": int(X_train.shape[1]),
    "gbdt_best_rounds": int(best_rounds),
    "winner": winner,
    "standalone_best": max(
        family_valid.keys(), key=lambda k: candidate_scores[k]
    ),
}))

# Refit the identical selected recipes on train+validation for test.
test = load("test")
combined_y = np.concatenate(
    [y_train, y_valid.astype(np.float32)]
)
combined = type("CombinedSplit", (), {})()
combined.X = {
    field: np.concatenate([
        np.asarray(train.X[field]),
        np.asarray(valid.X[field])
    ])
    for field in set(ENTITY_FIELDS + PAIR_FIELDS)
}
combined.user_id = np.concatenate([
    np.asarray(train.user_id), np.asarray(valid.user_id)
])
combined.date = np.concatenate([
    np.asarray(train.date), np.asarray(valid.date)
])

X_combined, eb_combined = build_history_features(
    combined, combined_y, combined, leave_one_out_reference=True
)
X_test, eb_test = build_history_features(
    combined, combined_y, test, leave_one_out_reference=False
)

# Only the family required by the winner is refit, avoiding needless test work.
base_name = winner.split("_blend")[0]
if base_name == "linear_history":
    final_linear, final_mean, final_std = fit_linear(
        X_combined, combined_y
    )
    new_test = predict_linear(
        final_linear, X_test, final_mean, final_std
    ).astype(np.float64)
elif base_name == "gbdt_history":
    final_gbdt, _ = fit_gbdt(
        X_combined, combined_y, rounds=best_rounds
    )
    new_test = final_gbdt.predict(
        X_test, num_iteration=best_rounds
    ).astype(np.float64)
elif base_name == "eb_hl3":
    new_test = eb_test[0].astype(np.float64)
elif base_name == "eb_hl7":
    new_test = eb_test[1].astype(np.float64)
elif base_name == "eb_hl20":
    new_test = eb_test[2].astype(np.float64)
else:
    raise RuntimeError("Unknown selected family: " + base_name)

if "_blend" in winner:
    new_weight = float(winner.rsplit("_blend", 1)[1])
    incumbent_test = np.load(inc_test_path).astype(np.float64)
    test_users = np.asarray(test.user_id, dtype=np.int64)
    test_scores = (
        (1.0 - new_weight)
        * rank_within_user(test_users, incumbent_test)
        + new_weight
        * rank_within_user(test_users, new_test)
    )
else:
    test_scores = new_test

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

print("CANDIDATES " + json.dumps(
    {k: round(v, 7) for k, v in candidate_scores.items()},
    sort_keys=True
))
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(time.time() - START),
}))