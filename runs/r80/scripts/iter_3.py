import os
import time
import json
import copy
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 731
THREADS = max(1, min(8, os.cpu_count() or 1))
BATCH_SIZE = 8192
HALF_LIFE_DAYS = 5.0

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]
NUMERIC = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

offsets = []
total_cardinality = 0
for field in FIELDS:
    offsets.append(total_cardinality)
    total_cardinality += int(FEATURE_CARDINALITIES[field])
OFFSETS = np.asarray(offsets, dtype=np.int64)


def categorical_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[field], dtype=np.int64) + OFFSETS[j]
            for j, field in enumerate(FIELDS)
        ]),
        dtype=np.int64,
    )


def raw_numeric_matrix(split):
    cols = []
    for field in NUMERIC:
        x = np.asarray(split.num[field], dtype=np.float32)
        x = np.where(np.isfinite(x), np.maximum(x, 0.0), np.nan)
        cols.append(np.log1p(x).astype(np.float32))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def fit_numeric_transform(raw):
    med = np.nanmedian(raw, axis=0).astype(np.float32)
    filled = np.where(np.isfinite(raw), raw, med[None, :])
    center = np.median(filled, axis=0).astype(np.float32)
    q25 = np.percentile(filled, 25, axis=0).astype(np.float32)
    q75 = np.percentile(filled, 75, axis=0).astype(np.float32)
    scale = np.maximum(q75 - q25, 0.1).astype(np.float32)
    return med, center, scale


def transform_numeric(raw, stats):
    med, center, scale = stats
    missing = (~np.isfinite(raw)).astype(np.float32)
    filled = np.where(np.isfinite(raw), raw, med[None, :])
    values = np.clip((filled - center[None, :]) / scale[None, :], -8.0, 8.0)
    return np.ascontiguousarray(
        np.concatenate([values.astype(np.float32), missing], axis=1),
        dtype=np.float32,
    )


def training_weights(split):
    dates = np.asarray(split.date, dtype=np.int64)
    max_date = int(dates.max())

    def ordinal(d):
        s = str(int(d))
        import datetime
        return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal()

    unique_dates = np.unique(dates)
    ord_map = {int(d): ordinal(d) for d in unique_dates}
    max_ord = ordinal(max_date)
    age = np.asarray([max_ord - ord_map[int(d)] for d in unique_dates])
    date_to_age = {int(d): float(a) for d, a in zip(unique_dates, age)}
    row_age = np.fromiter(
        (date_to_age[int(d)] for d in dates),
        dtype=np.float32,
        count=len(dates),
    )
    recency = np.exp(-np.log(2.0) * row_age / HALF_LIFE_DAYS)

    uid = np.asarray(split.X["user_id"], dtype=np.int64)
    counts = np.bincount(
        uid, minlength=int(FEATURE_CARDINALITIES["user_id"])
    ).astype(np.float32)
    user_balance = 1.0 / np.sqrt(np.maximum(counts[uid], 1.0))
    user_balance /= float(np.mean(user_balance))
    user_balance = np.clip(user_balance, 0.30, 3.0)

    w = recency * user_balance
    w /= float(np.mean(w))
    return w.astype(np.float32)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if sd < 1e-12:
        return x - float(np.mean(x))
    return (x - float(np.mean(x))) / sd


class DCNv2Lite(nn.Module):
    def __init__(self, prevalence, n_numeric, rank=8, n_cross=3):
        super().__init__()
        self.embedding = nn.Embedding(
            total_cardinality, rank, sparse=True
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.015)

        dim = len(FIELDS) * rank + n_numeric
        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(dim)) for _ in range(n_cross)
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for _ in range(n_cross)
        ])
        for w in self.cross_w:
            nn.init.normal_(w, mean=0.0, std=0.02)

        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 64, 1)
        p = float(np.clip(prevalence, 1e-5, 1 - 1e-5))
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, np.log(p / (1.0 - p)))

    def forward(self, cats, nums):
        emb = self.embedding(cats).reshape(cats.shape[0], -1)
        x0 = torch.cat([emb, nums], dim=1)
        x = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = torch.sum(x * w[None, :], dim=1, keepdim=True)
            x = x + x0 * scalar + b[None, :]
        deep = self.deep(x0)
        return self.output(torch.cat([x, deep], dim=1)).squeeze(1)

    def sparse_parameters(self):
        return [self.embedding.weight]

    def dense_parameters(self):
        result = []
        for name, param in self.named_parameters():
            if name != "embedding.weight":
                result.append(param)
        return result


class ProductNN(nn.Module):
    def __init__(self, prevalence, n_numeric, rank=8):
        super().__init__()
        self.embedding = nn.Embedding(
            total_cardinality, rank, sparse=True
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.015)

        pair_i, pair_j = np.triu_indices(len(FIELDS), k=1)
        self.register_buffer(
            "pair_i", torch.from_numpy(pair_i.astype(np.int64))
        )
        self.register_buffer(
            "pair_j", torch.from_numpy(pair_j.astype(np.int64))
        )
        input_dim = len(FIELDS) * rank + len(pair_i) + n_numeric
        self.network = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        p = float(np.clip(prevalence, 1e-5, 1 - 1e-5))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(
            self.network[-1].bias, np.log(p / (1.0 - p))
        )

    def forward(self, cats, nums):
        e = self.embedding(cats)
        products = (
            e[:, self.pair_i, :] * e[:, self.pair_j, :]
        ).sum(dim=2)
        x = torch.cat(
            [e.reshape(e.shape[0], -1), products, nums], dim=1
        )
        return self.network(x).squeeze(1)

    def sparse_parameters(self):
        return [self.embedding.weight]

    def dense_parameters(self):
        return list(self.network.parameters())


def make_neural(family, prevalence, n_numeric):
    torch.manual_seed(SEED)
    if family == "dcnv2":
        return DCNv2Lite(prevalence, n_numeric)
    if family == "pnn":
        return ProductNN(prevalence, n_numeric)
    raise ValueError(family)


def predict_neural(model, cats, nums):
    model.eval()
    result = np.empty(len(cats), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(cats), 32768):
            end = min(start + 32768, len(cats))
            cb = torch.from_numpy(cats[start:end])
            nb = torch.from_numpy(nums[start:end])
            result[start:end] = model(cb, nb).cpu().numpy()
    return result


def train_neural_epoch(model, cats, nums, labels, weights, rng):
    model.train()
    sparse_opt = getattr(model, "_sparse_opt")
    dense_opt = getattr(model, "_dense_opt")
    order = rng.permutation(len(labels))
    loss_sum = 0.0

    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        cb = torch.from_numpy(cats[idx])
        nb = torch.from_numpy(nums[idx])
        yb = torch.from_numpy(labels[idx])
        wb = torch.from_numpy(weights[idx])

        sparse_opt.zero_grad(set_to_none=True)
        dense_opt.zero_grad(set_to_none=True)
        logits = model(cb, nb)
        losses = F.binary_cross_entropy_with_logits(
            logits, yb, reduction="none"
        )
        loss = torch.sum(losses * wb) / torch.sum(wb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.dense_parameters(), max_norm=5.0
        )
        sparse_opt.step()
        dense_opt.step()
        loss_sum += float(loss.detach()) * len(idx)

    return loss_sum / len(labels)


def fit_neural_valid(
    family, cats_tr, nums_tr, y_tr, weights,
    cats_va, nums_va, y_va, users_va
):
    model = make_neural(family, float(y_tr.mean()), nums_tr.shape[1])
    model._sparse_opt = torch.optim.SparseAdam(
        model.sparse_parameters(), lr=0.002
    )
    model._dense_opt = torch.optim.AdamW(
        model.dense_parameters(), lr=0.0015, weight_decay=1e-5
    )
    rng = np.random.default_rng(SEED)

    best_primary = -np.inf
    best_epoch = 1
    best_predictions = None
    best_state = None

    for epoch in range(1, 5):
        loss = train_neural_epoch(
            model, cats_tr, nums_tr, y_tr, weights, rng
        )
        pred = predict_neural(model, cats_va, nums_va)
        metric = evaluate(users_va, y_va, pred)
        print(
            "FIT family=%s epoch=%d loss=%.6f primary=%.6f"
            % (family, epoch, loss, metric["primary"]),
            flush=True,
        )
        if metric["primary"] > best_primary:
            best_primary = float(metric["primary"])
            best_epoch = epoch
            best_predictions = pred.copy()
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_predictions, best_epoch


def fit_neural_fixed(family, cats, nums, labels, weights, epochs):
    model = make_neural(family, float(labels.mean()), nums.shape[1])
    model._sparse_opt = torch.optim.SparseAdam(
        model.sparse_parameters(), lr=0.002
    )
    model._dense_opt = torch.optim.AdamW(
        model.dense_parameters(), lr=0.0015, weight_decay=1e-5
    )
    rng = np.random.default_rng(SEED)
    for _ in range(epochs):
        train_neural_epoch(
            model, cats, nums, labels, weights, rng
        )
    return model


def lgb_matrix(split, numeric):
    cats = np.column_stack([
        np.asarray(split.X[field], dtype=np.int32)
        for field in FIELDS
    ])
    return np.ascontiguousarray(
        np.column_stack([cats, numeric]).astype(np.float32)
    )


def grouped_order(user_ids):
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1, len(sorted_users)]
    groups = np.diff(starts).astype(np.int32)
    return order, groups


def fit_lambdarank(matrix, labels, weights, user_ids, rounds=140):
    order, groups = grouped_order(np.asarray(user_ids, dtype=np.int64))
    dataset = lgb.Dataset(
        matrix[order],
        label=labels[order],
        weight=weights[order],
        group=groups,
        categorical_feature=list(range(len(FIELDS))),
        free_raw_data=False,
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "label_gain": [0, 1],
        "learning_rate": 0.055,
        "num_leaves": 63,
        "min_data_in_leaf": 250,
        "max_bin": 127,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 3.0,
        "lambdarank_truncation_level": 10,
        "num_threads": THREADS,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=rounds)


tr = load("train")
va = load("valid")

y_tr = np.asarray(tr.y, dtype=np.float32)
y_va_float = np.asarray(va.y, dtype=np.float32)
y_va = np.asarray(va.y, dtype=np.int8)
users_va = np.asarray(va.user_id, dtype=np.int64)

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

cats_tr = categorical_matrix(tr)
cats_va = categorical_matrix(va)
raw_num_tr = raw_numeric_matrix(tr)
raw_num_va = raw_numeric_matrix(va)
numeric_stats = fit_numeric_transform(raw_num_tr)
nums_tr = transform_numeric(raw_num_tr, numeric_stats)
nums_va = transform_numeric(raw_num_va, numeric_stats)
weights_tr = training_weights(tr)

predictions = {}
recipes = {}

for family in ("dcnv2", "pnn"):
    pred, epoch = fit_neural_valid(
        family,
        cats_tr,
        nums_tr,
        y_tr,
        weights_tr,
        cats_va,
        nums_va,
        y_va,
        users_va,
    )
    predictions[family] = pred
    recipes[family] = {"epochs": int(epoch)}

lgb_tr = lgb_matrix(tr, nums_tr)
lgb_va = lgb_matrix(va, nums_va)
rank_model = fit_lambdarank(
    lgb_tr,
    y_tr,
    weights_tr,
    np.asarray(tr.user_id, dtype=np.int64),
    rounds=140,
)
predictions["lambdarank"] = rank_model.predict(
    lgb_va, num_iteration=140
).astype(np.float64)
recipes["lambdarank"] = {"rounds": 140}

candidate_scores = {}
candidate_payload = {}

inc_metric = evaluate(users_va, y_va, inc_valid)
candidate_scores["incumbent"] = float(inc_metric["primary"])
candidate_payload["incumbent"] = ("incumbent", 0.0, inc_valid)

inc_z = zscore(inc_valid)
for family, pred in predictions.items():
    standalone_metric = evaluate(users_va, y_va, pred)
    standalone_name = family + "_standalone"
    candidate_scores[standalone_name] = float(
        standalone_metric["primary"]
    )
    candidate_payload[standalone_name] = (family, 1.0, pred)

    pred_z = zscore(pred)
    for alpha in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90):
        blended = (1.0 - alpha) * inc_z + alpha * pred_z
        metric = evaluate(users_va, y_va, blended)
        name = family + "_blend_%.2f" % alpha
        candidate_scores[name] = float(metric["primary"])
        candidate_payload[name] = (
            family, float(alpha), blended.copy()
        )

winner_name = max(candidate_scores, key=candidate_scores.get)
winner_family, winner_alpha, valid_scores = candidate_payload[winner_name]
metrics = evaluate(users_va, y_va, valid_scores)

print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)
print(
    "FINDINGS " + json.dumps({
        "winner": winner_name,
        "family": winner_family,
        "new_model_weight": float(winner_alpha),
        "half_life_days": HALF_LIFE_DAYS,
        "recipe": recipes.get(winner_family, {}),
    }, sort_keys=True),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

te = load("test")

if winner_family == "incumbent" or winner_alpha == 0.0:
    test_scores = inc_test.copy()
else:
    cats_te = categorical_matrix(te)
    raw_num_te = raw_numeric_matrix(te)

    cats_combined = np.concatenate([cats_tr, cats_va], axis=0)
    raw_num_combined = np.concatenate(
        [raw_num_tr, raw_num_va], axis=0
    )
    y_combined = np.concatenate([y_tr, y_va_float], axis=0)
    combined_stats = fit_numeric_transform(raw_num_combined)
    nums_combined = transform_numeric(
        raw_num_combined, combined_stats
    )
    nums_te = transform_numeric(raw_num_te, combined_stats)

    class CombinedSplit:
        pass

    combined_proxy = CombinedSplit()
    combined_proxy.date = np.concatenate([
        np.asarray(tr.date), np.asarray(va.date)
    ])
    combined_proxy.X = {
        "user_id": np.concatenate([
            np.asarray(tr.X["user_id"]),
            np.asarray(va.X["user_id"]),
        ])
    }
    weights_combined = training_weights(combined_proxy)

    if winner_family in ("dcnv2", "pnn"):
        selected_model = fit_neural_fixed(
            winner_family,
            cats_combined,
            nums_combined,
            y_combined,
            weights_combined,
            recipes[winner_family]["epochs"],
        )
        new_test = predict_neural(
            selected_model, cats_te, nums_te
        )
    elif winner_family == "lambdarank":
        lgb_combined = np.concatenate(
            [lgb_tr, lgb_va], axis=0
        )
        lgb_te = lgb_matrix(te, nums_te)
        combined_users = np.concatenate([
            np.asarray(tr.user_id, dtype=np.int64),
            np.asarray(va.user_id, dtype=np.int64),
        ])
        selected_model = fit_lambdarank(
            lgb_combined,
            y_combined,
            weights_combined,
            combined_users,
            rounds=recipes[winner_family]["rounds"],
        )
        new_test = selected_model.predict(
            lgb_te,
            num_iteration=recipes[winner_family]["rounds"],
        ).astype(np.float64)
    else:
        raise ValueError(winner_family)

    test_scores = (
        (1.0 - winner_alpha) * zscore(inc_test)
        + winner_alpha * zscore(new_test)
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.4f}'
    % (
        metrics["primary"],
        metrics["gauc"],
        metrics["ndcg@5"],
        elapsed,
    ),
    flush=True,
)