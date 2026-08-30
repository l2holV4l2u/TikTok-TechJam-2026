import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 7321
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "upload_type",
    "music_type",
    "duration_bucket",
    "hour",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat7",
    "user_active_degree",
    "register_days_bucket",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
HALF_LIFE_DAYS = 7.0
BATCH_SIZE = 8192
FIB_EPOCHS = 4
DEVICE = torch.device("cpu")

train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)


def temporal_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    day_index = {int(d): i for i, d in enumerate(unique_dates.tolist())}
    idx = np.fromiter(
        (day_index[int(d)] for d in dates),
        dtype=np.int16,
        count=len(dates),
    )
    age = idx.max() - idx
    return np.power(0.5, age.astype(np.float32) / HALF_LIFE_DAYS).astype(np.float32)


def make_lgb_matrix(split):
    cols = []
    for f in CAT_FIELDS:
        cols.append(np.asarray(split.X[f], dtype=np.float32))
    for f in NUM_FIELDS:
        x = np.asarray(split.num[f], dtype=np.float32)
        x = np.nan_to_num(x, nan=-1.0, posinf=1e8, neginf=-1.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        cols.append(x.astype(np.float32, copy=False))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def train_lgb_valid():
    Xtr = make_lgb_matrix(train)
    Xva = make_lgb_matrix(valid)
    weights = temporal_weights(train.date)

    dtrain = lgb.Dataset(
        Xtr,
        label=y_train,
        weight=weights,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        Xva,
        label=y_valid,
        reference=dtrain,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "cat_smooth": 30.0,
        "cat_l2": 10.0,
        "verbosity": -1,
        "num_threads": min(8, os.cpu_count() or 1),
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
    }
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=320,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(35, verbose=False)],
    )
    score = model.predict(
        Xva, num_iteration=model.best_iteration, raw_score=True
    ).astype(np.float64)
    rounds = int(model.best_iteration)
    del model, dtrain, dvalid, Xtr, Xva, weights
    gc.collect()
    return score, rounds, params


offsets = np.cumsum(
    [0] + [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS[:-1]],
    dtype=np.int64,
)
TOTAL_CARD = int(sum(int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS))


def make_cat_matrix(split):
    x = np.column_stack([split.X[f] for f in CAT_FIELDS]).astype(
        np.int64, copy=False
    )
    x = x + offsets[None, :]
    return np.ascontiguousarray(x)


class FiBiNetLite(nn.Module):
    def __init__(self, cardinality, n_fields, emb_dim=8):
        super().__init__()
        self.n_fields = n_fields
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(cardinality, emb_dim)
        self.linear = nn.Embedding(cardinality, 1)
        reduction = max(4, n_fields // 3)
        self.se_fc1 = nn.Linear(n_fields, reduction)
        self.se_fc2 = nn.Linear(reduction, n_fields)
        self.bilinear = nn.Linear(emb_dim, emb_dim, bias=False)

        n_pairs = n_fields * (n_fields - 1) // 2
        self.mlp = nn.Sequential(
            nn.Linear(n_pairs * emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.015)
        nn.init.zeros_(self.linear.weight)

        pi, pj = np.triu_indices(n_fields, k=1)
        self.register_buffer("pair_i", torch.from_numpy(pi.astype(np.int64)))
        self.register_buffer("pair_j", torch.from_numpy(pj.astype(np.int64)))

    def forward(self, x):
        e = self.embedding(x)
        squeeze = e.mean(dim=2)
        gates = torch.sigmoid(
            self.se_fc2(torch.relu(self.se_fc1(squeeze)))
        ).unsqueeze(2)
        gated = e * gates
        left = self.bilinear(gated[:, self.pair_i, :])
        pair_features = left * gated[:, self.pair_j, :]
        deep = self.mlp(pair_features.flatten(1)).squeeze(1)
        wide = self.linear(x).sum(dim=1).squeeze(1)
        return self.bias + wide + deep


@torch.no_grad()
def torch_predict(model, x_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float64)
    for start in range(0, len(x_np), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(x_np))
        xb = torch.from_numpy(x_np[start:end]).to(DEVICE)
        result[start:end] = model(xb).cpu().numpy().astype(np.float64)
    return result


def fit_fibinet_valid():
    xtr = make_cat_matrix(train)
    xva = make_cat_matrix(valid)
    weights = temporal_weights(train.date)
    model = FiBiNetLite(TOTAL_CARD, len(CAT_FIELDS)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0015, weight_decay=1e-6
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 20)

    best_primary = -1.0
    best_score = None
    best_state = None
    best_epoch = 1

    ytr_t = torch.from_numpy(y_train)
    wtr_t = torch.from_numpy(weights)
    n = len(xtr)

    for epoch in range(1, FIB_EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = torch.from_numpy(xtr[idx.numpy()]).to(DEVICE)
            yb = ytr_t[idx].to(DEVICE)
            wb = wtr_t[idx].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (losses * wb).sum() / wb.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        score = torch_predict(model, xva)
        met = evaluate(valid.user_id, y_valid, score)
        if float(met["primary"]) > best_primary:
            best_primary = float(met["primary"])
            best_score = score.copy()
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    del model, optimizer, xtr, xva, weights, ytr_t, wtr_t
    gc.collect()
    return best_score, best_epoch


def fit_svd_scores(fit_splits, target_split, rank=28):
    user_card = int(FEATURE_CARDINALITIES["user_id"])
    video_card = int(FEATURE_CARDINALITIES["video_id"])

    all_u = []
    all_v = []
    all_values = []
    for split, labels in fit_splits:
        labels = np.asarray(labels, dtype=np.float32)
        weights = temporal_weights(split.date)
        positive = labels > 0.5
        all_u.append(np.asarray(split.user_id[positive], dtype=np.int32))
        all_v.append(np.asarray(split.video_id[positive], dtype=np.int32))
        all_values.append(weights[positive])

    u = np.concatenate(all_u)
    v = np.concatenate(all_v)
    values = np.concatenate(all_values)
    matrix = sparse.coo_matrix(
        (values, (u, v)),
        shape=(user_card, video_card),
        dtype=np.float32,
    ).tocsr()
    matrix.data = np.log1p(matrix.data)
    user_norm = np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())
    user_norm = np.maximum(user_norm, 1.0).astype(np.float32)
    matrix = sparse.diags(1.0 / user_norm).dot(matrix).tocsr()

    U, S, VT = svds(
        matrix,
        k=rank,
        which="LM",
        return_singular_vectors=True,
        random_state=SEED,
    )
    order = np.argsort(S)[::-1]
    U = U[:, order].astype(np.float32)
    S = S[order].astype(np.float32)
    VT = VT[order].astype(np.float32)

    tu = np.asarray(target_split.user_id, dtype=np.int64)
    tv = np.asarray(target_split.video_id, dtype=np.int64)
    score = np.sum((U[tu] * S[None, :]) * VT[:, tv].T, axis=1)
    return score.astype(np.float64)


def standardized_blend(own, incumbent, own_scale, incumbent_scale, weight):
    return (
        weight * np.asarray(own, dtype=np.float64) / own_scale
        + (1.0 - weight)
        * np.asarray(incumbent, dtype=np.float64)
        / incumbent_scale
    )


shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

lgb_valid, lgb_rounds, lgb_params = train_lgb_valid()
fib_valid, fib_epoch = fit_fibinet_valid()
svd_valid = fit_svd_scores([(train, y_train)], valid)

families = {
    "recency_lgb": lgb_valid,
    "fibinet_bilinear": fib_valid,
    "latent_svd": svd_valid,
}
inc_scale = max(float(np.std(inc_valid)), 1e-8)

candidate_scores = {}
candidate_details = {}
best_primary = -np.inf
best_name = None
best_family = None
best_weight = None
best_valid = None
best_raw_valid = None
best_metrics = None
best_own_scale = None

for family, own in families.items():
    own_scale = max(float(np.std(own)), 1e-8)

    own_metrics = evaluate(valid.user_id, y_valid, own)
    own_name = family + "_standalone"
    candidate_scores[own_name] = float(own_metrics["primary"])
    candidate_details[own_name] = {
        "family": family,
        "weight": 1.0,
        "scale": own_scale,
    }
    if float(own_metrics["primary"]) > best_primary:
        best_primary = float(own_metrics["primary"])
        best_name = own_name
        best_family = family
        best_weight = 1.0
        best_valid = own.copy()
        best_raw_valid = own.copy()
        best_metrics = own_metrics
        best_own_scale = own_scale

    family_best = (-np.inf, None, None, None)
    for weight in [0.15, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85]:
        blended = standardized_blend(
            own, inc_valid, own_scale, inc_scale, weight
        )
        met = evaluate(valid.user_id, y_valid, blended)
        p = float(met["primary"])
        if p > family_best[0]:
            family_best = (p, weight, blended, met)

    p, weight, blended, met = family_best
    blend_name = family + "_incumbent_blend"
    candidate_scores[blend_name] = p
    candidate_details[blend_name] = {
        "family": family,
        "weight": float(weight),
        "scale": own_scale,
    }
    if p > best_primary:
        best_primary = p
        best_name = blend_name
        best_family = family
        best_weight = float(weight)
        best_valid = blended.copy()
        best_raw_valid = own.copy()
        best_metrics = met
        best_own_scale = own_scale

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s family=%s own_weight=%.2f lgb_rounds=%d fib_epoch=%d"
    % (best_name, best_family, best_weight, lgb_rounds, fib_epoch)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    if best_weight < 0.999999:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

# Refit only the selected family on train + validation.
test = load("test")
y_combined = np.concatenate(
    [y_train, y_valid.astype(np.float32)], axis=0
)

if best_family == "recency_lgb":
    Xtr = make_lgb_matrix(train)
    Xva = make_lgb_matrix(valid)
    Xcombined = np.ascontiguousarray(
        np.concatenate([Xtr, Xva], axis=0), dtype=np.float32
    )
    combined_dates = np.concatenate(
        [np.asarray(train.date), np.asarray(valid.date)]
    )
    combined_weights = temporal_weights(combined_dates)
    dfinal = lgb.Dataset(
        Xcombined,
        label=y_combined,
        weight=combined_weights,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    final_model = lgb.train(
        lgb_params,
        dfinal,
        num_boost_round=lgb_rounds,
    )
    Xtest = make_lgb_matrix(test)
    raw_test = final_model.predict(
        Xtest, num_iteration=lgb_rounds, raw_score=True
    ).astype(np.float64)

elif best_family == "fibinet_bilinear":
    xtr = make_cat_matrix(train)
    xva = make_cat_matrix(valid)
    xcombined = np.ascontiguousarray(
        np.concatenate([xtr, xva], axis=0)
    )
    combined_dates = np.concatenate(
        [np.asarray(train.date), np.asarray(valid.date)]
    )
    combined_weights = temporal_weights(combined_dates)

    torch.manual_seed(SEED)
    final_model = FiBiNetLite(TOTAL_CARD, len(CAT_FIELDS)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=0.0015, weight_decay=1e-6
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 20)
    y_t = torch.from_numpy(y_combined)
    w_t = torch.from_numpy(combined_weights)
    n = len(xcombined)

    for _ in range(fib_epoch):
        final_model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = torch.from_numpy(xcombined[idx.numpy()]).to(DEVICE)
            yb = y_t[idx].to(DEVICE)
            wb = w_t[idx].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = final_model(xb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (losses * wb).sum() / wb.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 5.0)
            optimizer.step()

    xtest = make_cat_matrix(test)
    raw_test = torch_predict(final_model, xtest)

else:
    raw_test = fit_svd_scores(
        [(train, y_train), (valid, y_valid.astype(np.float32))],
        test,
    )

if best_weight < 0.999999:
    inc_test = np.load(inc_test_path).astype(np.float64)
    test_scores = standardized_blend(
        raw_test,
        inc_test,
        best_own_scale,
        inc_scale,
        best_weight,
    )
else:
    test_scores = raw_test

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        float(elapsed),
    )
)