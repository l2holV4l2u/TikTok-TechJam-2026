import os
import gc
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

DEVICE = torch.device("cpu")
TRAIN_BATCH = 65536
PRED_BATCH = 262144
MF_DIM = 12
MF_EPOCHS = 4
LINEAR_EPOCHS = 4

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms",
    "user_follow_user_num",
    "user_fans_user_num",
    "user_friend_user_num",
    "user_register_days",
]
RATE_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]
RATE_SMOOTHING = {
    "video_id": 40.0,
    "author_id": 70.0,
    "tag": 500.0,
    "tab": 800.0,
    "duration_bucket": 800.0,
    "upload_type": 700.0,
    "onehot_feat3": 250.0,
    "onehot_feat8": 250.0,
}


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 0.015, 0.985)
    return np.log(p) - np.log1p(-p)


def build_rate_tables(cat, y, global_rate):
    tables = {}
    loo = {}
    for field in RATE_FIELDS:
        x = cat[field]
        card = int(FEATURE_CARDINALITIES[field])
        cnt = np.bincount(x, minlength=card).astype(np.float32)
        pos = np.bincount(x, weights=y, minlength=card).astype(np.float32)
        smooth = float(RATE_SMOOTHING[field])
        full_rate = (pos + smooth * global_rate) / (cnt + smooth)
        row_rate = (
            pos[x] - y + smooth * global_rate
        ) / np.maximum(cnt[x] - 1.0 + smooth, 1.0)
        tables[field] = full_rate.astype(np.float32)
        loo[field] = row_rate.astype(np.float32)
    return tables, loo


def select_history_arrays(split_name):
    selected = {}
    for entity in ("video_id", "author_id"):
        hist = historical_features(split_name, key=entity)
        preferred = []
        for key in hist:
            if (
                key.endswith("train_count_log1p")
                or key.endswith("long_view_rate")
                or key.endswith("is_click_rate")
                or key.endswith("play_time_ms_logmean")
                or key.endswith("comment_stay_time_logmean")
            ):
                preferred.append(key)
        preferred.sort()
        for key in preferred:
            selected[key] = np.asarray(hist[key], dtype=np.float32)
        del hist
        gc.collect()
    return selected


class HistoryLinear(nn.Module):
    def __init__(self, n_features, bias):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(float(bias))

    def forward(self, x):
        return self.linear(x).squeeze(1)


class CollaborativeMF(nn.Module):
    def __init__(self, cards, dim, bias):
        super().__init__()
        self.user = nn.Embedding(cards["user_id"], dim, sparse=True)
        self.video = nn.Embedding(cards["video_id"], dim, sparse=True)
        self.author = nn.Embedding(cards["author_id"], dim, sparse=True)
        self.tag = nn.Embedding(cards["tag"], dim, sparse=True)

        self.user_bias = nn.Embedding(cards["user_id"], 1, sparse=True)
        self.video_bias = nn.Embedding(cards["video_id"], 1, sparse=True)
        self.author_bias = nn.Embedding(cards["author_id"], 1, sparse=True)
        self.tag_bias = nn.Embedding(cards["tag"], 1, sparse=True)

        self.register_buffer(
            "intercept", torch.tensor(float(bias), dtype=torch.float32)
        )
        with torch.no_grad():
            for emb in (self.user, self.video, self.author, self.tag):
                emb.weight.normal_(0.0, 0.025)
            for emb in (
                self.user_bias,
                self.video_bias,
                self.author_bias,
                self.tag_bias,
            ):
                emb.weight.zero_()

    def forward(self, u, v, a, t):
        user_vec = self.user(u)
        item_vec = self.video(v) + self.author(a) + self.tag(t)
        interaction = (user_vec * item_vec).sum(dim=1)
        bias = (
            self.user_bias(u).squeeze(1)
            + self.video_bias(v).squeeze(1)
            + self.author_bias(a).squeeze(1)
            + self.tag_bias(t).squeeze(1)
        )
        return self.intercept + bias + interaction


def make_linear_features(
    indices,
    history_arrays,
    num_arrays,
    rate_arrays,
    rate_source,
    means=None,
    scales=None,
):
    cols = []
    for key in sorted(history_arrays):
        z = np.asarray(history_arrays[key][indices], dtype=np.float32)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(z)

    for field in NUM_FIELDS:
        z = np.asarray(num_arrays[field][indices], dtype=np.float32)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = np.sign(z) * np.log1p(np.abs(z))
        cols.append(z)

    for field in RATE_FIELDS:
        if rate_source == "loo":
            z = rate_arrays[field][indices]
        else:
            ids = rate_source[field][indices]
            z = rate_arrays[field][ids]
        cols.append(safe_logit(z).astype(np.float32))

    x = np.column_stack(cols).astype(np.float32, copy=False)
    if means is not None:
        x = (x - means) / scales
    return x


def empirical_bayes_scores(cat, rate_tables):
    weights = {
        "video_id": 1.10,
        "author_id": 1.45,
        "tag": 0.45,
        "tab": 0.40,
        "duration_bucket": 0.45,
        "upload_type": 0.35,
        "onehot_feat3": 0.65,
        "onehot_feat8": 0.55,
    }
    result = np.zeros(len(next(iter(cat.values()))), dtype=np.float32)
    weight_sum = 0.0
    for field, weight in weights.items():
        result += weight * safe_logit(rate_tables[field][cat[field]])
        weight_sum += weight
    return result / weight_sum


@torch.inference_mode()
def predict_history_linear(
    model,
    n,
    history_arrays,
    num_arrays,
    rate_tables,
    cat,
    means,
    scales,
):
    model.eval()
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, PRED_BATCH):
        end = min(start + PRED_BATCH, n)
        ids = np.arange(start, end, dtype=np.int64)
        xb = make_linear_features(
            ids,
            history_arrays,
            num_arrays,
            rate_tables,
            cat,
            means,
            scales,
        )
        out[start:end] = model(torch.from_numpy(xb)).cpu().numpy()
    return out


@torch.inference_mode()
def predict_mf(model, cat):
    model.eval()
    n = len(cat["user_id"])
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, PRED_BATCH):
        end = min(start + PRED_BATCH, n)
        out[start:end] = model(
            torch.from_numpy(cat["user_id"][start:end]),
            torch.from_numpy(cat["video_id"][start:end]),
            torch.from_numpy(cat["author_id"][start:end]),
            torch.from_numpy(cat["tag"][start:end]),
        ).cpu().numpy()
    return out


# -------------------------- Load training arrays --------------------------

train = load("train")
n_train = len(train.user_id)
train_y = np.asarray(train.y, dtype=np.float32).copy()
global_rate = float(train_y.mean())
initial_bias = float(np.log(global_rate / (1.0 - global_rate)))

train_cat = {
    f: np.asarray(train.X[f], dtype=np.int64).copy() for f in CAT_FIELDS
}
train_num = {
    f: np.asarray(train.num[f], dtype=np.float32).copy() for f in NUM_FIELDS
}
del train
gc.collect()

rate_tables, train_loo_rates = build_rate_tables(
    train_cat, train_y, global_rate
)
train_history = select_history_arrays("train")
history_keys = sorted(train_history)

print(
    "FINDINGS history_features=%d global_rate=%.6f"
    % (len(history_keys), global_rate),
    flush=True,
)

# ----------------------- Family 1: history stacker -----------------------

sample_rng = np.random.default_rng(SEED)
scale_ids = sample_rng.choice(
    n_train, size=min(400000, n_train), replace=False
).astype(np.int64)
scale_x = make_linear_features(
    scale_ids,
    train_history,
    train_num,
    train_loo_rates,
    "loo",
)
feature_means = scale_x.mean(axis=0, dtype=np.float64).astype(np.float32)
feature_scales = scale_x.std(axis=0, dtype=np.float64).astype(np.float32)
feature_scales = np.maximum(feature_scales, 1e-3)
n_linear_features = scale_x.shape[1]
del scale_x, scale_ids
gc.collect()

history_model = HistoryLinear(n_linear_features, initial_bias)
history_opt = torch.optim.AdamW(
    history_model.parameters(), lr=0.025, weight_decay=2e-4
)
gen = torch.Generator()
gen.manual_seed(SEED)

for epoch in range(LINEAR_EPOCHS):
    permutation = torch.randperm(n_train, generator=gen)
    total_loss = 0.0
    total_rows = 0
    history_model.train()
    for start in range(0, n_train, TRAIN_BATCH):
        ids_t = permutation[start:start + TRAIN_BATCH]
        ids = ids_t.numpy()
        xb = make_linear_features(
            ids,
            train_history,
            train_num,
            train_loo_rates,
            "loo",
            feature_means,
            feature_scales,
        )
        yb = torch.from_numpy(train_y[ids])
        history_opt.zero_grad(set_to_none=True)
        logits = history_model(torch.from_numpy(xb))
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        history_opt.step()
        total_loss += float(loss.detach()) * len(ids)
        total_rows += len(ids)
    print(
        "history_epoch=%d loss=%.6f"
        % (epoch + 1, total_loss / total_rows),
        flush=True,
    )
    del permutation

# ---------------------- Family 2: collaborative MF -----------------------

mf_cards = {
    f: int(FEATURE_CARDINALITIES[f])
    for f in ("user_id", "video_id", "author_id", "tag")
}
mf_model = CollaborativeMF(mf_cards, MF_DIM, initial_bias)
mf_opt = torch.optim.SparseAdam(mf_model.parameters(), lr=0.006)
mf_gen = torch.Generator()
mf_gen.manual_seed(SEED + 17)

for epoch in range(MF_EPOCHS):
    permutation = torch.randperm(n_train, generator=mf_gen)
    total_loss = 0.0
    total_rows = 0
    mf_model.train()
    for start in range(0, n_train, TRAIN_BATCH):
        ids = permutation[start:start + TRAIN_BATCH].numpy()
        yb = torch.from_numpy(train_y[ids])
        mf_opt.zero_grad(set_to_none=True)
        logits = mf_model(
            torch.from_numpy(train_cat["user_id"][ids]),
            torch.from_numpy(train_cat["video_id"][ids]),
            torch.from_numpy(train_cat["author_id"][ids]),
            torch.from_numpy(train_cat["tag"][ids]),
        )
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        mf_opt.step()
        total_loss += float(loss.detach()) * len(ids)
        total_rows += len(ids)
    print(
        "mf_epoch=%d loss=%.6f"
        % (epoch + 1, total_loss / total_rows),
        flush=True,
    )
    del permutation

del train_history, train_num, train_loo_rates, train_y
gc.collect()

# ----------------------------- Validation -------------------------------

valid = load("valid")
n_valid = len(valid.user_id)
valid_users = np.asarray(valid.user_id)
valid_y = np.asarray(valid.y)
valid_cat = {
    f: np.asarray(valid.X[f], dtype=np.int64) for f in CAT_FIELDS
}
valid_num = {
    f: np.asarray(valid.num[f], dtype=np.float32) for f in NUM_FIELDS
}
valid_history = select_history_arrays("valid")

valid_history_scores = predict_history_linear(
    history_model,
    n_valid,
    valid_history,
    valid_num,
    rate_tables,
    valid_cat,
    feature_means,
    feature_scales,
)
valid_mf_scores = predict_mf(mf_model, valid_cat)
valid_eb_scores = empirical_bayes_scores(valid_cat, rate_tables)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float32)

families_valid = {
    "history_linear": valid_history_scores,
    "collaborative_mf": valid_mf_scores,
    "empirical_bayes": valid_eb_scores,
}

candidate_scores = {}
best_name = None
best_primary = -np.inf
best_valid_scores = None
best_raw_name = None
best_alpha = None

for name, scores in families_valid.items():
    standalone = evaluate(valid_users, valid_y, scores)
    candidate_scores[name] = float(standalone["primary"])

    # The trusted-incumbent contract explicitly permits choosing a validation
    # blend weight and applying that same fixed weight to test.
    for alpha in (0.15, 0.25, 0.35, 0.50):
        blended = (1.0 - alpha) * inc_valid + alpha * scores
        result = evaluate(valid_users, valid_y, blended)
        blend_name = "%s_blend_%.2f" % (name, alpha)
        candidate_scores[blend_name] = float(result["primary"])
        if result["primary"] > best_primary:
            best_primary = float(result["primary"])
            best_name = blend_name
            best_valid_scores = blended.copy()
            best_raw_name = name
            best_alpha = float(alpha)

final_metrics = evaluate(valid_users, valid_y, best_valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True), flush=True)
print(
    "FINDINGS winner=%s own_family=%s alpha=%.2f"
    % (best_name, best_raw_name, best_alpha),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(families_valid[best_raw_name], dtype=np.float64),
    )

del valid, valid_history, valid_num, valid_cat
del valid_history_scores, valid_mf_scores, valid_eb_scores
del families_valid, best_valid_scores, inc_valid
gc.collect()

# -------------------------------- Test ----------------------------------

test = load("test")
n_test = len(test.user_id)
test_cat = {
    f: np.asarray(test.X[f], dtype=np.int64) for f in CAT_FIELDS
}
test_num = {
    f: np.asarray(test.num[f], dtype=np.float32) for f in NUM_FIELDS
}
test_history = select_history_arrays("test")

if best_raw_name == "history_linear":
    own_test_scores = predict_history_linear(
        history_model,
        n_test,
        test_history,
        test_num,
        rate_tables,
        test_cat,
        feature_means,
        feature_scales,
    )
elif best_raw_name == "collaborative_mf":
    own_test_scores = predict_mf(mf_model, test_cat)
elif best_raw_name == "empirical_bayes":
    own_test_scores = empirical_bayes_scores(test_cat, rate_tables)
else:
    raise RuntimeError("Unknown selected family")

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float32)
test_scores = (1.0 - best_alpha) * inc_test + best_alpha * own_test_scores

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": elapsed,
        }
    )
)