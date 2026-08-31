import os
import time
import json
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 91731
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag", "upload_type",
    "onehot_feat3", "onehot_feat8", "duration_bucket", "music_type",
    "user_active_degree", "register_days_bucket",
]
TE_FIELDS = [
    "video_id", "author_id", "tab", "tag", "upload_type",
    "onehot_feat3", "onehot_feat8", "duration_bucket",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
HISTORY_SUFFIXES = [
    "train_count_log1p", "long_view_rate", "is_click_rate",
    "play_time_ms_logmean",
]

BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
HALF_LIFE = 4.0


def recency_weights(dates, half_life=HALF_LIFE):
    dates = np.asarray(dates, dtype=np.int64)
    age = dates.max() - dates
    w = np.power(0.5, age.astype(np.float32) / half_life)
    w /= max(float(w.mean()), 1e-8)
    return np.ascontiguousarray(w, dtype=np.float32)


def make_cat_matrix(split):
    cards = [int(FEATURE_CARDINALITIES[name]) for name in CAT_FIELDS]
    offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
    x = np.empty((len(split.user_id), len(CAT_FIELDS)), dtype=np.int64)
    for j, name in enumerate(CAT_FIELDS):
        x[:, j] = np.asarray(split.X[name], dtype=np.int64) + offsets[j]
    return np.ascontiguousarray(x), cards


def make_numeric_raw(split, split_name):
    columns = []
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    for entity in ("video_id", "author_id"):
        history = historical_features(split_name, key=entity)
        for suffix in HISTORY_SUFFIXES:
            key = entity + "_" + suffix
            x = np.asarray(history[key], dtype=np.float32)
            columns.append(
                np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                .astype(np.float32)
            )
        del history

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def fit_numeric_transform(x):
    lo = np.quantile(x, 0.002, axis=0).astype(np.float32)
    hi = np.quantile(x, 0.998, axis=0).astype(np.float32)
    z = np.clip(x, lo, hi)
    mean = z.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = z.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return lo, hi, mean, std


def apply_numeric_transform(x, transform):
    lo, hi, mean, std = transform
    return np.ascontiguousarray(
        (np.clip(x, lo, hi) - mean) / std,
        dtype=np.float32,
    )


def safe_logit(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def make_target_encoding_features(train, valid, test, weights):
    y = np.asarray(train.y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    prior = float(np.sum(w * y) / np.sum(w))

    train_cols = []
    valid_cols = []
    test_cols = []

    for name in TE_FIELDS:
        card = int(FEATURE_CARDINALITIES[name])
        tr_ids = np.asarray(train.X[name], dtype=np.int64)
        va_ids = np.asarray(valid.X[name], dtype=np.int64)
        te_ids = np.asarray(test.X[name], dtype=np.int64)

        total_w = np.bincount(tr_ids, weights=w, minlength=card)
        total_y = np.bincount(tr_ids, weights=w * y, minlength=card)

        # More shrinkage for high-cardinality identities and less for stable
        # low-cardinality context fields.
        if name in ("video_id", "author_id"):
            strength = 24.0
        elif name in ("onehot_feat3", "onehot_feat8"):
            strength = 16.0
        else:
            strength = 10.0

        loo_num = total_y[tr_ids] - w * y + strength * prior
        loo_den = total_w[tr_ids] - w + strength
        tr_rate = loo_num / np.maximum(loo_den, 1e-8)

        full_rate = (total_y + strength * prior) / (total_w + strength)
        va_rate = full_rate[va_ids]
        te_rate = full_rate[te_ids]

        train_cols.append(safe_logit(tr_rate).astype(np.float32))
        valid_cols.append(safe_logit(va_rate).astype(np.float32))
        test_cols.append(safe_logit(te_rate).astype(np.float32))

    tr = np.ascontiguousarray(np.column_stack(train_cols), dtype=np.float32)
    va = np.ascontiguousarray(np.column_stack(valid_cols), dtype=np.float32)
    te = np.ascontiguousarray(np.column_stack(test_cols), dtype=np.float32)

    mean = tr.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = tr.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    tr = np.ascontiguousarray((tr - mean) / std, dtype=np.float32)
    va = np.ascontiguousarray((va - mean) / std, dtype=np.float32)
    te = np.ascontiguousarray((te - mean) / std, dtype=np.float32)
    return tr, va, te


class AdditiveTargetModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def train_additive_target_model(x, y, weights):
    model = AdditiveTargetModel(x.shape[1])
    nn.init.zeros_(model.linear.weight)
    nn.init.zeros_(model.linear.bias)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.025, weight_decay=2e-3
    )
    rng = np.random.default_rng(SEED + 1)
    y = np.asarray(y, dtype=np.float32)

    model.train()
    for _ in range(3):
        order = rng.permutation(len(y))
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = torch.from_numpy(x[idx])
            yb = torch.from_numpy(y[idx])
            wb = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (row_loss * wb).sum() / wb.sum()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_additive(model, x):
    model.eval()
    out = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(x))
        out[start:end] = model(torch.from_numpy(x[start:end])).numpy()
    return out


def fit_svd_scores(train, valid, test, rank=28):
    users = np.asarray(train.user_id, dtype=np.int64)
    videos = np.asarray(train.video_id, dtype=np.int64)
    y = np.asarray(train.y, dtype=np.float64)
    weights = recency_weights(train.date).astype(np.float64)

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])

    user_w = np.bincount(users, weights=weights, minlength=n_users)
    user_y = np.bincount(users, weights=weights * y, minlength=n_users)
    global_rate = float(np.sum(weights * y) / np.sum(weights))
    user_rate = (user_y + 12.0 * global_rate) / (user_w + 12.0)

    residual = (y - user_rate[users]) * weights
    sum_matrix = sparse.coo_matrix(
        (residual, (users, videos)),
        shape=(n_users, n_videos),
        dtype=np.float64,
    ).tocsr()
    count_matrix = sparse.coo_matrix(
        (weights, (users, videos)),
        shape=(n_users, n_videos),
        dtype=np.float64,
    ).tocsr()

    sum_matrix.sum_duplicates()
    count_matrix.sum_duplicates()
    sum_matrix.data /= np.maximum(count_matrix.data, 1e-8)

    u, singular, vt = svds(
        sum_matrix.astype(np.float32),
        k=rank,
        which="LM",
        random_state=SEED,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)

    left = u * np.sqrt(singular)[None, :]
    right = vt.T * np.sqrt(singular)[None, :]

    def score(split):
        su = np.asarray(split.user_id, dtype=np.int64)
        sv = np.asarray(split.video_id, dtype=np.int64)
        return np.sum(left[su] * right[sv], axis=1).astype(np.float32)

    return score(valid), score(test)


class ESMM(nn.Module):
    def __init__(self, total_categories, n_fields, n_num, k=10):
        super().__init__()
        self.embedding = nn.Embedding(total_categories, k)
        self.linear = nn.Embedding(total_categories, 1)
        dim = n_fields * k + n_num

        self.shared = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.click_tower = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.conditional_tower = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.num_linear = nn.Linear(n_num, 1)

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, cats, nums, return_parts=False):
        emb = self.embedding(cats).flatten(start_dim=1)
        shared = self.shared(torch.cat([emb, nums], dim=1))

        wide = (
            self.linear(cats).sum(dim=1).squeeze(1)
            + self.num_linear(nums).squeeze(1)
        )
        click_logit = self.click_tower(shared).squeeze(1)
        conditional_logit = self.conditional_tower(shared).squeeze(1) + wide

        log_p = (
            F.logsigmoid(click_logit)
            + F.logsigmoid(conditional_logit)
        )
        log_p = torch.clamp(log_p, max=-1e-6)
        long_logit = log_p - torch.log(-torch.expm1(log_p))

        if return_parts:
            return long_logit, click_logit
        return long_logit


def train_esmm(model, cats, nums, long_labels, click_labels, weights):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0023, weight_decay=3e-5
    )
    rng = np.random.default_rng(SEED + 2)
    y = np.asarray(long_labels, dtype=np.float32)
    click = np.asarray(click_labels, dtype=np.float32)

    model.train()
    for _ in range(2):
        order = rng.permutation(len(y))
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            cb = torch.from_numpy(cats[idx])
            xb = torch.from_numpy(nums[idx])
            yb = torch.from_numpy(y[idx])
            kb = torch.from_numpy(click[idx])
            wb = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            long_logit, click_logit = model(cb, xb, return_parts=True)
            main_loss = F.binary_cross_entropy_with_logits(
                long_logit, yb, reduction="none"
            )
            click_loss = F.binary_cross_entropy_with_logits(
                click_logit, kb, reduction="none"
            )
            loss = ((main_loss + 0.30 * click_loss) * wb).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


@torch.no_grad()
def predict_esmm(model, cats, nums):
    model.eval()
    out = np.empty(len(cats), dtype=np.float32)
    for start in range(0, len(cats), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(cats))
        out[start:end] = model(
            torch.from_numpy(cats[start:end]),
            torch.from_numpy(nums[start:end]),
        ).numpy()
    return out


def user_standardize(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    _, inv = np.unique(users, return_inverse=True)
    counts = np.bincount(inv)
    means = np.bincount(inv, weights=scores) / np.maximum(counts, 1)
    centered = scores - means[inv]
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0
    return centered / scale


train = load("train")
valid = load("valid")
test = load("test")

train_y = np.asarray(train.y, dtype=np.float32)
weights = recency_weights(train.date)

# Family 1: recency-weighted empirical-Bayes target encodings with an
# additive calibrated prediction rule.
te_train, te_valid, te_test = make_target_encoding_features(
    train, valid, test, weights
)
target_model = train_additive_target_model(
    te_train, train_y, weights
)
target_valid = predict_additive(target_model, te_valid)
target_test = predict_additive(target_model, te_test)
del te_train, te_valid, te_test, target_model
gc.collect()

# Family 2: collaborative latent residual structure.
try:
    svd_valid, svd_test = fit_svd_scores(train, valid, test)
    svd_ok = True
except Exception as exc:
    print("FINDINGS svd_failure=" + repr(exc))
    svd_valid = np.zeros(len(valid.user_id), dtype=np.float32)
    svd_test = np.zeros(len(test.user_id), dtype=np.float32)
    svd_ok = False
gc.collect()

# Family 3: ESMM cascade trained with click only as a train-side auxiliary
# target; no row outcome is used at prediction time.
train_cats, cards = make_cat_matrix(train)
valid_cats, _ = make_cat_matrix(valid)
test_cats, _ = make_cat_matrix(test)

raw_train = make_numeric_raw(train, "train")
raw_valid = make_numeric_raw(valid, "valid")
raw_test = make_numeric_raw(test, "test")
num_transform = fit_numeric_transform(raw_train)
train_num = apply_numeric_transform(raw_train, num_transform)
valid_num = apply_numeric_transform(raw_valid, num_transform)
test_num = apply_numeric_transform(raw_test, num_transform)
del raw_train, raw_valid, raw_test
gc.collect()

if "is_click" in train.aux:
    click_y = np.asarray(train.aux["is_click"], dtype=np.float32)
    click_y = np.nan_to_num(click_y, nan=0.0, posinf=1.0, neginf=0.0)
    click_y = np.clip(click_y, 0.0, 1.0)
else:
    click_y = train_y.copy()

esmm = ESMM(
    total_categories=int(sum(cards)),
    n_fields=len(CAT_FIELDS),
    n_num=train_num.shape[1],
)
train_esmm(
    esmm, train_cats, train_num, train_y, click_y, weights
)
esmm_valid = predict_esmm(esmm, valid_cats, valid_num)
esmm_test = predict_esmm(esmm, test_cats, test_num)

del esmm, train_cats, valid_cats, test_cats
del train_num, valid_num, test_num, click_y
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)
valid_labels = np.asarray(valid.y, dtype=np.int8)

inc_v = user_standardize(inc_valid, valid_users)
inc_t = user_standardize(inc_test, test_users)

families = {
    "target_stats": (
        user_standardize(target_valid, valid_users),
        user_standardize(target_test, test_users),
    ),
    "esmm": (
        user_standardize(esmm_valid, valid_users),
        user_standardize(esmm_test, test_users),
    ),
}
if svd_ok:
    families["latent_svd"] = (
        user_standardize(svd_valid, valid_users),
        user_standardize(svd_test, test_users),
    )

# A cross-family ensemble is also structurally complementary, but its blend
# weights are fixed equal weights rather than fitted on validation.
family_names = list(families.keys())
ensemble_valid = np.mean(
    np.stack([families[n][0] for n in family_names], axis=0), axis=0
)
ensemble_test = np.mean(
    np.stack([families[n][1] for n in family_names], axis=0), axis=0
)
families["cross_family_ensemble"] = (
    user_standardize(ensemble_valid, valid_users),
    user_standardize(ensemble_test, test_users),
)

candidate_scores = {}
best_primary = -np.inf
best_name = None
best_valid = None
best_test = None
best_raw_valid = None

inc_metrics = evaluate(valid_users, valid_labels, inc_v)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
best_primary = float(inc_metrics["primary"])
best_name = "incumbent"
best_valid = inc_v
best_test = inc_t
best_raw_valid = inc_v

for name, (own_v, own_t) in families.items():
    standalone = evaluate(valid_users, valid_labels, own_v)
    candidate_scores[name] = float(standalone["primary"])

    if float(standalone["primary"]) > best_primary:
        best_primary = float(standalone["primary"])
        best_name = name
        best_valid = own_v
        best_test = own_t
        best_raw_valid = own_v

    for alpha in (0.15, 0.30, 0.50, 0.70, 0.85):
        blend_v = alpha * own_v + (1.0 - alpha) * inc_v
        blend_t = alpha * own_t + (1.0 - alpha) * inc_t
        metrics = evaluate(valid_users, valid_labels, blend_v)
        key = "{}_blend_{:.2f}".format(name, alpha)
        candidate_scores[key] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_name = key
            best_valid = blend_v
            best_test = blend_t
            best_raw_valid = own_v

final_metrics = evaluate(valid_users, valid_labels, best_valid)

print(
    "FINDINGS selected={} svd_ok={} target_primary={:.6f} "
    "esmm_primary={:.6f}".format(
        best_name,
        svd_ok,
        candidate_scores["target_stats"],
        candidate_scores["esmm"],
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if "blend" in best_name:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)