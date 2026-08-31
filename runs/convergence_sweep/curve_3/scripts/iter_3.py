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
SEED = 2408
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "duration_bucket",
    "music_type",
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

HISTORY_SUFFIXES = [
    "train_count_log1p",
    "long_view_rate",
    "is_click_rate",
    "play_time_ms_logmean",
]

BATCH_SIZE = 8192
EPOCHS = 3


def load_histories(split_name):
    return {
        "video_id": historical_features(split_name, key="video_id"),
        "author_id": historical_features(split_name, key="author_id"),
    }


def make_cat_matrix(split):
    cards = [int(FEATURE_CARDINALITIES[x]) for x in CAT_FIELDS]
    offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
    out = np.empty((len(split.user_id), len(CAT_FIELDS)), dtype=np.int64)
    for j, name in enumerate(CAT_FIELDS):
        out[:, j] = np.asarray(split.X[name], dtype=np.int64) + offsets[j]
    return out, cards


def raw_numeric_matrix(split, histories):
    cols = []
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    for entity in ("video_id", "author_id"):
        h = histories[entity]
        for suffix in HISTORY_SUFFIXES:
            name = entity + "_" + suffix
            x = np.asarray(h[name], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(x)

    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def fit_numeric_transform(x):
    lo = np.quantile(x, 0.002, axis=0).astype(np.float32)
    hi = np.quantile(x, 0.998, axis=0).astype(np.float32)
    clipped = np.clip(x, lo, hi)
    mean = clipped.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = clipped.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-4] = 1.0
    return lo, hi, mean, std


def apply_numeric_transform(x, transform):
    lo, hi, mean, std = transform
    return np.ascontiguousarray(
        (np.clip(x, lo, hi) - mean) / std,
        dtype=np.float32,
    )


def center_scale(scores, user_ids, scale=None):
    scores = np.asarray(scores, dtype=np.float64)
    _, inv = np.unique(np.asarray(user_ids), return_inverse=True)
    sums = np.bincount(inv, weights=scores)
    counts = np.bincount(inv)
    centered = scores - (sums / np.maximum(counts, 1))[inv]
    if scale is None:
        scale = float(np.std(centered))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
    return centered / scale, float(scale)


class NFM(nn.Module):
    def __init__(self, total_categories, n_fields, n_num, k=12):
        super().__init__()
        self.n_fields = n_fields
        self.embedding = nn.Embedding(total_categories, k)
        self.linear = nn.Embedding(total_categories, 1)
        self.num_linear = nn.Linear(n_num, 1)
        self.mlp = nn.Sequential(
            nn.Linear(k + n_num, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, cats, nums):
        e = self.embedding(cats)
        summed = e.sum(dim=1)
        bi = 0.5 * (summed.square() - e.square().sum(dim=1))
        wide = self.linear(cats).sum(dim=1).squeeze(1)
        wide = wide + self.num_linear(nums).squeeze(1)
        deep = self.mlp(torch.cat([bi, nums], dim=1)).squeeze(1)
        return wide + deep


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x0, x):
        scalar = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x + x0 * scalar + self.bias


class DCN(nn.Module):
    def __init__(self, total_categories, n_fields, n_num, k=8):
        super().__init__()
        self.embedding = nn.Embedding(total_categories, k)
        dim = n_fields * k + n_num
        self.cross1 = CrossLayer(dim)
        self.cross2 = CrossLayer(dim)
        self.deep = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 32, 1)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, cats, nums):
        e = self.embedding(cats).flatten(start_dim=1)
        x0 = torch.cat([e, nums], dim=1)
        cross = self.cross1(x0, x0)
        cross = self.cross2(x0, cross)
        deep = self.deep(x0)
        return self.output(torch.cat([cross, deep], dim=1)).squeeze(1)


def train_neural(model, cats, nums, labels, weights, seed):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003, weight_decay=2e-5
    )
    n = len(labels)
    rng = np.random.default_rng(seed)
    epoch_losses = []

    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        total_loss = 0.0
        total_weight = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            c = torch.from_numpy(cats[idx])
            x = torch.from_numpy(nums[idx])
            y = torch.from_numpy(labels[idx])
            w = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(c, x)
            per_row = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )
            loss = torch.sum(per_row * w) / torch.sum(w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(torch.sum(per_row * w).detach())
            total_weight += float(torch.sum(w))

        epoch_losses.append(total_loss / max(total_weight, 1.0))

    return epoch_losses


@torch.no_grad()
def neural_predict(model, cats, nums):
    model.eval()
    out = np.empty(len(cats), dtype=np.float64)
    for start in range(0, len(cats), 16384):
        stop = min(start + 16384, len(cats))
        c = torch.from_numpy(cats[start:stop])
        x = torch.from_numpy(nums[start:stop])
        out[start:stop] = model(c, x).cpu().numpy()
    return out


def train_svd(train, rank=24):
    users = np.asarray(train.user_id, dtype=np.int64)
    videos = np.asarray(train.video_id, dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.float32)

    positive = labels > 0
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])

    matrix = sparse.coo_matrix(
        (
            np.ones(int(positive.sum()), dtype=np.float32),
            (users[positive], videos[positive]),
        ),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.data[:] = np.log1p(matrix.data)

    u, s, vt = svds(
        matrix,
        k=rank,
        which="LM",
        random_state=SEED,
        return_singular_vectors=True,
    )
    order = np.argsort(s)[::-1]
    s = s[order]
    u = u[:, order]
    vt = vt[order]

    user_factors = np.asarray(u * s[None, :], dtype=np.float32)
    video_factors = np.asarray(vt.T, dtype=np.float32)
    return user_factors, video_factors


def svd_predict(split, user_factors, video_factors):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    return np.einsum(
        "ij,ij->i",
        user_factors[users],
        video_factors[videos],
        optimize=True,
    ).astype(np.float64)


def primary(split, labels, scores):
    return float(evaluate(split.user_id, labels, scores)["primary"])


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

hist_train = load_histories("train")
hist_valid = load_histories("valid")

cat_train, cards = make_cat_matrix(train)
cat_valid, _ = make_cat_matrix(valid)

num_train_raw = raw_numeric_matrix(train, hist_train)
num_valid_raw = raw_numeric_matrix(valid, hist_valid)
num_transform = fit_numeric_transform(num_train_raw)
num_train = apply_numeric_transform(num_train_raw, num_transform)
num_valid = apply_numeric_transform(num_valid_raw, num_transform)

del num_train_raw, num_valid_raw, hist_train
gc.collect()

last_date = int(np.max(np.asarray(train.date)))
days_old = last_date - np.asarray(train.date, dtype=np.int64)
train_weights = np.exp2(-days_old / 4.0).astype(np.float32)
train_weights /= np.mean(train_weights)

total_categories = int(sum(cards))
n_fields = len(CAT_FIELDS)
n_num = num_train.shape[1]

nfm = NFM(total_categories, n_fields, n_num)
nfm_losses = train_neural(
    nfm, cat_train, num_train, y_train, train_weights, SEED + 1
)
nfm_valid = neural_predict(nfm, cat_valid, num_valid)

dcn = DCN(total_categories, n_fields, n_num)
dcn_losses = train_neural(
    dcn, cat_train, num_train, y_train, train_weights, SEED + 2
)
dcn_valid = neural_predict(dcn, cat_valid, num_valid)

user_factors, video_factors = train_svd(train, rank=24)
svd_valid = svd_predict(valid, user_factors, video_factors)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)

families = {
    "nfm": nfm_valid,
    "dcn": dcn_valid,
    "latent_svd": svd_valid,
}

# A neural ensemble is also structurally distinct from either single network:
# it averages normalized bi-interaction and explicit-cross predictions.
nfm_norm, nfm_scale = center_scale(nfm_valid, valid.user_id)
dcn_norm, dcn_scale = center_scale(dcn_valid, valid.user_id)
families["nfm_dcn_ensemble"] = 0.5 * nfm_norm + 0.5 * dcn_norm

inc_norm, inc_scale = center_scale(inc_valid, valid.user_id)
alpha_grid = [0.10, 0.20, 0.30, 0.45, 0.60, 0.75]

candidate_scores = {
    "trusted_incumbent": primary(valid, y_valid, inc_valid)
}
family_scales = {}
family_blend_alpha = {}

best_name = "trusted_incumbent"
best_scores = inc_valid.copy()
best_primary = candidate_scores["trusted_incumbent"]
best_family = None
best_alpha = None
best_raw = None

best_own_name = None
best_own_primary = -np.inf
best_own_scores = None

for name, scores in families.items():
    p_raw = primary(valid, y_valid, scores)
    candidate_scores[name] = p_raw

    if p_raw > best_own_primary:
        best_own_primary = p_raw
        best_own_name = name
        best_own_scores = scores.copy()

    if p_raw > best_primary:
        best_primary = p_raw
        best_name = name
        best_scores = scores.copy()
        best_family = name
        best_alpha = None
        best_raw = scores.copy()

    norm, scale = center_scale(scores, valid.user_id)
    family_scales[name] = scale

    local_best_p = -np.inf
    local_best_alpha = None
    local_best_scores = None
    for alpha in alpha_grid:
        blend = alpha * norm + (1.0 - alpha) * inc_norm
        p = primary(valid, y_valid, blend)
        if p > local_best_p:
            local_best_p = p
            local_best_alpha = alpha
            local_best_scores = blend.copy()

    blend_name = name + "_incumbent_blend"
    candidate_scores[blend_name] = local_best_p
    family_blend_alpha[name] = local_best_alpha

    if local_best_p > best_primary:
        best_primary = local_best_p
        best_name = blend_name
        best_scores = local_best_scores
        best_family = name
        best_alpha = local_best_alpha
        best_raw = scores.copy()

metrics = evaluate(valid.user_id, y_valid, best_scores)

print(
    "FINDINGS nfm_losses=%s dcn_losses=%s winner=%s best_own=%s"
    % (
        ",".join("%.5f" % x for x in nfm_losses),
        ",".join("%.5f" % x for x in dcn_losses),
        best_name,
        best_own_name,
    )
)
print(
    "FINDINGS blend_alphas="
    + json.dumps(
        {k: float(v) for k, v in family_blend_alpha.items()},
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_scores.items()},
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_family is not None:
        raw_to_save = best_raw
    else:
        raw_to_save = best_own_scores
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(raw_to_save, dtype=np.float64),
    )

# All fitting and validation-based family/blend selection are complete before
# loading test. Test labels are never accessed.
test = load("test")
hist_test = load_histories("test")
cat_test, _ = make_cat_matrix(test)
num_test_raw = raw_numeric_matrix(test, hist_test)
num_test = apply_numeric_transform(num_test_raw, num_transform)
del num_test_raw, hist_test
gc.collect()

required_test = set()
if best_family is not None:
    if best_family == "nfm_dcn_ensemble":
        required_test.update(["nfm", "dcn"])
    else:
        required_test.add(best_family)

# If the incumbent itself wins, scores_test is the exact trusted incumbent.
if best_family is None:
    test_scores = np.load(
        os.path.join(shared, "incumbent_test_scores.npy")
    ).astype(np.float64)
else:
    test_components = {}

    if "nfm" in required_test:
        test_components["nfm"] = neural_predict(nfm, cat_test, num_test)
    if "dcn" in required_test:
        test_components["dcn"] = neural_predict(dcn, cat_test, num_test)
    if "latent_svd" in required_test:
        test_components["latent_svd"] = svd_predict(
            test, user_factors, video_factors
        )

    if best_family == "nfm_dcn_ensemble":
        tn, _ = center_scale(
            test_components["nfm"],
            test.user_id,
            scale=nfm_scale,
        )
        td, _ = center_scale(
            test_components["dcn"],
            test.user_id,
            scale=dcn_scale,
        )
        own_test = 0.5 * tn + 0.5 * td
    else:
        own_test = test_components[best_family]

    if best_alpha is None:
        test_scores = own_test
    else:
        own_norm, _ = center_scale(
            own_test,
            test.user_id,
            scale=family_scales[best_family],
        )
        inc_test = np.load(
            os.path.join(shared, "incumbent_test_scores.npy")
        ).astype(np.float64)
        inc_test_norm, _ = center_scale(
            inc_test,
            test.user_id,
            scale=inc_scale,
        )
        test_scores = (
            best_alpha * own_norm
            + (1.0 - best_alpha) * inc_test_norm
        )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
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