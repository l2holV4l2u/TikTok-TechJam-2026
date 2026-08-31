import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

FIELDS = list(FEATURE_CARDINALITIES.keys())
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
BATCH = 4096
PRED_BATCH = 16384

offsets = np.cumsum(
    np.asarray([0] + [FEATURE_CARDINALITIES[f] for f in FIELDS[:-1]],
               dtype=np.int64)
)
TOTAL_CATS = int(sum(FEATURE_CARDINALITIES[f] for f in FIELDS))


def make_x(split):
    cols = [
        np.asarray(split.X[f], dtype=np.int32) + np.int32(offsets[j])
        for j, f in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.int32)


def fit_numeric_transform(train):
    means = []
    stds = []
    for f in NUM_FIELDS:
        a = np.asarray(train.num[f], dtype=np.float64)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.sign(a) * np.log1p(np.abs(a))
        means.append(float(a.mean()))
        stds.append(float(max(a.std(), 1e-3)))
    return np.asarray(means, np.float32), np.asarray(stds, np.float32)


def make_num(split, means, stds):
    cols = []
    for j, f in enumerate(NUM_FIELDS):
        a = np.asarray(split.num[f], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.sign(a) * np.log1p(np.abs(a))
        cols.append((a - means[j]) / stds[j])
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.float32)


class WideLinear(nn.Module):
    def __init__(self, n_cats, n_num, bias):
        super().__init__()
        self.linear = nn.Embedding(n_cats, 1)
        self.num = nn.Linear(n_num, 1, bias=False)
        self.bias = nn.Parameter(torch.tensor(float(bias), dtype=torch.float32))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.num.weight)

    def forward(self, x, z):
        return (
            self.bias
            + self.linear(x).sum(dim=1).squeeze(-1)
            + self.num(z).squeeze(-1)
        )


class PairwiseFM(nn.Module):
    def __init__(self, n_cats, n_num, dim=12):
        super().__init__()
        self.linear = nn.Embedding(n_cats, 1)
        self.embedding = nn.Embedding(n_cats, dim)
        self.num_linear = nn.Linear(n_num, 1, bias=False)
        self.num_proj = nn.Linear(n_num, dim, bias=False)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.015)
        nn.init.zeros_(self.num_linear.weight)
        nn.init.normal_(self.num_proj.weight, std=0.01)

    def forward(self, x, z):
        e = self.embedding(x)
        ne = self.num_proj(z).unsqueeze(1)
        all_e = torch.cat([e, ne], dim=1)
        summed = all_e.sum(dim=1)
        inter = 0.5 * (
            summed.square() - all_e.square().sum(dim=1)
        ).sum(dim=1)
        return (
            self.linear(x).sum(dim=1).squeeze(-1)
            + self.num_linear(z).squeeze(-1)
            + inter
        )


class DeepCross(nn.Module):
    def __init__(self, n_cats, n_fields, n_num, bias, dim=4):
        super().__init__()
        self.embedding = nn.Embedding(n_cats, dim)
        self.wide = nn.Embedding(n_cats, 1)
        self.wide_num = nn.Linear(n_num, 1, bias=False)

        d = n_fields * dim + n_num
        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(d)) for _ in range(2)
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(d)) for _ in range(2)
        ])
        self.deep = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 1),
        )
        self.cross_out = nn.Linear(d, 1)
        self.bias = nn.Parameter(torch.tensor(float(bias), dtype=torch.float32))

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.wide.weight)
        nn.init.zeros_(self.wide_num.weight)
        for w in self.cross_w:
            nn.init.normal_(w, std=0.01)
        for layer in self.deep:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.cross_out.bias)

    def forward(self, x, z):
        x0 = torch.cat([self.embedding(x).flatten(1), z], dim=1)
        xc = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = (xc * w).sum(dim=1, keepdim=True)
            xc = x0 * scalar + b + xc
        wide = (
            self.wide(x).sum(dim=1).squeeze(-1)
            + self.wide_num(z).squeeze(-1)
        )
        return (
            self.bias
            + wide
            + self.cross_out(xc).squeeze(-1)
            + self.deep(x0).squeeze(-1)
        )


def train_pointwise(model, x_np, z_np, y_np, epochs, lr):
    x = torch.from_numpy(x_np)
    z = torch.from_numpy(z_np)
    y = torch.from_numpy(y_np)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(SEED + epochs)
    n = len(y)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for st in range(0, n, BATCH):
            ix = perm[st:st + BATCH]
            xb = x[ix].long()
            zb = z[ix]
            yb = y[ix]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, zb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


def make_pairs(user_ids, labels):
    order = np.lexsort((
        np.arange(len(user_ids), dtype=np.int64),
        np.asarray(user_ids, dtype=np.int64),
    ))
    su = np.asarray(user_ids, dtype=np.int64)[order]
    sy = np.asarray(labels, dtype=np.int8)[order]

    positives = []
    negatives = []
    for shift in (1, 2, 3, 5, 8, 13, 21):
        if shift >= len(order):
            continue
        left = np.arange(0, len(order) - shift, dtype=np.int64)
        right = left + shift
        mask = (su[left] == su[right]) & (sy[left] != sy[right])
        left = left[mask]
        right = right[mask]
        lp = sy[left] == 1
        positives.append(np.where(lp, order[left], order[right]))
        negatives.append(np.where(lp, order[right], order[left]))

    pos = np.concatenate(positives).astype(np.int64, copy=False)
    neg = np.concatenate(negatives).astype(np.int64, copy=False)

    rng = np.random.default_rng(SEED)
    max_pairs = 1800000
    if len(pos) > max_pairs:
        take = rng.choice(len(pos), max_pairs, replace=False)
        pos = pos[take]
        neg = neg[take]
    return pos, neg


def train_pairwise(model, x_np, z_np, pos, neg, epochs=3):
    x = torch.from_numpy(x_np)
    z = torch.from_numpy(z_np)
    pos_t = torch.from_numpy(pos)
    neg_t = torch.from_numpy(neg)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-6)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(SEED + 91)
    n = len(pos)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for st in range(0, n, BATCH):
            q = perm[st:st + BATCH]
            ip = pos_t[q]
            ineg = neg_t[q]
            opt.zero_grad(set_to_none=True)
            sp = model(x[ip].long(), z[ip])
            sn = model(x[ineg].long(), z[ineg])
            target = torch.ones_like(sp)
            loss = loss_fn(sp - sn, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


def predict_torch(model, x_np, z_np):
    model.eval()
    out = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for st in range(0, len(x_np), PRED_BATCH):
            en = min(st + PRED_BATCH, len(x_np))
            xb = torch.from_numpy(x_np[st:en]).long()
            zb = torch.from_numpy(z_np[st:en])
            out[st:en] = model(xb, zb).cpu().numpy()
    return out


def fit_svd(train, rank=32):
    users = np.asarray(train.user_id, dtype=np.int64)
    videos = np.asarray(train.video_id, dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)
    n_users = FEATURE_CARDINALITIES["user_id"]
    n_items = FEATURE_CARDINALITIES["video_id"]

    mask = labels == 1
    rows = users[mask]
    cols = videos[mask]
    vals = np.ones(mask.sum(), dtype=np.float32)
    mat = sp.coo_matrix(
        (vals, (rows, cols)), shape=(n_users, n_items)
    ).tocsr()
    mat.data[:] = 1.0

    user_deg = np.asarray(mat.sum(axis=1)).ravel()
    item_deg = np.asarray(mat.sum(axis=0)).ravel()
    ur = 1.0 / np.sqrt(np.maximum(user_deg, 1.0))
    ir = 1.0 / np.sqrt(np.maximum(item_deg, 1.0))
    norm = sp.diags(ur).dot(mat).dot(sp.diags(ir)).astype(np.float32)

    u, s, vt = svds(norm, k=rank, which="LM", random_state=SEED)
    idx = np.argsort(s)[::-1]
    s = s[idx]
    u = u[:, idx]
    vt = vt[idx]
    user_factors = (u * s[None, :]).astype(np.float32)
    item_factors = vt.T.astype(np.float32)
    return user_factors, item_factors


def svd_predict(split, uf, vf):
    u = np.asarray(split.user_id, dtype=np.int64)
    v = np.asarray(split.video_id, dtype=np.int64)
    ok = (
        (u >= 0) & (u < len(uf))
        & (v >= 0) & (v < len(vf))
    )
    out = np.zeros(len(u), dtype=np.float32)
    out[ok] = np.einsum(
        "ij,ij->i", uf[u[ok]], vf[v[ok]], optimize=True
    )
    return out


def metric(scores, valid):
    return evaluate(valid.user_id, valid.y, scores)


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)

x_train = make_x(train)
x_valid = make_x(valid)
num_mean, num_std = fit_numeric_transform(train)
z_train = make_num(train, num_mean, num_std)
z_valid = make_num(valid, num_mean, num_std)

rate = float(np.clip(y_train.mean(), 1e-6, 1.0 - 1e-6))
bias = float(np.log(rate / (1.0 - rate)))

# Family 1: wide additive categorical/numeric model.
wide = WideLinear(TOTAL_CATS, len(NUM_FIELDS), bias)
train_pointwise(wide, x_train, z_train, y_train, epochs=3, lr=1.5e-3)
wide_valid = predict_torch(wide, x_valid, z_valid)

# Family 2: explicit deep-and-cross network.
dcn = DeepCross(
    TOTAL_CATS, len(FIELDS), len(NUM_FIELDS), bias, dim=4
)
train_pointwise(dcn, x_train, z_train, y_train, epochs=3, lr=9e-4)
dcn_valid = predict_torch(dcn, x_valid, z_valid)

# Family 3: pairwise FM trained only on opposite-label impressions of the same user.
pair_pos, pair_neg = make_pairs(train.user_id, train.y)
pairfm = PairwiseFM(TOTAL_CATS, len(NUM_FIELDS), dim=12)
train_pairwise(pairfm, x_train, z_train, pair_pos, pair_neg, epochs=3)
pair_valid = predict_torch(pairfm, x_valid, z_valid)

# Family 4: latent positive user-video interaction factorization.
svd_u, svd_v = fit_svd(train, rank=32)
svd_train = svd_predict(train, svd_u, svd_v)
svd_valid_raw = svd_predict(valid, svd_u, svd_v)
svd_mu = float(svd_train.mean())
svd_sigma = float(max(svd_train.std(), 1e-6))
svd_valid = (svd_valid_raw - svd_mu) / svd_sigma

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)

families_valid = {
    "wide": wide_valid.astype(np.float64),
    "dcn": dcn_valid.astype(np.float64),
    "pairwise_fm": pair_valid.astype(np.float64),
    "latent_svd": svd_valid.astype(np.float64),
}

candidate_scores = {}
candidate_arrays = {}
candidate_meta = {}

inc_metrics = metric(inc_valid, valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
candidate_arrays["incumbent"] = inc_valid
candidate_meta["incumbent"] = ("incumbent", 0.0)

for name, pred in families_valid.items():
    m = metric(pred, valid)
    candidate_scores[name] = float(m["primary"])
    candidate_arrays[name] = pred
    candidate_meta[name] = (name, 1.0)

    # The trusted-incumbent note explicitly permits validation selection
    # of a blend weight, applied unchanged to test.
    for alpha in (0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80):
        blended = (1.0 - alpha) * inc_valid + alpha * pred
        bm = metric(blended, valid)
        key = f"incumbent+{name}@{alpha:.2f}"
        candidate_scores[key] = float(bm["primary"])
        candidate_arrays[key] = blended
        candidate_meta[key] = (name, alpha)

# Also test a conservative three-family aggregate before blending it.
ensemble_valid = (
    0.50 * dcn_valid.astype(np.float64)
    + 0.35 * pair_valid.astype(np.float64)
    + 0.15 * wide_valid.astype(np.float64)
)
em = metric(ensemble_valid, valid)
candidate_scores["deep_pair_wide"] = float(em["primary"])
candidate_arrays["deep_pair_wide"] = ensemble_valid
candidate_meta["deep_pair_wide"] = ("ensemble", 1.0)

for alpha in (0.10, 0.20, 0.30, 0.40, 0.50, 0.65):
    blended = (1.0 - alpha) * inc_valid + alpha * ensemble_valid
    bm = metric(blended, valid)
    key = f"incumbent+deep_pair_wide@{alpha:.2f}"
    candidate_scores[key] = float(bm["primary"])
    candidate_arrays[key] = blended
    candidate_meta[key] = ("ensemble", alpha)

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_arrays[winner]
winner_metrics = metric(valid_scores, valid)

print("FINDINGS " + json.dumps({
    "pair_count": int(len(pair_pos)),
    "winner": winner,
    "wide_primary": candidate_scores["wide"],
    "dcn_primary": candidate_scores["dcn"],
    "pairwise_fm_primary": candidate_scores["pairwise_fm"],
    "latent_svd_primary": candidate_scores["latent_svd"],
}, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    family_name, alpha = candidate_meta[winner]
    if alpha < 1.0:
        if family_name == "ensemble":
            raw = ensemble_valid
        elif family_name == "incumbent":
            raw = inc_valid
        else:
            raw = families_valid[family_name]
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(raw, dtype=np.float64),
        )

# Score test only after all training and validation-side selection are complete.
test = load("test")
x_test = make_x(test)
z_test = make_num(test, num_mean, num_std)

family_name, alpha = candidate_meta[winner]
inc_test = np.load(inc_test_path).astype(np.float64)

if family_name == "incumbent":
    own_test = inc_test
elif family_name == "wide":
    own_test = predict_torch(wide, x_test, z_test).astype(np.float64)
elif family_name == "dcn":
    own_test = predict_torch(dcn, x_test, z_test).astype(np.float64)
elif family_name == "pairwise_fm":
    own_test = predict_torch(pairfm, x_test, z_test).astype(np.float64)
elif family_name == "latent_svd":
    raw = svd_predict(test, svd_u, svd_v)
    own_test = ((raw - svd_mu) / svd_sigma).astype(np.float64)
elif family_name == "ensemble":
    wt = predict_torch(wide, x_test, z_test).astype(np.float64)
    dt = predict_torch(dcn, x_test, z_test).astype(np.float64)
    pt = predict_torch(pairfm, x_test, z_test).astype(np.float64)
    own_test = 0.50 * dt + 0.35 * pt + 0.15 * wt
else:
    raise RuntimeError("Unknown winning family")

if family_name == "incumbent":
    test_scores = inc_test
elif alpha < 1.0:
    test_scores = (1.0 - alpha) * inc_test + alpha * own_test
else:
    test_scores = own_test

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(winner_metrics["primary"]),
    "gauc": float(winner_metrics["gauc"]),
    "ndcg@5": float(winner_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))