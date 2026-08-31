import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load
from pipeline.evaluate import evaluate

START = time.time()
SEED = 7319
BATCH = 8192
EPOCHS = 4
PRED_BATCH = 65536

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))


def sequence_features(s):
    """Safe features determined solely by the impressions in the scored split."""
    user = np.asarray(s.user_id, dtype=np.int64)
    tm = np.asarray(s.time_ms, dtype=np.int64)
    n = user.size
    row = np.arange(n, dtype=np.int64)

    # Ties follow original row position, as specified by the API.
    order = np.lexsort((row, tm, user))
    us = user[order]
    ts = tm[order]

    new_user = np.empty(n, dtype=bool)
    new_user[0] = True
    new_user[1:] = us[1:] != us[:-1]
    starts = np.flatnonzero(new_user)
    counts = np.diff(np.r_[starts, n])
    start_for_row = np.repeat(starts, counts)

    pos_sorted = np.arange(n, dtype=np.int64) - start_for_row
    user_n_sorted = np.repeat(counts, counts)
    rev_sorted = user_n_sorted - 1 - pos_sorted

    new_batch = np.empty(n, dtype=bool)
    new_batch[0] = True
    new_batch[1:] = (us[1:] != us[:-1]) | (ts[1:] != ts[:-1])
    batch_starts = np.flatnonzero(new_batch)
    batch_counts = np.diff(np.r_[batch_starts, n])
    batch_start_for_row = np.repeat(batch_starts, batch_counts)
    batch_pos_sorted = np.arange(n, dtype=np.int64) - batch_start_for_row
    batch_n_sorted = np.repeat(batch_counts, batch_counts)

    prev_gap = np.zeros(n, dtype=np.float64)
    next_gap = np.zeros(n, dtype=np.float64)
    if n > 1:
        d = np.maximum(ts[1:] - ts[:-1], 0)
        same = us[1:] == us[:-1]
        prev_gap[1:] = np.where(same, d, 0)
        next_gap[:-1] = np.where(same, d, 0)

    def unsort(a):
        out = np.empty_like(a)
        out[order] = a
        return out

    pos = unsort(pos_sorted).astype(np.float32)
    rev = unsort(rev_sorted).astype(np.float32)
    user_n = unsort(user_n_sorted).astype(np.float32)
    batch_pos = unsort(batch_pos_sorted).astype(np.float32)
    batch_n = unsort(batch_n_sorted).astype(np.float32)
    pgap = unsort(prev_gap).astype(np.float32)
    ngap = unsort(next_gap).astype(np.float32)

    denom = np.maximum(user_n - 1.0, 1.0)
    frac = pos / denom
    reverse_frac = rev / denom
    bfrac = batch_pos / np.maximum(batch_n - 1.0, 1.0)

    hour = np.asarray(s.X["hour"], dtype=np.float32)
    tab = np.asarray(s.X["tab"], dtype=np.int64)
    duration = np.asarray(s.num["duration_ms"], dtype=np.float32)
    duration = np.nan_to_num(duration, nan=0.0, posinf=0.0, neginf=0.0)
    log_duration = np.log1p(np.maximum(duration, 0.0))

    hour_angle = 2.0 * np.pi * hour / 24.0
    x = np.column_stack([
        frac,
        reverse_frac,
        frac * frac,
        frac * frac * frac,
        np.sqrt(np.maximum(frac, 0.0)),
        np.log1p(pos),
        np.log1p(rev),
        np.log1p(user_n),
        np.log1p(batch_n),
        bfrac,
        np.log1p(np.maximum(pgap, 0.0)),
        np.log1p(np.maximum(ngap, 0.0)),
        np.sin(hour_angle),
        np.cos(hour_angle),
        log_duration,
        (batch_n > 1).astype(np.float32),
        (pos == 0).astype(np.float32),
        (rev == 0).astype(np.float32),
    ]).astype(np.float32)

    # Discrete representation shared by the non-parametric family.
    pos_bin = np.minimum((frac * 8.0).astype(np.int64), 7)
    n_bin = np.digitize(
        user_n, np.asarray([1.5, 2.5, 4.5, 7.5, 15.5], dtype=np.float32)
    ).astype(np.int64)
    hour_bin = (hour.astype(np.int64) // 4).clip(0, 5)
    tab_bin = np.clip(tab, 0, 19)
    batch_bin = np.minimum(batch_n.astype(np.int64) - 1, 3).clip(0, 3)

    discrete = np.column_stack(
        [pos_bin, n_bin, hour_bin, tab_bin, batch_bin]
    ).astype(np.int64)
    return x, discrete


def fit_scaler(x):
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def apply_scaler(x, mean, std):
    return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)


class LinearContext(nn.Module):
    def __init__(self, d, base_rate):
        super().__init__()
        self.linear = nn.Linear(d, 1)
        nn.init.zeros_(self.linear.weight)
        p = float(np.clip(base_rate, 1e-5, 1 - 1e-5))
        nn.init.constant_(self.linear.bias, np.log(p / (1 - p)))

    def forward(self, x):
        return self.linear(x).squeeze(1)


class ContextMLP(nn.Module):
    def __init__(self, d, base_rate):
        super().__init__()
        p = float(np.clip(base_rate, 1e-5, 1 - 1e-5))
        self.net = nn.Sequential(
            nn.Linear(d, 48),
            nn.SiLU(),
            nn.Linear(48, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )
        for mod in self.net:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)
        nn.init.constant_(self.net[-1].bias, np.log(p / (1 - p)))

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_neural(x, y, family, epochs=EPOCHS):
    torch.manual_seed(SEED + (0 if family == "linear" else 101))
    if family == "linear":
        model = LinearContext(x.shape[1], float(y.mean()))
        lr = 0.012
        wd = 2e-5
    else:
        model = ContextMLP(x.shape[1], float(y.mean()))
        lr = 0.0025
        wd = 1e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.float32, copy=False))
    gen = torch.Generator()
    gen.manual_seed(SEED + 17)

    n = len(y)
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        model.train()
        for st in range(0, n, BATCH):
            idx = order[st:min(st + BATCH, n)]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_neural(model, x):
    model.eval()
    out = np.empty(x.shape[0], dtype=np.float32)
    for st in range(0, x.shape[0], PRED_BATCH):
        en = min(st + PRED_BATCH, x.shape[0])
        out[st:en] = model(torch.from_numpy(x[st:en])).cpu().numpy()
    return out


HIST_DIMS = (8, 6, 6, 20, 4)


def flat_key(d):
    return np.ravel_multi_index(
        tuple(d[:, j] for j in range(d.shape[1])), HIST_DIMS
    )


def fit_histogram(d, y):
    y = y.astype(np.float64, copy=False)
    global_rate = float(y.mean())
    joint_key = flat_key(d)
    joint_size = int(np.prod(HIST_DIMS))
    jc = np.bincount(joint_key, minlength=joint_size).astype(np.float64)
    jp = np.bincount(joint_key, weights=y, minlength=joint_size)
    joint = (jp + 35.0 * global_rate) / (jc + 35.0)

    marginals = []
    for j, size in enumerate(HIST_DIMS):
        c = np.bincount(d[:, j], minlength=size).astype(np.float64)
        p = np.bincount(d[:, j], weights=y, minlength=size)
        marginals.append(((p + 80.0 * global_rate) / (c + 80.0)).astype(np.float32))

    return {
        "global": global_rate,
        "joint": joint.astype(np.float32),
        "marginals": marginals,
    }


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def predict_histogram(model, d):
    base = logit(model["global"])
    score = 0.58 * logit(model["joint"][flat_key(d)])
    weights = [0.16, 0.09, 0.05, 0.08, 0.04]
    for j, w in enumerate(weights):
        score += w * logit(model["marginals"][j][d[:, j]])
    score -= (0.58 + sum(weights) - 1.0) * base
    return np.asarray(score, dtype=np.float32)


def standardize_scores(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd < 1e-12:
        sd = 1.0
    return (x - float(x.mean())) / sd


train = load("train")
valid = load("valid")
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

train_raw, train_disc = sequence_features(train)
valid_raw, valid_disc = sequence_features(valid)
mean, std = fit_scaler(train_raw)
train_x = apply_scaler(train_raw, mean, std)
valid_x = apply_scaler(valid_raw, mean, std)

hist_model = fit_histogram(train_disc, train_y)
hist_valid = predict_histogram(hist_model, valid_disc)

linear_model = train_neural(train_x, train_y, "linear")
linear_valid = predict_neural(linear_model, valid_x)

mlp_model = train_neural(train_x, train_y, "mlp")
mlp_valid = predict_neural(mlp_model, valid_x)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(os.path.join(shared, "incumbent_valid_scores.npy"))
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_vz = standardize_scores(inc_valid)

families = {
    "position_histogram": hist_valid,
    "position_linear": linear_valid,
    "position_mlp": mlp_valid,
}

candidate_scores = {}
best_primary = -np.inf
best_name = None
best_family = None
best_weight = 0.0
best_valid_scores = None
best_metrics = None

for name, scores in families.items():
    met = evaluate(valid_users, valid_y, scores)
    candidate_scores[name] = float(met["primary"])
    if float(met["primary"]) > best_primary:
        best_primary = float(met["primary"])
        best_name = name
        best_family = name
        best_weight = 0.0
        best_valid_scores = scores.copy()
        best_metrics = met

    own_z = standardize_scores(scores)
    for w in (0.08, 0.14, 0.20, 0.28, 0.36, 0.45):
        blended = (1.0 - w) * inc_vz + w * own_z
        met_b = evaluate(valid_users, valid_y, blended)
        cname = "%s_blend_%.2f" % (name, w)
        candidate_scores[cname] = float(met_b["primary"])
        if float(met_b["primary"]) > best_primary:
            best_primary = float(met_b["primary"])
            best_name = cname
            best_family = name
            best_weight = float(w)
            best_valid_scores = blended.copy()
            best_metrics = met_b

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True), flush=True)
print(
    "FINDINGS selected=%s family=%s incumbent_weight=%.2f"
    % (best_name, best_family, 1.0 - best_weight),
    flush=True,
)

# Refit the selected positional family on train + validation.
combined_y = np.concatenate(
    [train_y, valid_y.astype(np.float32, copy=False)]
)
combined_raw = np.concatenate([train_raw, valid_raw], axis=0)
combined_disc = np.concatenate([train_disc, valid_disc], axis=0)

test = load("test")
test_raw, test_disc = sequence_features(test)

if best_family == "position_histogram":
    final_model = fit_histogram(combined_disc, combined_y)
    own_test = predict_histogram(final_model, test_disc)
elif best_family == "position_linear":
    cmean, cstd = fit_scaler(combined_raw)
    combined_x = apply_scaler(combined_raw, cmean, cstd)
    test_x = apply_scaler(test_raw, cmean, cstd)
    final_model = train_neural(combined_x, combined_y, "linear")
    own_test = predict_neural(final_model, test_x)
else:
    cmean, cstd = fit_scaler(combined_raw)
    combined_x = apply_scaler(combined_raw, cmean, cstd)
    test_x = apply_scaler(test_raw, cmean, cstd)
    final_model = train_neural(combined_x, combined_y, "mlp")
    own_test = predict_neural(final_model, test_x)

if best_weight > 0:
    inc_test = np.load(inc_test_path)
    test_scores = (
        (1.0 - best_weight) * standardize_scores(inc_test)
        + best_weight * standardize_scores(own_test)
    )
    own_valid_selected = families[best_family]
else:
    test_scores = own_test
    own_valid_selected = best_valid_scores

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if best_weight > 0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(own_valid_selected, dtype=np.float64),
        )

elapsed = float(time.time() - START)
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))