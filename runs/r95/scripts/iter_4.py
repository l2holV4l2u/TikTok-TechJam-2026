import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
GLOBAL_RATE = float(y_train.mean())
EPS = 1e-5


def logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


GLOBAL_LOGIT = float(logit(GLOBAL_RATE))


# ----------------------------------------------------------------------
# Family 1: empirical-Bayes additive target statistics.
# Each categorical effect is shrunk strongly toward the global rate.
# User ID is intentionally omitted because it is constant within user.
# ----------------------------------------------------------------------
EB_SPECS = [
    ("video_id", 24.0, 0.45),
    ("author_id", 35.0, 0.27),
    ("tag", 80.0, 0.18),
    ("tab", 100.0, 0.12),
    ("duration_bucket", 100.0, 0.12),
    ("upload_type", 100.0, 0.08),
    ("onehot_feat3", 80.0, 0.10),
    ("onehot_feat7", 100.0, 0.08),
    ("onehot_feat8", 80.0, 0.10),
]

eb_tables = {}

for name, strength, weight in EB_SPECS:
    x = np.asarray(train.X[name], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[name])
    cnt = np.bincount(x, minlength=card).astype(np.float64)
    pos = np.bincount(x, weights=y_train, minlength=card).astype(np.float64)
    rate = (pos + strength * GLOBAL_RATE) / (cnt + strength)
    effect = logit(rate) - GLOBAL_LOGIT
    effect[cnt == 0] = 0.0
    eb_tables[name] = (effect.astype(np.float32), float(weight))


def predict_eb(split):
    result = np.full(len(split.user_id), GLOBAL_LOGIT, dtype=np.float32)
    for name, (effect, weight) in eb_tables.items():
        ids = np.asarray(split.X[name], dtype=np.int64)
        result += weight * effect[ids]
    return result


eb_valid = predict_eb(valid)


# ----------------------------------------------------------------------
# Family 2: explicit low-rank logistic matrix factorization.
# This forms predictions only from user/item latent affinity and biases,
# unlike the incumbent's all-pairs five-field FM.
# ----------------------------------------------------------------------
class LogisticMF(nn.Module):
    def __init__(self, n_users, n_items, rank=24):
        super().__init__()
        self.user_vec = nn.Embedding(n_users, rank)
        self.item_vec = nn.Embedding(n_items, rank)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor([GLOBAL_LOGIT], dtype=torch.float32))

        nn.init.normal_(self.user_vec.weight, std=0.025)
        nn.init.normal_(self.item_vec.weight, std=0.025)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        uv = self.user_vec(users)
        iv = self.item_vec(items)
        interaction = (uv * iv).sum(dim=1)
        return (
            interaction
            + self.user_bias(users).squeeze(1)
            + self.item_bias(items).squeeze(1)
            + self.global_bias
        )


u_train_np = np.asarray(train.X["user_id"], dtype=np.int64)
v_train_np = np.asarray(train.X["video_id"], dtype=np.int64)
u_train = torch.from_numpy(u_train_np)
v_train = torch.from_numpy(v_train_np)
yt = torch.from_numpy(y_train)

mf = LogisticMF(
    int(FEATURE_CARDINALITIES["user_id"]),
    int(FEATURE_CARDINALITIES["video_id"]),
    rank=24,
)

mf_opt = torch.optim.AdamW(mf.parameters(), lr=0.006, weight_decay=2e-5)
mf_loss = nn.BCEWithLogitsLoss()
gen = torch.Generator()
gen.manual_seed(SEED)
n_train = len(y_train)
batch_size = 16384

mf.train()
for epoch in range(5):
    perm = torch.randperm(n_train, generator=gen)
    loss_sum = 0.0
    seen = 0
    for st in range(0, n_train, batch_size):
        idx = perm[st:min(st + batch_size, n_train)]
        ub = u_train.index_select(0, idx)
        vb = v_train.index_select(0, idx)
        yb = yt.index_select(0, idx)

        mf_opt.zero_grad(set_to_none=True)
        pred = mf(ub, vb)
        loss = mf_loss(pred, yb)
        loss.backward()
        mf_opt.step()

        bn = len(idx)
        loss_sum += float(loss.detach()) * bn
        seen += bn
    print("mf_epoch=%d loss=%.6f" % (epoch + 1, loss_sum / seen), flush=True)


def predict_mf(split, batch=65536):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    items = np.asarray(split.X["video_id"], dtype=np.int64)
    out = np.empty(len(users), dtype=np.float32)
    mf.eval()
    with torch.inference_mode():
        for st in range(0, len(users), batch):
            en = min(st + batch, len(users))
            ub = torch.from_numpy(users[st:en])
            vb = torch.from_numpy(items[st:en])
            out[st:en] = mf(ub, vb).cpu().numpy()
    return out


mf_valid = predict_mf(valid)


# ----------------------------------------------------------------------
# Family 3: sequential non-parametric transition model.
# Learn P(long_view for current video | immediately previous displayed
# video) on train. At inference, only earlier impression identities from
# the same split are used; no validation/test outcomes are accessed.
# ----------------------------------------------------------------------
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])
train_video = np.asarray(train.X["video_id"], dtype=np.int64)
train_user = np.asarray(train.user_id, dtype=np.int64)
train_time = np.asarray(train.time_ms, dtype=np.int64)
train_row = np.arange(len(train_user), dtype=np.int64)

order = np.lexsort((train_row, train_time, train_user))
ou = train_user[order]
ov = train_video[order]
oy = y_train[order]

same_prev = np.zeros(len(order), dtype=bool)
same_prev[1:] = ou[1:] == ou[:-1]
cur_pos = np.flatnonzero(same_prev)
prev_video = ov[cur_pos - 1]
cur_video = ov[cur_pos]
pair_keys = prev_video.astype(np.int64) * VIDEO_CARD + cur_video.astype(np.int64)
pair_y = oy[cur_pos].astype(np.float64)

uniq_keys, inverse = np.unique(pair_keys, return_inverse=True)
pair_cnt = np.bincount(inverse).astype(np.float64)
pair_pos = np.bincount(inverse, weights=pair_y).astype(np.float64)

video_cnt = np.bincount(train_video, minlength=VIDEO_CARD).astype(np.float64)
video_pos = np.bincount(
    train_video, weights=y_train, minlength=VIDEO_CARD
).astype(np.float64)
video_rate = (video_pos + 25.0 * GLOBAL_RATE) / (video_cnt + 25.0)
video_logit = logit(video_rate).astype(np.float32)


def predict_transition(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    rows = np.arange(len(users), dtype=np.int64)

    result = video_logit[videos].astype(np.float32, copy=True)

    ordx = np.lexsort((rows, times, users))
    su = users[ordx]
    sv = videos[ordx]

    has_prev = np.zeros(len(ordx), dtype=bool)
    has_prev[1:] = su[1:] == su[:-1]
    loc = np.flatnonzero(has_prev)
    if len(loc) == 0:
        return result

    pv = sv[loc - 1]
    cv = sv[loc]
    keys = pv.astype(np.int64) * VIDEO_CARD + cv.astype(np.int64)

    where = np.searchsorted(uniq_keys, keys)
    clipped = np.minimum(where, len(uniq_keys) - 1)
    found = (where < len(uniq_keys)) & (uniq_keys[clipped] == keys)

    rates = video_rate[cv].astype(np.float64, copy=True)
    if found.any():
        pidx = where[found]
        support = pair_cnt[pidx]
        # Shrink pair estimates toward the current video's train prior.
        rates[found] = (
            pair_pos[pidx] + 10.0 * video_rate[cv[found]]
        ) / (support + 10.0)

    sequence_scores = logit(rates).astype(np.float32)
    original_rows = ordx[loc]
    result[original_rows] = sequence_scores
    return result


transition_valid = predict_transition(valid)


# ----------------------------------------------------------------------
# Validate standalone families and all blends with trusted incumbent.
# The same selected alpha is later applied to test.
# ----------------------------------------------------------------------
shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float32)

families_valid = {
    "empirical_bayes": eb_valid,
    "latent_mf": mf_valid,
    "video_transition": transition_valid,
}

candidate_log = {}
best_primary = -1.0
best_name = None
best_alpha = None
best_valid = None
best_raw_valid = None
best_metrics = None

alphas = [0.0, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.0]

for name, raw_scores in families_valid.items():
    raw_metrics = evaluate(valid.user_id, valid.y, raw_scores)
    candidate_log[name + "_raw"] = float(raw_metrics["primary"])

    family_best = -1.0
    family_best_alpha = None
    for alpha in alphas:
        blended = (1.0 - alpha) * inc_valid + alpha * raw_scores
        metrics = evaluate(valid.user_id, valid.y, blended)
        primary = float(metrics["primary"])
        candidate_log[name + "_blend_%.2f" % alpha] = primary

        if primary > family_best:
            family_best = primary
            family_best_alpha = alpha

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_alpha = float(alpha)
            best_valid = np.asarray(blended, dtype=np.float32)
            best_raw_valid = np.asarray(raw_scores, dtype=np.float32)
            best_metrics = metrics

    print(
        "FINDINGS family=%s raw=%.6f best_blend=%.6f alpha=%.2f"
        % (name, raw_metrics["primary"], family_best, family_best_alpha),
        flush=True,
    )

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True), flush=True)
print(
    "FINDINGS winner=%s alpha=%.2f primary=%.6f"
    % (best_name, best_alpha, best_primary),
    flush=True,
)


# ----------------------------------------------------------------------
# Produce test predictions with the already selected family and alpha.
# ----------------------------------------------------------------------
test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float32)

if best_name == "empirical_bayes":
    best_raw_test = predict_eb(test)
elif best_name == "latent_mf":
    best_raw_test = predict_mf(test)
elif best_name == "video_transition":
    best_raw_test = predict_transition(test)
else:
    raise RuntimeError("Unknown winning family")

best_test = (1.0 - best_alpha) * inc_test + best_alpha * best_raw_test

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)