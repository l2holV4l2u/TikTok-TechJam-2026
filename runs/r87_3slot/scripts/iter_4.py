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
SEED = 7319
BATCH = 8192
EPOCHS = 3
FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
]
EMBED_DIM = 12
HIST_LEN = 8

torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_all(SEED)

cards = np.asarray([int(FEATURE_CARDINALITIES[f]) for f in FIELDS], dtype=np.int64)
offsets = np.cumsum(np.r_[0, cards[:-1]]).astype(np.int64)
total_categories = int(cards.sum())
uid_card = int(FEATURE_CARDINALITIES["user_id"])
video_offset = int(offsets[FIELDS.index("video_id")])


def concatenate_array(parts, getter, dtype):
    if len(parts) == 1:
        return np.asarray(getter(parts[0]), dtype=dtype)
    return np.concatenate(
        [np.asarray(getter(p), dtype=dtype) for p in parts]
    ).astype(dtype, copy=False)


def make_x(parts):
    cols = []
    for field, off in zip(FIELDS, offsets):
        col = concatenate_array(parts, lambda p, f=field: p.X[f], np.int64)
        cols.append(col + off)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


def build_causal_histories(parts, labels):
    """
    For every fitting row, use only earlier positive impressions of that user.
    Also return a compact terminal positive-history state for future splits.
    """
    uid = concatenate_array(parts, lambda p: p.X["user_id"], np.int64)
    vid = concatenate_array(parts, lambda p: p.X["video_id"], np.int64)
    tm = concatenate_array(parts, lambda p: p.time_ms, np.int64)
    y = np.asarray(labels, dtype=np.int8)
    row = np.arange(len(uid), dtype=np.int64)

    order = np.lexsort((row, tm, uid))
    su = uid[order]
    sv = vid[order]
    sy = y[order]

    starts_mask = np.r_[True, su[1:] != su[:-1]]
    group_starts = np.flatnonzero(starts_mask)
    group_sizes = np.diff(np.r_[group_starts, len(su)])

    global_cs = np.cumsum(sy, dtype=np.int64)
    base_at_start = global_cs[group_starts] - sy[group_starts]
    bases = np.repeat(base_at_start, group_sizes)
    positive_rank_before = global_cs - sy - bases

    positive_videos = sv[sy == 1]
    positive_counts = np.bincount(
        su[sy == 1], minlength=uid_card
    ).astype(np.int64)
    positive_starts = np.zeros(uid_card + 1, dtype=np.int64)
    np.cumsum(positive_counts, out=positive_starts[1:])

    sorted_hist = np.zeros((len(uid), HIST_LEN), dtype=np.int64)
    user_pos_start = positive_starts[su]
    for lag in range(1, HIST_LEN + 1):
        rank = positive_rank_before - lag
        ok = rank >= 0
        lookup = user_pos_start[ok] + rank[ok]
        sorted_hist[ok, lag - 1] = positive_videos[lookup]

    hist = np.empty_like(sorted_hist)
    hist[order] = sorted_hist
    state = (positive_videos, positive_starts, positive_counts)
    return hist, state


def histories_from_state(part, state):
    positive_videos, positive_starts, positive_counts = state
    uid = np.asarray(part.X["user_id"], dtype=np.int64)
    result = np.zeros((len(uid), HIST_LEN), dtype=np.int64)
    counts = positive_counts[uid]
    starts = positive_starts[uid]
    for lag in range(1, HIST_LEN + 1):
        rank = counts - lag
        ok = rank >= 0
        result[ok, lag - 1] = positive_videos[starts[ok] + rank[ok]]
    return result


class DINModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(total_categories, EMBED_DIM)
        self.net = nn.Sequential(
            nn.Linear(len(FIELDS) * EMBED_DIM + 2 * EMBED_DIM, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        nn.init.normal_(self.emb.weight, std=0.025)

    def forward(self, x, hist):
        fields = self.emb(x)
        candidate = fields[:, FIELDS.index("video_id"), :]
        hist_global = hist + video_offset
        h = self.emb(hist_global)
        mask = hist.ne(0)
        att = (h * candidate[:, None, :]).sum(dim=2) / (EMBED_DIM ** 0.5)
        att = att.masked_fill(~mask, -1e4)
        weights = torch.softmax(att, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (h * weights[:, :, None]).sum(dim=1)
        z = torch.cat(
            [fields.flatten(1), pooled, pooled * candidate], dim=1
        )
        return self.net(z).squeeze(1)


class MMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(total_categories, EMBED_DIM)
        width = len(FIELDS) * EMBED_DIM
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(width, 56), nn.ReLU(),
                          nn.Linear(56, 40), nn.ReLU())
            for _ in range(3)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(width, 3) for _ in range(3)
        ])
        self.heads = nn.ModuleList([nn.Linear(40, 1) for _ in range(3)])
        nn.init.normal_(self.emb.weight, std=0.025)

    def forward(self, x, hist):
        z = self.emb(x).flatten(1)
        experts = torch.stack([e(z) for e in self.experts], dim=1)
        outputs = []
        for gate, head in zip(self.gates, self.heads):
            w = torch.softmax(gate(z), dim=1)
            mixed = (experts * w[:, :, None]).sum(dim=1)
            outputs.append(head(mixed).squeeze(1))
        return tuple(outputs)


class HistoryLatentModel(nn.Module):
    def __init__(self):
        super().__init__()
        d = 24
        self.user = nn.Embedding(uid_card, d)
        self.video = nn.Embedding(int(FEATURE_CARDINALITIES["video_id"]), d)
        self.user_bias = nn.Embedding(uid_card, 1)
        self.video_bias = nn.Embedding(
            int(FEATURE_CARDINALITIES["video_id"]), 1
        )
        self.context_bias = nn.Embedding(total_categories, 1)
        nn.init.normal_(self.user.weight, std=0.03)
        nn.init.normal_(self.video.weight, std=0.03)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.video_bias.weight)
        nn.init.zeros_(self.context_bias.weight)

    def forward(self, x, hist):
        uid = x[:, FIELDS.index("user_id")] - offsets[FIELDS.index("user_id")]
        vid = x[:, FIELDS.index("video_id")] - offsets[FIELDS.index("video_id")]
        u = self.user(uid)
        v = self.video(vid)

        hv = self.video(hist)
        mask = hist.ne(0).float()
        pooled = (hv * mask[:, :, None]).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        affinity = (u * v).sum(dim=1) / (u.shape[1] ** 0.5)
        sequential = (pooled * v).sum(dim=1) / (u.shape[1] ** 0.5)
        bias = self.user_bias(uid).squeeze(1) + self.video_bias(vid).squeeze(1)
        bias = bias + self.context_bias(x).sum(dim=1).squeeze(1)
        return affinity + 0.75 * sequential + bias


MODEL_CLASSES = {
    "din_sequence": DINModel,
    "mmoe_multitask": MMoEModel,
    "history_latent": HistoryLatentModel,
}


def predict(model, x_np, h_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(x_np), BATCH * 2):
            end = min(start + BATCH * 2, len(x_np))
            xb = torch.from_numpy(x_np[start:end])
            hb = torch.from_numpy(h_np[start:end])
            out = model(xb, hb)
            if isinstance(out, tuple):
                out = out[0]
            result[start:end] = out.cpu().numpy().astype(np.float64)
    return result


def fit_family(name, x_np, hist_np, y_np, aux_np, epochs,
               validation=None, seed=SEED):
    seed_all(seed)
    model = MODEL_CLASSES[name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    bce = nn.BCEWithLogitsLoss()

    x = torch.from_numpy(x_np)
    h = torch.from_numpy(hist_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    aux = None
    if aux_np is not None:
        aux = torch.from_numpy(np.asarray(aux_np, dtype=np.float32))

    best_primary = -np.inf
    best_epoch = epochs
    best_scores = None
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + 101 * epoch)
        order = torch.randperm(len(x_np), generator=generator)

        for start in range(0, len(x_np), BATCH):
            idx = order[start:start + BATCH]
            optimizer.zero_grad(set_to_none=True)
            output = model(x[idx], h[idx])

            if name == "mmoe_multitask":
                long_logit, click_logit, like_logit = output
                loss = bce(long_logit, y[idx])
                loss = loss + 0.20 * bce(click_logit, aux[idx, 0])
                loss = loss + 0.12 * bce(like_logit, aux[idx, 1])
            else:
                loss = bce(output, y[idx])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if validation is not None:
            vx, vh, vu, vy = validation
            scores = predict(model, vx, vh)
            met = evaluate(vu, vy, scores)
            print(
                "%s epoch=%d primary=%.6f gauc=%.6f ndcg5=%.6f"
                % (name, epoch, met["primary"], met["gauc"], met["ndcg@5"]),
                flush=True,
            )
            if met["primary"] > best_primary:
                best_primary = float(met["primary"])
                best_epoch = epoch
                best_scores = scores.copy()
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }

    if validation is not None:
        model.load_state_dict(best_state)
        return model, best_epoch, best_scores
    return model


def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / max(a.std(), 1e-8)


train = load("train")
valid = load("valid")

x_train = make_x([train])
x_valid = make_x([valid])
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

hist_train, train_state = build_causal_histories([train], y_train)
hist_valid = histories_from_state(valid, train_state)

aux_train = np.column_stack([
    np.asarray(train.aux["is_click"], dtype=np.float32),
    np.asarray(train.aux["is_like"], dtype=np.float32),
])

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(os.path.join(shared, "incumbent_valid_scores.npy"))
inc_valid_z = zscore(inc_valid)

records = {}
candidate_scores = {}
candidate_metrics = {}

for family_index, family in enumerate(MODEL_CLASSES):
    model, best_epoch, raw_valid = fit_family(
        family,
        x_train,
        hist_train,
        y_train,
        aux_train,
        EPOCHS,
        validation=(x_valid, hist_valid, valid.user_id, y_valid),
        seed=SEED + 1000 * family_index,
    )

    raw_metrics = evaluate(valid.user_id, y_valid, raw_valid)
    candidate_scores[family] = float(raw_metrics["primary"])

    own_z = zscore(raw_valid)
    best_blend_primary = -np.inf
    best_weight = 1.0
    best_blend = raw_valid
    best_blend_metrics = raw_metrics

    for own_weight in np.linspace(0.0, 1.0, 11):
        blended = own_weight * own_z + (1.0 - own_weight) * inc_valid_z
        met = evaluate(valid.user_id, y_valid, blended)
        if met["primary"] > best_blend_primary:
            best_blend_primary = float(met["primary"])
            best_weight = float(own_weight)
            best_blend = blended.copy()
            best_blend_metrics = met

    candidate_scores[family + "_blend"] = best_blend_primary
    records[family] = {
        "epoch": best_epoch,
        "raw_valid": raw_valid,
        "blend_valid": best_blend,
        "weight": best_weight,
        "metrics": best_blend_metrics,
    }

winner = max(records, key=lambda n: records[n]["metrics"]["primary"])
winner_record = records[winner]
valid_scores = winner_record["blend_valid"]
metrics = winner_record["metrics"]
selected_epoch = int(winner_record["epoch"])
selected_weight = float(winner_record["weight"])
raw_winner_valid = winner_record["raw_valid"]

print(
    "FINDINGS winner=%s epoch=%d incumbent_blend_own_weight=%.2f"
    % (winner, selected_epoch, selected_weight),
    flush=True,
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True), flush=True)

# Refit the selected recipe on all labels available before test.
x_fit = np.concatenate([x_train, x_valid], axis=0)
y_fit = np.concatenate([y_train, y_valid.astype(np.float32)])
hist_fit, fit_state = build_causal_histories([train, valid], y_fit)

aux_valid = np.column_stack([
    np.asarray(valid.aux["is_click"], dtype=np.float32),
    np.asarray(valid.aux["is_like"], dtype=np.float32),
])
aux_fit = np.concatenate([aux_train, aux_valid], axis=0)

refit = fit_family(
    winner,
    x_fit,
    hist_fit,
    y_fit,
    aux_fit,
    selected_epoch,
    validation=None,
    seed=SEED + 1000 * list(MODEL_CLASSES).index(winner),
)

test = load("test")
x_test = make_x([test])
hist_test = histories_from_state(test, fit_state)
raw_test = predict(refit, x_test, hist_test)

inc_test = np.load(os.path.join(shared, "incumbent_test_scores.npy"))
test_scores = (
    selected_weight * zscore(raw_test)
    + (1.0 - selected_weight) * zscore(inc_test)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if selected_weight < 0.999:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_winner_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.3f}'
    % (
        metrics["primary"],
        metrics["gauc"],
        metrics["ndcg@5"],
        elapsed,
    )
)