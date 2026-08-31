import os
import time
import json
import math
import gc
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 38191
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

DEVICE = torch.device("cpu")
HIST_LEN = 12
BATCH_SIZE = 8192
EPOCHS = 2
HALF_LIFE_DAYS = 5.0

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "hour",
]
VIDEO_COL = CAT_FIELDS.index("video_id")

USER_CARD = int(FEATURE_CARDINALITIES["user_id"])
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])
TAG_CARD = int(FEATURE_CARDINALITIES["tag"])

field_cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
field_offsets = np.cumsum([0] + field_cards[:-1], dtype=np.int64)
TOTAL_CAT_CARD = int(sum(field_cards))


def cat_matrix(split):
    x = np.column_stack(
        [np.asarray(split.X[f], dtype=np.int64) for f in CAT_FIELDS]
    )
    x += field_offsets[None, :]
    return np.ascontiguousarray(x, dtype=np.int64)


def concatenate_cat(splits):
    return np.ascontiguousarray(
        np.concatenate([cat_matrix(s) for s in splits], axis=0),
        dtype=np.int64,
    )


def recency_weights(splits):
    dates = np.concatenate(
        [np.asarray(s.date, dtype=np.int32) for s in splits]
    ).astype(np.float64)
    latest = float(dates.max())
    age = np.maximum(latest - dates, 0.0)
    w = np.exp(-math.log(2.0) * age / HALF_LIFE_DAYS)
    w /= max(float(w.mean()), 1e-8)
    return w.astype(np.float32)


def gap_bucket(current_ms, prior_ms):
    hours = np.maximum(
        (current_ms.astype(np.float64) - prior_ms.astype(np.float64))
        / 3600000.0,
        0.0,
    )
    # Zero is reserved for padding; real elapsed-time buckets are 1..32.
    return np.minimum(
        np.floor(np.log2(hours + 1.0)).astype(np.int64) + 1,
        32,
    ).astype(np.uint8)


def causal_histories(splits, labels, hist_len=HIST_LEN):
    users = np.concatenate(
        [np.asarray(s.user_id, dtype=np.int64) for s in splits]
    )
    times = np.concatenate(
        [np.asarray(s.time_ms, dtype=np.int64) for s in splits]
    )
    videos = np.concatenate(
        [np.asarray(s.video_id, dtype=np.int64) for s in splits]
    )
    labels = np.asarray(labels, dtype=np.int8)
    n = len(labels)
    positions = np.arange(n, dtype=np.int64)

    order = np.lexsort((positions, times, users))
    su = users[order]
    st = times[order]
    sv = videos[order]
    sy = labels[order]

    pos_mask = sy > 0
    pos_counts = np.bincount(
        su[pos_mask], minlength=USER_CARD
    ).astype(np.int64)
    pos_starts = np.empty(USER_CARD, dtype=np.int64)
    pos_starts[0] = 0
    np.cumsum(pos_counts[:-1], out=pos_starts[1:])

    global_before = np.cumsum(sy, dtype=np.int64) - sy.astype(np.int64)
    new_user = np.empty(n, dtype=bool)
    new_user[0] = True
    new_user[1:] = su[1:] != su[:-1]
    bases = np.zeros(n, dtype=np.int64)
    bases[new_user] = global_before[new_user]
    bases = np.maximum.accumulate(bases)
    prior_count = global_before - bases

    pos_videos = sv[pos_mask]
    pos_times = st[pos_mask]
    starts_for_rows = pos_starts[su]

    hv_sorted = np.zeros((n, hist_len), dtype=np.int32)
    hg_sorted = np.zeros((n, hist_len), dtype=np.uint8)

    for j in range(hist_len):
        lag = hist_len - j
        idx = starts_for_rows + prior_count - lag
        ok = idx >= starts_for_rows
        if np.any(ok):
            hv_sorted[ok, j] = pos_videos[idx[ok]].astype(np.int32) + 1
            hg_sorted[ok, j] = gap_bucket(st[ok], pos_times[idx[ok]])

    hv = np.empty_like(hv_sorted)
    hg = np.empty_like(hg_sorted)
    hv[order] = hv_sorted
    hg[order] = hg_sorted
    return hv, hg


def static_histories(fit_splits, fit_labels, target, hist_len=HIST_LEN):
    users = np.concatenate(
        [np.asarray(s.user_id, dtype=np.int64) for s in fit_splits]
    )
    times = np.concatenate(
        [np.asarray(s.time_ms, dtype=np.int64) for s in fit_splits]
    )
    videos = np.concatenate(
        [np.asarray(s.video_id, dtype=np.int64) for s in fit_splits]
    )
    labels = np.concatenate(
        [np.asarray(y, dtype=np.int8) for y in fit_labels]
    )
    positions = np.arange(len(labels), dtype=np.int64)
    positive = labels > 0

    pu = users[positive]
    pt = times[positive]
    pv = videos[positive]
    pp = positions[positive]
    order = np.lexsort((pp, pt, pu))
    pu = pu[order]
    pt = pt[order]
    pv = pv[order]

    counts = np.bincount(pu, minlength=USER_CARD).astype(np.int64)
    ends = np.cumsum(counts, dtype=np.int64)
    starts = ends - counts

    tu = np.asarray(target.user_id, dtype=np.int64)
    tt = np.asarray(target.time_ms, dtype=np.int64)
    starts_row = starts[tu]
    counts_row = counts[tu]

    hv = np.zeros((len(tu), hist_len), dtype=np.int32)
    hg = np.zeros((len(tu), hist_len), dtype=np.uint8)
    for j in range(hist_len):
        lag = hist_len - j
        ok = counts_row >= lag
        idx = starts_row + counts_row - lag
        if np.any(ok):
            hv[ok, j] = pv[idx[ok]].astype(np.int32) + 1
            hg[ok, j] = gap_bucket(tt[ok], pt[idx[ok]])
    return hv, hg


class BaseTemporalModel(nn.Module):
    def __init__(self, seq_dim=20, cat_dim=8):
        super().__init__()
        self.seq_dim = seq_dim
        self.cat_embedding = nn.Embedding(TOTAL_CAT_CARD, cat_dim)
        self.linear = nn.Embedding(TOTAL_CAT_CARD, 1)
        self.video_embedding = nn.Embedding(
            VIDEO_CARD + 1, seq_dim, padding_idx=0
        )
        self.gap_embedding = nn.Embedding(33, seq_dim, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(1))

        deep_in = len(CAT_FIELDS) * cat_dim + seq_dim * 3
        self.head = nn.Sequential(
            nn.Linear(deep_in, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

        nn.init.normal_(self.cat_embedding.weight, std=0.015)
        nn.init.normal_(self.video_embedding.weight, std=0.015)
        nn.init.normal_(self.gap_embedding.weight, std=0.01)
        nn.init.zeros_(self.video_embedding.weight[0])
        nn.init.zeros_(self.gap_embedding.weight[0])
        nn.init.zeros_(self.linear.weight)

    def form_interest(self, hist, gaps):
        raise NotImplementedError

    def forward(self, cats, hist, gaps):
        cat_emb = self.cat_embedding(cats).flatten(1)
        candidate = (
            cats[:, VIDEO_COL] - int(field_offsets[VIDEO_COL]) + 1
        )
        candidate_emb = self.video_embedding(candidate)
        interest = self.form_interest(hist, gaps)
        interaction = candidate_emb * interest
        deep = self.head(
            torch.cat(
                [cat_emb, candidate_emb, interest, interaction], dim=1
            )
        ).squeeze(1)
        wide = self.linear(cats).sum(dim=1).squeeze(1)
        return self.bias + wide + deep


class ContinuousDecayPool(BaseTemporalModel):
    def __init__(self):
        super().__init__()
        self.log_decay = nn.Parameter(torch.tensor(-1.2))

    def form_interest(self, hist, gaps):
        emb = self.video_embedding(hist) + self.gap_embedding(gaps)
        mask = hist != 0
        # Gap buckets approximate log2(hours + 1).
        approximate_hours = torch.pow(
            2.0, torch.clamp(gaps.float() - 1.0, min=0.0)
        ) - 1.0
        decay = nn.functional.softplus(self.log_decay) + 1e-4
        weights = torch.exp(-decay * approximate_hours / 24.0)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-7)
        return torch.sum(emb * weights.unsqueeze(2), dim=1)


class TimeAwareGRU(BaseTemporalModel):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=self.seq_dim,
            hidden_size=self.seq_dim,
            batch_first=True,
        )

    def form_interest(self, hist, gaps):
        x = self.video_embedding(hist) + self.gap_embedding(gaps)
        out, _ = self.gru(x)
        lengths = (hist != 0).sum(dim=1)
        # Histories are right-aligned, so the last column is the latest event.
        result = out[:, -1, :]
        result = result * (lengths > 0).float().unsqueeze(1)
        return result


class TemporalConvNet(BaseTemporalModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(
            self.seq_dim, self.seq_dim, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            self.seq_dim, self.seq_dim, kernel_size=3, padding=2, dilation=2
        )
        self.norm = nn.LayerNorm(self.seq_dim)

    def form_interest(self, hist, gaps):
        mask = hist != 0
        x = self.video_embedding(hist) + self.gap_embedding(gaps)
        z = x.transpose(1, 2)
        z = nn.functional.relu(self.conv1(z))
        z = nn.functional.relu(self.conv2(z)).transpose(1, 2)
        z = self.norm(z)
        z = z.masked_fill(~mask.unsqueeze(2), -1e4)
        pooled = z.max(dim=1).values
        empty = ~mask.any(dim=1)
        pooled[empty] = 0.0
        return pooled


MODEL_CLASSES = {
    "continuous_decay_pool": ContinuousDecayPool,
    "time_aware_gru": TimeAwareGRU,
    "temporal_conv": TemporalConvNet,
}


@torch.no_grad()
def predict_model(model, cats, hist, gaps):
    model.eval()
    result = np.empty(len(cats), dtype=np.float64)
    infer_batch = BATCH_SIZE * 2
    for start in range(0, len(cats), infer_batch):
        end = min(start + infer_batch, len(cats))
        xb = torch.from_numpy(cats[start:end]).to(DEVICE)
        hb = torch.from_numpy(
            hist[start:end].astype(np.int64, copy=False)
        ).to(DEVICE)
        gb = torch.from_numpy(
            gaps[start:end].astype(np.int64, copy=False)
        ).to(DEVICE)
        result[start:end] = (
            model(xb, hb, gb).cpu().numpy().astype(np.float64)
        )
    return result


def fit_temporal(
    model_name,
    fit_splits,
    fit_labels,
    target,
    seed_offset=0,
):
    torch.manual_seed(SEED + seed_offset)

    labels = np.concatenate(
        [np.asarray(y, dtype=np.float32) for y in fit_labels]
    )
    cats = concatenate_cat(fit_splits)
    hist, gaps = causal_histories(fit_splits, labels)
    weights = recency_weights(fit_splits)

    target_cats = cat_matrix(target)
    target_hist, target_gaps = static_histories(
        fit_splits, fit_labels, target
    )

    model = MODEL_CLASSES[model_name]().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0015, weight_decay=2e-6
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 100 + seed_offset)

    n = len(labels)
    y_tensor = torch.from_numpy(labels)
    w_tensor = torch.from_numpy(weights)

    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            ix = idx.numpy()

            xb = torch.from_numpy(cats[ix]).to(DEVICE)
            hb = torch.from_numpy(
                hist[ix].astype(np.int64, copy=False)
            ).to(DEVICE)
            gb = torch.from_numpy(
                gaps[ix].astype(np.int64, copy=False)
            ).to(DEVICE)
            yb = y_tensor[idx].to(DEVICE)
            wb = w_tensor[idx].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb, gb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (losses * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    scores = predict_model(
        model, target_cats, target_hist, target_gaps
    )

    del (
        model, optimizer, labels, cats, hist, gaps, weights,
        target_cats, target_hist, target_gaps, y_tensor, w_tensor
    )
    gc.collect()
    return scores


def empirical_user_tag_scores(fit_splits, fit_labels, target):
    users = np.concatenate(
        [np.asarray(s.user_id, dtype=np.int64) for s in fit_splits]
    )
    tags = np.concatenate(
        [np.asarray(s.X["tag"], dtype=np.int64) for s in fit_splits]
    )
    labels = np.concatenate(
        [np.asarray(y, dtype=np.float64) for y in fit_labels]
    )
    weights = recency_weights(fit_splits).astype(np.float64)

    key = users * np.int64(TAG_CARD) + tags
    size = USER_CARD * TAG_CARD
    total = np.bincount(
        key, weights=weights, minlength=size
    ).astype(np.float64)
    positive = np.bincount(
        key, weights=weights * labels, minlength=size
    ).astype(np.float64)

    tag_total = np.bincount(
        tags, weights=weights, minlength=TAG_CARD
    ).astype(np.float64)
    tag_positive = np.bincount(
        tags, weights=weights * labels, minlength=TAG_CARD
    ).astype(np.float64)
    global_rate = float(np.sum(weights * labels) / np.sum(weights))
    tag_prior = (
        tag_positive + 30.0 * global_rate
    ) / (tag_total + 30.0)

    tu = np.asarray(target.user_id, dtype=np.int64)
    tt = np.asarray(target.X["tag"], dtype=np.int64)
    tkey = tu * np.int64(TAG_CARD) + tt

    score = (
        positive[tkey] + 8.0 * tag_prior[tt]
    ) / (total[tkey] + 8.0)
    return score.astype(np.float64)


def standardized_blend(own, incumbent, own_weight):
    own = np.asarray(own, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)
    own_z = (own - own.mean()) / max(float(own.std()), 1e-8)
    inc_z = (incumbent - incumbent.mean()) / max(
        float(incumbent.std()), 1e-8
    )
    return own_weight * own_z + (1.0 - own_weight) * inc_z


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ["SHARED_ARTIFACTS"]
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

raw_predictions = {}
for i, name in enumerate(MODEL_CLASSES):
    raw_predictions[name] = fit_temporal(
        name, [train], [y_train], valid, seed_offset=17 * (i + 1)
    )

raw_predictions["empirical_user_tag_decay"] = empirical_user_tag_scores(
    [train], [y_train], valid
)

candidate_log = {}
best_primary = -np.inf
best_name = None
best_weight = None
best_valid = None
best_raw_valid = None
best_metrics = None

blend_weights = [0.0, 0.25, 0.50, 0.75, 1.0]

for name, raw in raw_predictions.items():
    raw_met = evaluate(valid.user_id, y_valid, raw)
    candidate_log[name + "_raw"] = float(raw_met["primary"])

    for weight in blend_weights:
        blended = standardized_blend(raw, inc_valid, weight)
        met = evaluate(valid.user_id, y_valid, blended)
        key = name + "_blend_" + str(weight)
        candidate_log[key] = float(met["primary"])

        if float(met["primary"]) > best_primary:
            best_primary = float(met["primary"])
            best_name = name
            best_weight = float(weight)
            best_valid = blended.copy()
            best_raw_valid = raw.copy()
            best_metrics = met

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_family": best_name,
            "selected_own_weight": best_weight,
            "half_life_days": HALF_LIFE_DAYS,
            "history_length": HIST_LEN,
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    if best_weight < 1.0:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

# Refit the selected recipe on train + validation and score test. If validation
# selected the pure incumbent, its already-refit trusted test predictions are
# exactly the corresponding recipe.
test = load("test")
if best_weight == 0.0:
    test_scores = np.load(inc_test_path).astype(np.float64)
else:
    if best_name == "empirical_user_tag_decay":
        own_test = empirical_user_tag_scores(
            [train, valid], [y_train, y_valid], test
        )
    else:
        own_test = fit_temporal(
            best_name,
            [train, valid],
            [y_train, y_valid.astype(np.float32)],
            test,
            seed_offset=211,
        )

    if best_weight < 1.0:
        inc_test = np.load(inc_test_path).astype(np.float64)
        test_scores = standardized_blend(
            own_test, inc_test, best_weight
        )
    else:
        test_scores = own_test

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
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)