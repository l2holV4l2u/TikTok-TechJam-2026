import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 9143
EMBED_DIM = 12
HISTORY_LEN = 8
EPOCHS = 3
BATCH_SIZE = 8192
PRED_BATCH = 32768
HALF_LIFE_DAYS = 5.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "upload_type",
    "duration_bucket",
    "hour",
    "user_active_degree",
]

CARDINALITIES = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDINALITIES[:-1]).astype(np.int64)
TOTAL_CARDINALITY = int(sum(CARDINALITIES))
VIDEO_FIELD_INDEX = FIELDS.index("video_id")
VIDEO_OFFSET = int(OFFSETS[VIDEO_FIELD_INDEX])
USER_CARDINALITY = int(FEATURE_CARDINALITIES["user_id"])


def categorical_matrix(s):
    return np.column_stack(
        [np.asarray(s.X[f], dtype=np.int64) for f in FIELDS]
    ).astype(np.int64, copy=False)


def days_from_yyyymmdd(dates):
    dates = np.asarray(dates, dtype=np.int64)
    year = dates // 10000
    month = (dates // 100) % 100
    day = dates % 100

    # All supplied dates are in April/May 2022. This representation remains
    # monotone and gives the correct distance over the April-May boundary.
    return ((year - 2022) * 365 + (month - 4) * 30 + day).astype(np.float32)


def recency_weights(dates):
    day = days_from_yyyymmdd(dates)
    age = float(day.max()) - day
    w = np.exp(-np.log(2.0) * age / HALF_LIFE_DAYS).astype(np.float32)
    w /= max(float(w.mean()), 1e-8)
    return w


def causal_positive_history(user, time_ms, video, labels, k=HISTORY_LEN):
    """
    For each training row, retrieve only positive videos occurring earlier
    for that user. Tied timestamps use original row order as required.
    """
    user = np.asarray(user, dtype=np.int64)
    time_ms = np.asarray(time_ms, dtype=np.int64)
    video = np.asarray(video, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    n = user.size
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, time_ms, user))
    us = user[order]
    vs = video[order]
    ys = labels[order]

    new_user = np.empty(n, dtype=bool)
    new_user[0] = True
    new_user[1:] = us[1:] != us[:-1]
    starts = np.flatnonzero(new_user)

    cumulative = np.cumsum(ys.astype(np.int64))
    positives_before_group = cumulative[starts] - ys[starts].astype(np.int64)
    counts = np.diff(np.r_[starts, n])
    group_base = np.repeat(positives_before_group, counts)
    before = cumulative - ys.astype(np.int64) - group_base

    positive_videos = vs[ys > 0]
    out_sorted = np.zeros((n, k), dtype=np.int64)

    for j in range(k):
        ok = before > j
        if np.any(ok):
            index = group_base[ok] + before[ok] - 1 - j
            out_sorted[ok, j] = positive_videos[index]

    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out


def latest_positive_profiles(user, time_ms, video, labels, k=HISTORY_LEN):
    """Latest k positive videos per user from a fully observed fitting split."""
    user = np.asarray(user, dtype=np.int64)
    time_ms = np.asarray(time_ms, dtype=np.int64)
    video = np.asarray(video, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    n = user.size
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, time_ms, user))
    us = user[order]
    vs = video[order]
    ys = labels[order]

    pos_users = us[ys > 0]
    pos_videos = vs[ys > 0]
    counts = np.bincount(pos_users, minlength=USER_CARDINALITY).astype(np.int64)
    ends = np.cumsum(counts)

    profiles = np.zeros((USER_CARDINALITY, k), dtype=np.int64)
    users = np.arange(USER_CARDINALITY, dtype=np.int64)
    for j in range(k):
        ok = counts > j
        idx = ends[ok] - 1 - j
        profiles[users[ok], j] = pos_videos[idx]
    return profiles


def histories_for_scoring(s, profiles):
    uid = np.asarray(s.X["user_id"], dtype=np.int64)
    uid = np.clip(uid, 0, profiles.shape[0] - 1)
    return profiles[uid]


class CommonCTR(nn.Module):
    def __init__(self, family, base_rate):
        super().__init__()
        self.family = family
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, EMBED_DIM)
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.bias = nn.Parameter(torch.zeros(()))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)
        p = float(np.clip(base_rate, 1e-5, 1.0 - 1e-5))
        self.bias.data.fill_(np.log(p / (1.0 - p)))

        flat_dim = len(FIELDS) * EMBED_DIM

        if family == "dcn":
            self.cross_w = nn.ParameterList(
                [nn.Parameter(torch.empty(flat_dim)) for _ in range(2)]
            )
            self.cross_b = nn.ParameterList(
                [nn.Parameter(torch.zeros(flat_dim)) for _ in range(2)]
            )
            for w in self.cross_w:
                nn.init.normal_(w, std=0.02)
            self.head = nn.Sequential(
                nn.Linear(flat_dim, 48),
                nn.SiLU(),
                nn.Linear(48, 1),
            )

        elif family == "din":
            self.attention = nn.Sequential(
                nn.Linear(4 * EMBED_DIM, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )
            self.head = nn.Sequential(
                nn.Linear(4 * EMBED_DIM, 48),
                nn.SiLU(),
                nn.Linear(48, 1),
            )

        elif family == "gru":
            self.gru_cell = nn.GRUCell(EMBED_DIM, EMBED_DIM)
            self.head = nn.Sequential(
                nn.Linear(4 * EMBED_DIM, 48),
                nn.SiLU(),
                nn.Linear(48, 1),
            )
        else:
            raise ValueError(f"Unknown family: {family}")

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def fm_score(self, ids, emb):
        linear = self.linear(ids).sum(dim=1).squeeze(-1) + self.bias
        summed = emb.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        ).sum(dim=1)
        return linear + interaction

    def forward(self, x, history):
        offsets = torch.as_tensor(OFFSETS, dtype=torch.long, device=x.device)
        ids = x + offsets
        emb = self.embedding(ids)
        score = self.fm_score(ids, emb)

        if self.family == "dcn":
            x0 = emb.flatten(1)
            z = x0
            for w, b in zip(self.cross_w, self.cross_b):
                scalar = (z * w).sum(dim=1, keepdim=True)
                z = x0 * scalar + b + z
            return score + self.head(z).squeeze(1)

        candidate = emb[:, VIDEO_FIELD_INDEX, :]
        mask = history.ne(0)
        hist_ids = history + VIDEO_OFFSET
        hist_emb = self.embedding(hist_ids)
        hist_emb = hist_emb * mask.unsqueeze(-1)

        if self.family == "din":
            q = candidate.unsqueeze(1).expand_as(hist_emb)
            att_input = torch.cat(
                [hist_emb, q, hist_emb * q, hist_emb - q], dim=-1
            )
            logits = self.attention(att_input).squeeze(-1)
            logits = logits.masked_fill(~mask, -1e4)
            attention = torch.softmax(logits, dim=1)
            attention = attention * mask
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
            profile = (attention.unsqueeze(-1) * hist_emb).sum(dim=1)

        else:
            # Histories are stored newest first. Process oldest to newest and
            # update the recurrent state only at real history positions.
            state = torch.zeros_like(candidate)
            for j in range(history.shape[1] - 1, -1, -1):
                proposed = self.gru_cell(hist_emb[:, j, :], state)
                m = mask[:, j].unsqueeze(1)
                state = torch.where(m, proposed, state)
            profile = state

        interaction = torch.cat(
            [candidate, profile, candidate * profile, candidate - profile],
            dim=1,
        )
        return score + self.head(interaction).squeeze(1)


def train_model(x, history, y, weights, family):
    family_seed = {"dcn": 11, "din": 29, "gru": 47}[family]
    torch.manual_seed(SEED + family_seed)

    model = CommonCTR(family, float(np.mean(y)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=2e-5)

    xt = torch.from_numpy(x)
    ht = torch.from_numpy(history)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    wt = torch.from_numpy(np.asarray(weights, dtype=np.float32))

    n = len(y)
    generator = torch.Generator()
    generator.manual_seed(SEED + family_seed + 100)

    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        model.train()
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:min(start + BATCH_SIZE, n)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xt[idx], ht[idx])
            losses = F.binary_cross_entropy_with_logits(
                logits, yt[idx], reduction="none"
            )
            loss = (losses * wt[idx]).sum() / wt[idx].sum().clamp_min(1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict_model(model, x, history):
    model.eval()
    scores = np.empty(x.shape[0], dtype=np.float32)
    xt = torch.from_numpy(x)
    ht = torch.from_numpy(history)
    for start in range(0, x.shape[0], PRED_BATCH):
        end = min(start + PRED_BATCH, x.shape[0])
        scores[start:end] = model(xt[start:end], ht[start:end]).cpu().numpy()
    return scores


def standardize(scores):
    scores = np.asarray(scores, dtype=np.float64)
    sd = float(scores.std())
    if sd < 1e-12:
        sd = 1.0
    return (scores - float(scores.mean())) / sd


train = load("train")
valid = load("valid")

train_x = categorical_matrix(train)
valid_x = categorical_matrix(valid)
train_y = np.asarray(train.y, dtype=np.int8)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_history = causal_positive_history(
    train.user_id, train.time_ms, train.video_id, train_y
)
train_profiles = latest_positive_profiles(
    train.user_id, train.time_ms, train.video_id, train_y
)
valid_history = histories_for_scoring(valid, train_profiles)
train_weights = recency_weights(train.date)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid_z = standardize(inc_valid)

families = ("dcn", "din", "gru")
own_valid = {}
candidate_results = {}
trained_models = {}

best_primary = -np.inf
best_family = None
best_weight = None
best_scores = None
best_metrics = None
best_raw = None

inc_metrics = evaluate(valid_users, valid_y, inc_valid)
candidate_results["trusted_incumbent"] = float(inc_metrics["primary"])

for family in families:
    model = train_model(
        train_x, train_history, train_y, train_weights, family
    )
    trained_models[family] = model
    raw_scores = predict_model(model, valid_x, valid_history)
    own_valid[family] = raw_scores

    raw_metrics = evaluate(valid_users, valid_y, raw_scores)
    candidate_results[f"{family}_standalone"] = float(raw_metrics["primary"])

    if float(raw_metrics["primary"]) > best_primary:
        best_primary = float(raw_metrics["primary"])
        best_family = family
        best_weight = 1.0
        best_scores = raw_scores.copy()
        best_metrics = raw_metrics
        best_raw = raw_scores

    raw_z = standardize(raw_scores)
    for own_weight in (0.08, 0.14, 0.20, 0.28, 0.36, 0.45, 0.55):
        blended = (
            (1.0 - own_weight) * inc_valid_z
            + own_weight * raw_z
        )
        metrics = evaluate(valid_users, valid_y, blended)
        name = f"{family}_blend_{own_weight:.2f}"
        candidate_results[name] = float(metrics["primary"])
        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_family = family
            best_weight = float(own_weight)
            best_scores = blended.copy()
            best_metrics = metrics
            best_raw = raw_scores

# Permit the trusted incumbent itself to win the internal comparison.
if float(inc_metrics["primary"]) > best_primary:
    best_primary = float(inc_metrics["primary"])
    best_family = "incumbent"
    best_weight = 0.0
    best_scores = inc_valid.copy()
    best_metrics = inc_metrics
    best_raw = None

print("CANDIDATES " + json.dumps(candidate_results, sort_keys=True), flush=True)
print(
    "FINDINGS selected_family=%s own_weight=%.2f history_len=%d half_life=%.1f"
    % (best_family, best_weight, HISTORY_LEN, HALF_LIFE_DAYS),
    flush=True,
)

test = load("test")

if best_family == "incumbent":
    test_scores = np.load(inc_test_path).astype(np.float64)
else:
    # Refit the identical selected recipe on train + validation for test.
    combined_x = np.concatenate([train_x, valid_x], axis=0)
    combined_y = np.concatenate([train_y, valid_y], axis=0)
    combined_user = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    combined_time = np.concatenate([
        np.asarray(train.time_ms, dtype=np.int64),
        np.asarray(valid.time_ms, dtype=np.int64),
    ])
    combined_video = np.concatenate([
        np.asarray(train.video_id, dtype=np.int64),
        np.asarray(valid.video_id, dtype=np.int64),
    ])
    combined_date = np.concatenate([
        np.asarray(train.date, dtype=np.int64),
        np.asarray(valid.date, dtype=np.int64),
    ])

    combined_history = causal_positive_history(
        combined_user, combined_time, combined_video, combined_y
    )
    combined_profiles = latest_positive_profiles(
        combined_user, combined_time, combined_video, combined_y
    )
    combined_weights = recency_weights(combined_date)

    test_x = categorical_matrix(test)
    test_history = histories_for_scoring(test, combined_profiles)

    final_model = train_model(
        combined_x,
        combined_history,
        combined_y,
        combined_weights,
        best_family,
    )
    own_test = predict_model(final_model, test_x, test_history)

    if best_weight < 1.0:
        inc_test = np.load(inc_test_path).astype(np.float64)
        test_scores = (
            (1.0 - best_weight) * standardize(inc_test)
            + best_weight * standardize(own_test)
        )
    else:
        test_scores = own_test

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if best_family != "incumbent" and best_weight < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

elapsed = float(time.time() - START)
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))