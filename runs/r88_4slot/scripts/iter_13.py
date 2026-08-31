import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 27183
BATCH_SIZE = 16384
EPOCHS = 3
LR = 0.003
HALF_LIFE = 7.0
EMBED_DIM = 12

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
    "video_type",
]
FIELD_INDEX = {name: i for i, name in enumerate(FIELDS)}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
TOTAL_CARD = int(sum(cards))
HASH_SIZE = 1 << 19
HASH_MASK = HASH_SIZE - 1


def encode(split):
    return np.stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in FIELDS],
        axis=1,
    )


def numeric_context(split):
    duration = np.asarray(split.num["duration_ms"], dtype=np.float32)
    duration = np.nan_to_num(duration, nan=0.0, posinf=0.0, neginf=0.0)
    logdur = np.log1p(np.maximum(duration, 0.0))
    logdur = (logdur - 11.0) / 1.5
    logdur = np.clip(logdur, -4.0, 4.0)

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    angle = 2.0 * np.pi * hour / 24.0

    age = np.asarray(split.num["user_register_days"], dtype=np.float32)
    age = np.nan_to_num(age, nan=0.0, posinf=0.0, neginf=0.0)
    age = np.clip((np.log1p(np.maximum(age, 0.0)) - 7.0) / 1.5, -4.0, 4.0)

    return np.column_stack(
        [
            logdur,
            logdur * logdur,
            np.sin(angle),
            np.cos(angle),
            np.sin(2.0 * angle),
            np.cos(2.0 * angle),
            age,
        ]
    ).astype(np.float32)


def concatenate_encoded(a, b):
    return np.concatenate([a, b], axis=0)


def recency_weights(dates):
    dates = np.asarray(dates)
    unique = np.unique(dates)
    day = np.searchsorted(unique, dates).astype(np.float32)
    age = float(len(unique) - 1) - day
    w = np.exp2(-age / HALF_LIFE).astype(np.float32)
    w /= max(float(w.mean()), 1e-6)
    return w


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    repeated_starts = np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    mask = repeated_counts > 1
    ranked[mask] = positions[mask] / (repeated_counts[mask] - 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


class AdditiveBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.offsets = nn.Parameter(
            torch.tensor(offsets, dtype=torch.long), requires_grad=False
        )
        self.wide = nn.Embedding(TOTAL_CARD, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.wide.weight)

    def additive(self, x):
        global_ids = x + self.offsets.unsqueeze(0)
        return self.wide(global_ids).sum(dim=1).squeeze(1) + self.bias


class CrossMemorization(AdditiveBase):
    """Exact prediction is formed from additive fields plus hashed conjunction tables."""

    def __init__(self):
        super().__init__()
        self.pairs = [
            ("user_id", "video_id"),
            ("user_id", "author_id"),
            ("user_id", "tag"),
            ("user_id", "tab"),
            ("user_id", "duration_bucket"),
            ("user_id", "upload_type"),
            ("author_id", "tag"),
            ("video_id", "tab"),
        ]
        self.tables = nn.ModuleList(
            [nn.Embedding(HASH_SIZE, 1) for _ in self.pairs]
        )
        for table in self.tables:
            nn.init.zeros_(table.weight)

    def forward(self, x, numeric):
        score = self.additive(x)
        for table, (left, right) in zip(self.tables, self.pairs):
            a = x[:, FIELD_INDEX[left]]
            b = x[:, FIELD_INDEX[right]]
            h = (a * 1000003 + b * 9176 + a * b * 37 + 101) & HASH_MASK
            score = score + table(h).squeeze(1)
        return score


class TensorCP3(AdditiveBase):
    """Prediction is formed by low-rank third-order user-item-context tensors."""

    def __init__(self):
        super().__init__()
        self.user = nn.Embedding(cards[FIELD_INDEX["user_id"]], EMBED_DIM)
        self.video = nn.Embedding(cards[FIELD_INDEX["video_id"]], EMBED_DIM)
        self.author = nn.Embedding(cards[FIELD_INDEX["author_id"]], EMBED_DIM)
        self.tag = nn.Embedding(cards[FIELD_INDEX["tag"]], EMBED_DIM)
        self.tab = nn.Embedding(cards[FIELD_INDEX["tab"]], EMBED_DIM)
        self.duration = nn.Embedding(
            cards[FIELD_INDEX["duration_bucket"]], EMBED_DIM
        )
        self.upload = nn.Embedding(cards[FIELD_INDEX["upload_type"]], EMBED_DIM)
        for emb in [
            self.user,
            self.video,
            self.author,
            self.tag,
            self.tab,
            self.duration,
            self.upload,
        ]:
            nn.init.normal_(emb.weight, std=0.04)

    def forward(self, x, numeric):
        u = self.user(x[:, FIELD_INDEX["user_id"]])
        v = self.video(x[:, FIELD_INDEX["video_id"]])
        a = self.author(x[:, FIELD_INDEX["author_id"]])
        tag = self.tag(x[:, FIELD_INDEX["tag"]])
        tab = self.tab(x[:, FIELD_INDEX["tab"]])
        dur = self.duration(x[:, FIELD_INDEX["duration_bucket"]])
        upload = self.upload(x[:, FIELD_INDEX["upload_type"]])

        scale = 1.0 / np.sqrt(float(EMBED_DIM))
        three_way = (
            (u * v * tab).sum(dim=1)
            + (u * a * tag).sum(dim=1)
            + (u * v * dur).sum(dim=1)
            + (u * a * upload).sum(dim=1)
        ) * scale
        return self.additive(x) + three_way


class VaryingCoefficientGAM(AdditiveBase):
    """Each user has coefficients for smooth duration and hour basis functions."""

    def __init__(self):
        super().__init__()
        n_users = cards[FIELD_INDEX["user_id"]]
        n_numeric = 7
        self.user_coeff = nn.Embedding(n_users, n_numeric)
        self.global_coeff = nn.Parameter(torch.zeros(n_numeric))
        self.user_author = nn.Embedding(HASH_SIZE, 1)
        nn.init.normal_(self.user_coeff.weight, std=0.015)
        nn.init.zeros_(self.user_author.weight)

    def forward(self, x, numeric):
        uid = x[:, FIELD_INDEX["user_id"]]
        aid = x[:, FIELD_INDEX["author_id"]]
        coeff = self.user_coeff(uid) + self.global_coeff.unsqueeze(0)
        smooth = (coeff * numeric).sum(dim=1)

        # A single interpretable preference intercept anchors the smooth model.
        h = (uid * 1000003 + aid * 9176 + 53) & HASH_MASK
        author_pref = self.user_author(h).squeeze(1)
        return self.additive(x) + smooth + author_pref


def make_model(name):
    if name == "cross_memorization":
        return CrossMemorization()
    if name == "tensor_cp3":
        return TensorCP3()
    if name == "varying_coefficient_gam":
        return VaryingCoefficientGAM()
    raise ValueError(name)


def predict(model, x_np, numeric_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for lo in range(0, len(x_np), BATCH_SIZE * 2):
            hi = min(lo + BATCH_SIZE * 2, len(x_np))
            xb = torch.from_numpy(x_np[lo:hi])
            nb = torch.from_numpy(numeric_np[lo:hi])
            result[lo:hi] = model(xb, nb).cpu().numpy()
    return result


def fit_model(name, x_np, numeric_np, labels, dates, epochs=EPOCHS):
    torch.manual_seed(SEED)
    model = make_model(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)

    x_t = torch.from_numpy(x_np)
    numeric_t = torch.from_numpy(numeric_np)
    y_t = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    w_t = torch.from_numpy(recency_weights(dates))

    generator = torch.Generator()
    generator.manual_seed(SEED + 19)
    n = len(x_np)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(x_t[idx], numeric_t[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, y_t[idx], reduction="none"
            )
            loss = (losses * w_t[idx]).sum() / w_t[idx].sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


train = load("train")
valid = load("valid")

x_train = encode(train)
x_valid = encode(valid)
n_train = numeric_context(train)
n_valid = numeric_context(valid)

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)
dates_train = np.asarray(train.date)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_rank = within_user_rank(valid_users, inc_valid)

families = [
    "cross_memorization",
    "tensor_cp3",
    "varying_coefficient_gam",
]

raw_valid = {}
raw_ranks = {}
candidate_log = {}

for family in families:
    model = fit_model(
        family, x_train, n_train, y_train, dates_train, epochs=EPOCHS
    )
    scores = predict(model, x_valid, n_valid)
    ranks = within_user_rank(valid_users, scores)
    metrics = evaluate(valid_users, y_valid, scores)

    raw_valid[family] = scores
    raw_ranks[family] = ranks
    candidate_log[family + "_standalone"] = float(metrics["primary"])

    del model
    gc.collect()

best_primary = -np.inf
best_name = None
best_family = None
best_alpha = None
best_scores = None
best_raw = None
best_metrics = None

alphas = np.linspace(0.0, 1.0, 9)

for family in families:
    for alpha in alphas:
        blended = (1.0 - alpha) * inc_valid_rank + alpha * raw_ranks[family]
        metrics = evaluate(valid_users, y_valid, blended)
        name = family + "_blend_" + str(round(float(alpha), 3))
        candidate_log[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_name = name
            best_family = family
            best_alpha = float(alpha)
            best_scores = blended.copy()
            best_raw = raw_valid[family].copy()
            best_metrics = metrics

# Also compare structurally heterogeneous rank ensembles.
ensemble_specs = [
    ("all_equal", [1 / 3, 1 / 3, 1 / 3]),
    ("cross_tensor", [0.5, 0.5, 0.0]),
    ("cross_gam", [0.5, 0.0, 0.5]),
    ("tensor_gam", [0.0, 0.5, 0.5]),
]
best_ensemble_weights = None

for ensemble_name, weights in ensemble_specs:
    new_rank = sum(
        weights[j] * raw_ranks[families[j]] for j in range(len(families))
    )
    for alpha in alphas:
        blended = (1.0 - alpha) * inc_valid_rank + alpha * new_rank
        metrics = evaluate(valid_users, y_valid, blended)
        name = ensemble_name + "_blend_" + str(round(float(alpha), 3))
        candidate_log[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_name = name
            best_family = "ensemble"
            best_alpha = float(alpha)
            best_ensemble_weights = list(weights)
            best_scores = blended.copy()
            best_raw = new_rank.copy()
            best_metrics = metrics

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": best_name,
            "blend_alpha": best_alpha,
            "family": best_family,
        },
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
    )

# Refit the selected identical recipe on train + validation, then score test.
test = load("test")
x_test = encode(test)
n_test = numeric_context(test)
test_users = np.asarray(test.user_id)

x_fit = concatenate_encoded(x_train, x_valid)
n_fit = concatenate_encoded(n_train, n_valid)
y_fit = np.concatenate(
    [y_train, np.asarray(valid.y, dtype=np.float32)], axis=0
)
dates_fit = np.concatenate(
    [np.asarray(train.date), np.asarray(valid.date)], axis=0
)

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
inc_test_rank = within_user_rank(test_users, inc_test)

if best_family != "ensemble":
    final_model = fit_model(
        best_family, x_fit, n_fit, y_fit, dates_fit, epochs=EPOCHS
    )
    raw_test = predict(final_model, x_test, n_test)
    new_test_rank = within_user_rank(test_users, raw_test)
    del final_model
else:
    new_test_rank = np.zeros(len(x_test), dtype=np.float64)
    for family, weight in zip(families, best_ensemble_weights):
        if weight == 0.0:
            continue
        final_model = fit_model(
            family, x_fit, n_fit, y_fit, dates_fit, epochs=EPOCHS
        )
        raw_test = predict(final_model, x_test, n_test)
        new_test_rank += weight * within_user_rank(test_users, raw_test)
        del final_model
        gc.collect()

test_scores = (1.0 - best_alpha) * inc_test_rank + best_alpha * new_test_rank

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
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