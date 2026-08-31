import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
THREADS = min(8, os.cpu_count() or 1)
BATCH_SIZE = 4096
EPOCHS = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

CAT_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
n_train = len(train.user_id)

# Four-day recency weighting is fixed before this experiment based on the
# previously established drift result.
dates = np.asarray(train.date, dtype=np.int64)
age = int(dates.max()) - dates
w_train = np.power(0.5, age.astype(np.float32) / 4.0).astype(np.float32)
w_train /= w_train.mean()

cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
offsets = np.zeros(len(cards), dtype=np.int64)
offsets[1:] = np.cumsum(cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def make_categorical(split):
    x = np.empty((len(split.user_id), len(CAT_FIELDS)), dtype=np.int64)
    for j, name in enumerate(CAT_FIELDS):
        x[:, j] = np.asarray(split.X[name], dtype=np.int64) + offsets[j]
    return x


xcat_train = make_categorical(train)
xcat_valid = make_categorical(valid)
xcat_test = make_categorical(test)


def raw_numeric(split_name, split):
    cols = []
    for name in NUM_FIELDS:
        a = np.asarray(split.num[name], dtype=np.float32)
        if name != "user_register_days":
            a = np.log1p(np.maximum(a, 0.0))
        cols.append(a)

    # Only training-derived entity histories are exposed by this API.
    for key in ["video_id", "author_id"]:
        hist = historical_features(split_name, key=key)
        for name in sorted(hist):
            cols.append(np.asarray(hist[name], dtype=np.float32))

    return np.column_stack(cols).astype(np.float32)


raw_train = raw_numeric("train", train)
raw_valid = raw_numeric("valid", valid)
raw_test = raw_numeric("test", test)

finite_train = np.where(np.isfinite(raw_train), raw_train, np.nan)
num_mean = np.nanmean(finite_train, axis=0).astype(np.float32)
num_mean = np.where(np.isfinite(num_mean), num_mean, 0.0).astype(np.float32)
num_std = np.nanstd(finite_train, axis=0).astype(np.float32)
num_std = np.where(
    np.isfinite(num_std) & (num_std > 1e-5), num_std, 1.0
).astype(np.float32)


def normalize_numeric(a):
    a = np.asarray(a, dtype=np.float32)
    a = np.where(np.isfinite(a), a, num_mean[None, :])
    a = (a - num_mean[None, :]) / num_std[None, :]
    return np.clip(a, -8.0, 8.0).astype(np.float32)


xnum_train = normalize_numeric(raw_train)
xnum_valid = normalize_numeric(raw_valid)
xnum_test = normalize_numeric(raw_test)
n_num = xnum_train.shape[1]

del raw_train, raw_valid, raw_test, finite_train

base_rate = float(np.clip(y_train.mean(), 1e-6, 1.0 - 1e-6))
base_logit = float(np.log(base_rate / (1.0 - base_rate)))


class FieldWeightedFM(nn.Module):
    """Pairwise factorization with a learned coefficient for each field pair."""

    def __init__(self, rank=16):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, rank)
        self.linear = nn.Embedding(total_cardinality, 1)
        self.pair_weight = nn.Parameter(torch.ones(len(CAT_FIELDS), len(CAT_FIELDS)))
        self.numeric_linear = nn.Linear(n_num, 1, bias=False)
        self.numeric_interaction = nn.Sequential(
            nn.Linear(n_num, 32),
            nn.ReLU(),
            nn.Linear(32, rank),
        )
        self.bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, xcat, xnum):
        e = self.embedding(xcat)
        score = self.linear(xcat).sum(dim=1).squeeze(1)
        for i in range(len(CAT_FIELDS)):
            for j in range(i + 1, len(CAT_FIELDS)):
                score = score + self.pair_weight[i, j] * (e[:, i] * e[:, j]).sum(1)

        numeric_vector = self.numeric_interaction(xnum)
        candidate_vector = e[:, 1] + 0.5 * e[:, 2]
        score = score + 0.25 * (numeric_vector * candidate_vector).sum(1)
        return self.bias + score + self.numeric_linear(xnum).squeeze(1)


class LowRankCross(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.u = nn.Linear(dim, rank, bias=False)
        self.v = nn.Linear(rank, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, x):
        return x + x0 * (self.v(self.u(x)) + self.bias)


class DCNv2(nn.Module):
    """Low-rank explicit crosses in parallel with a nonlinear tower."""

    def __init__(self, rank=10, cross_rank=32):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, rank)
        nn.init.normal_(self.embedding.weight, std=0.02)

        dim = len(CAT_FIELDS) * rank + n_num
        self.cross1 = LowRankCross(dim, cross_rank)
        self.cross2 = LowRankCross(dim, cross_rank)
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.06),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 64, 1)
        self.bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))

    def forward(self, xcat, xnum):
        e = self.embedding(xcat).flatten(1)
        x0 = torch.cat([e, xnum], dim=1)
        crossed = self.cross1(x0, x0)
        crossed = self.cross2(x0, crossed)
        deep = self.deep(x0)
        return self.bias + self.output(torch.cat([crossed, deep], dim=1)).squeeze(1)


class AutoInt(nn.Module):
    """Self-attention forms candidate-dependent interactions among fields."""

    def __init__(self, rank=12):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, rank)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.attn1 = nn.MultiheadAttention(
            rank, num_heads=3, dropout=0.04, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            rank, num_heads=3, dropout=0.04, batch_first=True
        )
        self.norm1 = nn.LayerNorm(rank)
        self.norm2 = nn.LayerNorm(rank)
        self.numeric_context = nn.Sequential(
            nn.Linear(n_num, 48),
            nn.ReLU(),
            nn.Linear(48, rank),
        )
        self.output = nn.Sequential(
            nn.Linear((len(CAT_FIELDS) + 1) * rank + n_num, 80),
            nn.ReLU(),
            nn.Dropout(0.06),
            nn.Linear(80, 1),
        )
        self.bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))

    def forward(self, xcat, xnum):
        fields = self.embedding(xcat)
        numeric_token = self.numeric_context(xnum).unsqueeze(1)
        z = torch.cat([fields, numeric_token], dim=1)
        a, _ = self.attn1(z, z, z, need_weights=False)
        z = self.norm1(z + a)
        a, _ = self.attn2(z, z, z, need_weights=False)
        z = self.norm2(z + a)
        return self.bias + self.output(
            torch.cat([z.flatten(1), xnum], dim=1)
        ).squeeze(1)


class MMoE(nn.Module):
    """Multiple experts shared by long-view, click, and like objectives."""

    def __init__(self, rank=10, n_experts=4):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, rank)
        nn.init.normal_(self.embedding.weight, std=0.02)
        dim = len(CAT_FIELDS) * rank + n_num

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, 96),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(96, 48),
                    nn.ReLU(),
                )
                for _ in range(n_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(dim, n_experts) for _ in range(3)]
        )
        self.heads = nn.ModuleList([nn.Linear(48, 1) for _ in range(3)])
        self.long_bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))

    def forward(self, xcat, xnum):
        x = torch.cat([self.embedding(xcat).flatten(1), xnum], dim=1)
        experts = torch.stack([expert(x) for expert in self.experts], dim=1)
        outputs = []
        for task in range(3):
            gate = torch.softmax(self.gates[task](x), dim=1).unsqueeze(2)
            representation = (experts * gate).sum(dim=1)
            logit = self.heads[task](representation).squeeze(1)
            if task == 0:
                logit = logit + self.long_bias
            outputs.append(logit)
        return tuple(outputs)


class LatentUserItemMF(nn.Module):
    """Specialized user-video and user-author latent preference model."""

    def __init__(self, rank=24):
        super().__init__()
        user_card, video_card, author_card = cards[0], cards[1], cards[2]
        self.user = nn.Embedding(user_card, rank)
        self.video = nn.Embedding(video_card, rank)
        self.author = nn.Embedding(author_card, rank)
        self.user_bias = nn.Embedding(user_card, 1)
        self.video_bias = nn.Embedding(video_card, 1)
        self.author_bias = nn.Embedding(author_card, 1)
        self.side = nn.Sequential(
            nn.Linear(n_num + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))

        nn.init.normal_(self.user.weight, std=0.03)
        nn.init.normal_(self.video.weight, std=0.03)
        nn.init.normal_(self.author.weight, std=0.03)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.video_bias.weight)
        nn.init.zeros_(self.author_bias.weight)

    def forward(self, xcat, xnum):
        # Remove global field offsets before indexing field-specific tables.
        uid = xcat[:, 0]
        vid = xcat[:, 1] - offsets[1]
        aid = xcat[:, 2] - offsets[2]
        tab = (xcat[:, 3] - offsets[3]).float()
        duration = (xcat[:, 4] - offsets[4]).float()

        u = self.user(uid)
        interaction = (u * self.video(vid)).sum(1)
        interaction = interaction + 0.65 * (u * self.author(aid)).sum(1)
        biases = (
            self.user_bias(uid).squeeze(1)
            + self.video_bias(vid).squeeze(1)
            + 0.5 * self.author_bias(aid).squeeze(1)
        )
        side = self.side(
            torch.cat(
                [
                    xnum,
                    (tab / max(cards[3] - 1, 1)).unsqueeze(1),
                    (duration / max(cards[4] - 1, 1)).unsqueeze(1),
                ],
                dim=1,
            )
        ).squeeze(1)
        return self.bias + interaction + biases + side


aux_click = np.asarray(train.aux["is_click"], dtype=np.float32)
aux_like = np.asarray(train.aux["is_like"], dtype=np.float32)


def train_model(model, seed, multitask=False):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.25e-3, weight_decay=2e-6
    )

    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(n_train)
        loss_sum = 0.0
        weight_sum = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            cat = torch.from_numpy(xcat_train[idx])
            num = torch.from_numpy(xnum_train[idx])
            target = torch.from_numpy(y_train[idx])
            weight = torch.from_numpy(w_train[idx])

            optimizer.zero_grad(set_to_none=True)
            output = model(cat, num)

            if multitask:
                long_logit, click_logit, like_logit = output
                long_loss = F.binary_cross_entropy_with_logits(
                    long_logit, target, reduction="none"
                )
                click_target = torch.from_numpy(aux_click[idx])
                like_target = torch.from_numpy(aux_like[idx])
                click_loss = F.binary_cross_entropy_with_logits(
                    click_logit, click_target, reduction="none"
                )
                like_loss = F.binary_cross_entropy_with_logits(
                    like_logit, like_target, reduction="none"
                )
                row_loss = long_loss + 0.18 * click_loss + 0.12 * like_loss
            else:
                row_loss = F.binary_cross_entropy_with_logits(
                    output, target, reduction="none"
                )

            loss = (row_loss * weight).sum() / weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += float((row_loss.detach() * weight).sum())
            weight_sum += float(weight.sum())

        print(
            "TRAIN family=%s epoch=%d loss=%.6f"
            % (model.__class__.__name__, epoch + 1, loss_sum / weight_sum),
            flush=True,
        )

    return model


def predict_model(model, xcat, xnum, multitask=False):
    result = np.empty(xcat.shape[0], dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, xcat.shape[0], 16384):
            end = min(start + 16384, xcat.shape[0])
            output = model(
                torch.from_numpy(xcat[start:end]),
                torch.from_numpy(xnum[start:end]),
            )
            if multitask:
                output = output[0]
            result[start:end] = output.cpu().numpy()
    return result


families = [
    ("field_weighted_fm", lambda: FieldWeightedFM(rank=16), False),
    ("dcnv2", lambda: DCNv2(rank=10, cross_rank=32), False),
    ("autoint", lambda: AutoInt(rank=12), False),
    ("mmoe", lambda: MMoE(rank=10, n_experts=4), True),
    ("latent_user_item_mf", lambda: LatentUserItemMF(rank=24), False),
]

valid_predictions = {}
test_predictions = {}

for family_index, (name, constructor, multitask) in enumerate(families):
    torch.manual_seed(SEED + 97 * family_index)
    model = constructor()
    model = train_model(model, SEED + 1000 + family_index, multitask=multitask)
    valid_predictions[name] = predict_model(
        model, xcat_valid, xnum_valid, multitask=multitask
    )
    test_predictions[name] = predict_model(
        model, xcat_test, xnum_test, multitask=multitask
    )
    del model


def within_user_rank(user_ids, scores):
    """Tie-aware percentile rank calculated independently inside each user."""
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((scores, user_ids))
    sorted_users = user_ids[order]
    sorted_scores = scores[order]

    user_starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    user_ends = np.r_[user_starts[1:], n]
    user_lengths = user_ends - user_starts

    tie_starts = np.flatnonzero(
        np.r_[
            True,
            (sorted_users[1:] != sorted_users[:-1])
            | (sorted_scores[1:] != sorted_scores[:-1]),
        ]
    )
    tie_ends = np.r_[tie_starts[1:], n]
    tie_lengths = tie_ends - tie_starts
    tie_midpoints = 0.5 * (tie_starts + tie_ends - 1)
    absolute_midpoints = np.repeat(tie_midpoints, tie_lengths)

    repeated_starts = np.repeat(user_starts, user_lengths)
    repeated_denominators = np.repeat(
        np.maximum(user_lengths - 1, 1), user_lengths
    )
    ranked_sorted = (
        absolute_midpoints - repeated_starts
    ) / repeated_denominators

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

alphas = [0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
candidate_scores = {}

best_primary = -np.inf
best_name = None
best_alpha = None
best_valid = None
best_test = None
best_raw_valid = None

for name, _, _ in families:
    raw_valid = valid_predictions[name]
    raw_test = test_predictions[name]

    raw_metrics = evaluate(valid.user_id, valid.y, raw_valid)
    candidate_scores[name + "_raw"] = float(raw_metrics["primary"])

    model_valid_rank = within_user_rank(valid.user_id, raw_valid)
    model_test_rank = within_user_rank(test.user_id, raw_test)

    for alpha in alphas:
        blend_valid = (
            alpha * model_valid_rank + (1.0 - alpha) * inc_valid_rank
        )
        blend_metrics = evaluate(valid.user_id, valid.y, blend_valid)
        key = "%s_blend_a%.2f" % (name, alpha)
        candidate_scores[key] = float(blend_metrics["primary"])

        if float(blend_metrics["primary"]) > best_primary:
            best_primary = float(blend_metrics["primary"])
            best_name = name
            best_alpha = alpha
            best_valid = blend_valid.copy()
            best_test = (
                alpha * model_test_rank + (1.0 - alpha) * inc_test_rank
            )
            best_raw_valid = raw_valid.copy()

final_metrics = evaluate(valid.user_id, valid.y, best_valid)

family_raw_summary = {
    name: candidate_scores[name + "_raw"] for name, _, _ in families
}
print(
    "FINDINGS "
    + json.dumps(
        {
            "raw_family_primary": family_raw_summary,
            "selected_family": best_name,
            "selected_model_weight": best_alpha,
        },
        sort_keys=True,
    ),
    flush=True,
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True), flush=True)

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
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    ),
    flush=True,
)