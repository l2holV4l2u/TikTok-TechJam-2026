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
SEED = 271828
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
train_dates = np.asarray(train.date, dtype=np.int64)

# Four-day half-life emphasizes behavior near the deployment boundary.
sample_weight = np.exp2(
    (train_dates - train_dates.max()).astype(np.float32) / 4.0
)
sample_weight /= sample_weight.mean()
sample_weight = sample_weight.astype(np.float32)

cards = {f: int(FEATURE_CARDINALITIES[f]) for f in FIELDS}


def field_matrix(split):
    return np.ascontiguousarray(
        np.stack(
            [np.asarray(split.X[f], dtype=np.int64) for f in FIELDS],
            axis=1,
        )
    )


x_train = field_matrix(train)
x_valid = field_matrix(valid)
x_test = field_matrix(test)

uid_train = np.asarray(train.user_id, dtype=np.int64)
uid_valid = np.asarray(valid.user_id, dtype=np.int64)
uid_test = np.asarray(test.user_id, dtype=np.int64)

# Structures for vectorized sampling of another impression from the same user.
sort_order = np.argsort(uid_train, kind="stable")
sorted_users = uid_train[sort_order]
group_starts = np.flatnonzero(
    np.r_[True, sorted_users[1:] != sorted_users[:-1]]
)
group_ends = np.r_[group_starts[1:], len(sort_order)]
group_counts = group_ends - group_starts

unique_users = sorted_users[group_starts]
max_user_id = max(
    int(uid_train.max(initial=0)),
    int(uid_valid.max(initial=0)),
    int(uid_test.max(initial=0)),
)
user_to_group = np.full(max_user_id + 1, -1, dtype=np.int32)
user_to_group[unique_users] = np.arange(len(unique_users), dtype=np.int32)
row_group = user_to_group[uid_train]

rng = np.random.default_rng(SEED)


def sample_opposite_pairs(anchor_indices):
    """Sample same-user partners, retaining only label-discordant pairs."""
    groups = row_group[anchor_indices]
    counts = group_counts[groups]
    starts = group_starts[groups]

    offsets = (rng.random(len(anchor_indices)) * counts).astype(np.int64)
    partners = sort_order[starts + offsets]

    # A second draw raises yield for users with imbalanced labels.
    same = y_train[partners] == y_train[anchor_indices]
    if np.any(same):
        offsets2 = (rng.random(int(same.sum())) * counts[same]).astype(np.int64)
        partners[same] = sort_order[starts[same] + offsets2]

    discordant = y_train[partners] != y_train[anchor_indices]
    a = anchor_indices[discordant]
    b = partners[discordant]

    a_is_positive = y_train[a] > y_train[b]
    positive = np.where(a_is_positive, a, b)
    negative = np.where(a_is_positive, b, a)
    return positive, negative


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort(
        (np.arange(n, dtype=np.int64), scores, user_ids)
    )
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.int64) - repeated_starts

    ordered_ranks = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ordered_ranks[multi] = (
        positions[multi] / (repeated_lengths[multi] - 1.0)
    )

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ordered_ranks
    return ranks


@torch.inference_mode()
def predict(model, x):
    model.eval()
    result = np.empty(len(x), dtype=np.float64)
    for lo in range(0, len(x), PRED_BATCH_SIZE):
        hi = min(lo + PRED_BATCH_SIZE, len(x))
        xb = torch.from_numpy(x[lo:hi])
        result[lo:hi] = (
            model(xb).detach().cpu().numpy().astype(np.float64)
        )
    return result


class WideCrossRanker(nn.Module):
    """
    Additive memorization model with exact low-cardinality crosses.

    Unlike an FM, every selected cross has an independent scalar effect and
    does not have to be represented through a low-rank inner product.
    """

    def __init__(self):
        super().__init__()
        self.main = nn.ModuleList(
            [nn.Embedding(cards[f], 1) for f in FIELDS]
        )

        self.user_tab = nn.Embedding(
            cards["user_id"] * cards["tab"], 1
        )
        self.video_tab = nn.Embedding(
            cards["video_id"] * cards["tab"], 1
        )
        self.author_tab = nn.Embedding(
            cards["author_id"] * cards["tab"], 1
        )
        self.video_duration = nn.Embedding(
            cards["video_id"] * cards["duration_bucket"], 1
        )
        self.author_duration = nn.Embedding(
            cards["author_id"] * cards["duration_bucket"], 1
        )
        self.bias = nn.Parameter(torch.zeros(1))

        for emb in self.main:
            nn.init.zeros_(emb.weight)
        for emb in [
            self.user_tab,
            self.video_tab,
            self.author_tab,
            self.video_duration,
            self.author_duration,
        ]:
            nn.init.zeros_(emb.weight)

    def forward(self, x):
        u, v, a, tab, dur = [x[:, i] for i in range(5)]
        score = self.bias.expand(x.shape[0])

        for i, emb in enumerate(self.main):
            score = score + emb(x[:, i]).squeeze(1)

        score = score + self.user_tab(
            u * cards["tab"] + tab
        ).squeeze(1)
        score = score + self.video_tab(
            v * cards["tab"] + tab
        ).squeeze(1)
        score = score + self.author_tab(
            a * cards["tab"] + tab
        ).squeeze(1)
        score = score + self.video_duration(
            v * cards["duration_bucket"] + dur
        ).squeeze(1)
        score = score + self.author_duration(
            a * cards["duration_bucket"] + dur
        ).squeeze(1)
        return score


class BPRMatrixFactorizer(nn.Module):
    """
    Low-rank personalized affinity model.

    User-video and user-author affinities form the score directly; the linear
    side terms provide stable fallback behavior for sparse users.
    """

    def __init__(self, rank=24):
        super().__init__()
        self.user = nn.Embedding(cards["user_id"], rank)
        self.video = nn.Embedding(cards["video_id"], rank)
        self.author = nn.Embedding(cards["author_id"], rank)

        self.video_bias = nn.Embedding(cards["video_id"], 1)
        self.author_bias = nn.Embedding(cards["author_id"], 1)
        self.tab_bias = nn.Embedding(cards["tab"], 1)
        self.duration_bias = nn.Embedding(cards["duration_bucket"], 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user.weight, std=0.03)
        nn.init.normal_(self.video.weight, std=0.03)
        nn.init.normal_(self.author.weight, std=0.03)
        for emb in [
            self.video_bias,
            self.author_bias,
            self.tab_bias,
            self.duration_bias,
        ]:
            nn.init.zeros_(emb.weight)

    def forward(self, x):
        u, v, a, tab, dur = [x[:, i] for i in range(5)]
        ue = self.user(u)

        affinity = (ue * self.video(v)).sum(dim=1)
        affinity = affinity + 0.6 * (ue * self.author(a)).sum(dim=1)

        return (
            self.global_bias
            + affinity
            + self.video_bias(v).squeeze(1)
            + self.author_bias(a).squeeze(1)
            + self.tab_bias(tab).squeeze(1)
            + self.duration_bias(dur).squeeze(1)
        )


class AutoIntRanker(nn.Module):
    """
    Self-attention over fields followed by a nonlinear prediction tower.

    Attention lets the model condition which field interactions matter on the
    actual impression instead of assigning every field pair a fixed form.
    """

    def __init__(self, emb_dim=16, heads=4):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cards[f], emb_dim) for f in FIELDS]
        )
        self.field_embedding = nn.Parameter(
            torch.empty(len(FIELDS), emb_dim)
        )

        self.attn1 = nn.MultiheadAttention(
            emb_dim, heads, dropout=0.0, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            emb_dim, heads, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.linear_terms = nn.ModuleList(
            [nn.Embedding(cards[f], 1) for f in FIELDS]
        )
        self.tower = nn.Sequential(
            nn.Linear(len(FIELDS) * emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.04),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.025)
        nn.init.normal_(self.field_embedding, std=0.02)
        for emb in self.linear_terms:
            nn.init.zeros_(emb.weight)

    def forward(self, x):
        z = torch.stack(
            [emb(x[:, i]) for i, emb in enumerate(self.embeddings)],
            dim=1,
        )
        z = z + self.field_embedding.unsqueeze(0)

        attended, _ = self.attn1(z, z, z, need_weights=False)
        z = self.norm1(z + F.relu(attended))

        attended, _ = self.attn2(z, z, z, need_weights=False)
        z = self.norm2(z + F.relu(attended))

        linear = torch.zeros(
            x.shape[0], dtype=z.dtype, device=z.device
        )
        for i, emb in enumerate(self.linear_terms):
            linear = linear + emb(x[:, i]).squeeze(1)

        return (
            self.bias
            + linear
            + self.tower(z.flatten(1)).squeeze(1)
        )


def train_pairwise(model, epochs, learning_rate):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=2e-6
    )
    n = len(y_train)

    for _ in range(epochs):
        model.train()
        order = rng.permutation(n)

        for lo in range(0, n, BATCH_SIZE):
            anchors = order[lo:lo + BATCH_SIZE]
            positive, negative = sample_opposite_pairs(anchors)
            if len(positive) < 32:
                continue

            xp = torch.from_numpy(x_train[positive])
            xn = torch.from_numpy(x_train[negative])
            weights_np = 0.5 * (
                sample_weight[positive] + sample_weight[negative]
            )
            weights = torch.from_numpy(weights_np)

            optimizer.zero_grad(set_to_none=True)
            margin = model(xp) - model(xn)
            losses = F.softplus(-margin)
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


def train_pointwise(model, epochs, learning_rate):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=2e-6
    )
    n = len(y_train)

    for _ in range(epochs):
        model.train()
        order = rng.permutation(n)

        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:lo + BATCH_SIZE]
            xb = torch.from_numpy(x_train[idx])
            target = torch.from_numpy(y_train[idx])
            weights = torch.from_numpy(sample_weight[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


valid_predictions = {}
test_predictions = {}

torch.manual_seed(SEED + 1)
wide = WideCrossRanker()
train_pairwise(wide, epochs=4, learning_rate=0.0020)
valid_predictions["wide_cross_ranknet"] = predict(wide, x_valid)
test_predictions["wide_cross_ranknet"] = predict(wide, x_test)
del wide

torch.manual_seed(SEED + 2)
bpr = BPRMatrixFactorizer(rank=24)
train_pairwise(bpr, epochs=5, learning_rate=0.0015)
valid_predictions["bpr_matrix_factorization"] = predict(bpr, x_valid)
test_predictions["bpr_matrix_factorization"] = predict(bpr, x_test)
del bpr

torch.manual_seed(SEED + 3)
autoint = AutoIntRanker(emb_dim=16, heads=4)
train_pointwise(autoint, epochs=3, learning_rate=0.0012)
valid_predictions["autoint_pointwise"] = predict(autoint, x_valid)
test_predictions["autoint_pointwise"] = predict(autoint, x_test)
del autoint

shared_dir = os.environ.get("SHARED_ARTIFACTS")
if not shared_dir:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared_dir, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared_dir, "incumbent_test_scores.npy")
).astype(np.float64)

if len(inc_valid) != len(valid_predictions["wide_cross_ranknet"]):
    raise RuntimeError("Incumbent validation prediction length mismatch")
if len(inc_test) != len(test_predictions["wide_cross_ranknet"]):
    raise RuntimeError("Incumbent test prediction length mismatch")

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

candidate_metrics = {}
candidate_valid = {}
candidate_test = {}
candidate_raw = {}
candidate_blended = {}

inc_metrics = evaluate(valid.user_id, valid.y, inc_valid)
candidate_metrics["trusted_incumbent"] = float(inc_metrics["primary"])
candidate_valid["trusted_incumbent"] = inc_valid
candidate_test["trusted_incumbent"] = inc_test
candidate_raw["trusted_incumbent"] = inc_valid
candidate_blended["trusted_incumbent"] = False

blend_alphas = [0.15, 0.25, 0.35, 0.50, 0.65]

for family in valid_predictions:
    raw_valid = valid_predictions[family]
    raw_test = test_predictions[family]

    raw_result = evaluate(valid.user_id, valid.y, raw_valid)
    candidate_metrics[family] = float(raw_result["primary"])
    candidate_valid[family] = raw_valid
    candidate_test[family] = raw_test
    candidate_raw[family] = raw_valid
    candidate_blended[family] = False

    own_valid_rank = within_user_rank(valid.user_id, raw_valid)
    own_test_rank = within_user_rank(test.user_id, raw_test)

    for alpha in blend_alphas:
        name = family + "_incumbent_blend_" + str(alpha)
        blend_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * own_valid_rank
        )
        blend_test = (
            (1.0 - alpha) * inc_test_rank
            + alpha * own_test_rank
        )

        result = evaluate(valid.user_id, valid.y, blend_valid)
        candidate_metrics[name] = float(result["primary"])
        candidate_valid[name] = blend_valid
        candidate_test[name] = blend_test
        candidate_raw[name] = raw_valid
        candidate_blended[name] = True

# Also test a heterogeneous ensemble of all new prediction mechanisms.
new_valid_ranks = [
    within_user_rank(valid.user_id, valid_predictions[name])
    for name in valid_predictions
]
new_test_ranks = [
    within_user_rank(test.user_id, test_predictions[name])
    for name in test_predictions
]
ensemble_valid_rank = np.mean(np.stack(new_valid_ranks), axis=0)
ensemble_test_rank = np.mean(np.stack(new_test_ranks), axis=0)
ensemble_raw_valid = np.mean(
    np.stack([valid_predictions[name] for name in valid_predictions]),
    axis=0,
)

ensemble_result = evaluate(
    valid.user_id, valid.y, ensemble_valid_rank
)
candidate_metrics["three_family_ensemble"] = float(
    ensemble_result["primary"]
)
candidate_valid["three_family_ensemble"] = ensemble_valid_rank
candidate_test["three_family_ensemble"] = ensemble_test_rank
candidate_raw["three_family_ensemble"] = ensemble_raw_valid
candidate_blended["three_family_ensemble"] = False

for alpha in blend_alphas:
    name = "three_family_ensemble_incumbent_blend_" + str(alpha)
    blend_valid = (
        (1.0 - alpha) * inc_valid_rank
        + alpha * ensemble_valid_rank
    )
    blend_test = (
        (1.0 - alpha) * inc_test_rank
        + alpha * ensemble_test_rank
    )
    result = evaluate(valid.user_id, valid.y, blend_valid)

    candidate_metrics[name] = float(result["primary"])
    candidate_valid[name] = blend_valid
    candidate_test[name] = blend_test
    candidate_raw[name] = ensemble_raw_valid
    candidate_blended[name] = True

winner = max(candidate_metrics, key=candidate_metrics.get)
valid_scores = np.asarray(candidate_valid[winner], dtype=np.float64)
test_scores = np.asarray(candidate_test[winner], dtype=np.float64)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

standalone_findings = {
    name: round(candidate_metrics[name], 6)
    for name in [
        "wide_cross_ranknet",
        "bpr_matrix_factorization",
        "autoint_pointwise",
        "three_family_ensemble",
        "trusted_incumbent",
    ]
}
print("FINDINGS " + json.dumps(standalone_findings, sort_keys=True))
print(
    "CANDIDATES "
    + json.dumps(
        {k: round(v, 6) for k, v in candidate_metrics.items()},
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        valid_scores,
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        test_scores,
    )
    if candidate_blended[winner]:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[winner], dtype=np.float64),
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