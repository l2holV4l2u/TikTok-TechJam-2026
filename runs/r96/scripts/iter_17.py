import os
import time
import json
import gc
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 91731
BATCH_SIZE = 4096
INFER_BATCH_SIZE = 8192
EPOCHS = 2
EMBED_DIM = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "hour",
    "upload_type",
    "music_type",
    "video_type",
    "user_active_degree",
    "is_video_author",
    "is_live_streamer",
    "onehot_feat3",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

FIELD_INDEX = {name: i for i, name in enumerate(CAT_FIELDS)}


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=np.bool_)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    first = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=np.bool_)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    group_ends = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, group_ends])
    row_sizes = np.repeat(sizes, sizes)

    position = np.arange(n, dtype=np.int64) - first
    ranked = (position.astype(np.float64) + 0.5) / row_sizes

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def fit_numeric_state(train):
    state = {}
    for name in NUM_FIELDS:
        x = np.asarray(train.num[name], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        mean = float(x.mean())
        std = max(float(x.std()), 1e-5)
        state[name] = (mean, std)
    return state


def make_features(split, numeric_state):
    cats = np.column_stack([
        np.asarray(split.X[name], dtype=np.int64)
        for name in CAT_FIELDS
    ])

    nums = np.empty((len(split), len(NUM_FIELDS)), dtype=np.float32)
    for j, name in enumerate(NUM_FIELDS):
        x = np.asarray(split.num[name], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        mean, std = numeric_state[name]
        nums[:, j] = np.clip((x - mean) / std, -8.0, 8.0)

    return cats, nums


def build_hardness_prior(train):
    y = np.asarray(train.y, dtype=np.float64)

    video = np.asarray(train.video_id, dtype=np.int64)
    n_video = int(FEATURE_CARDINALITIES["video_id"])
    video_count = np.bincount(video, minlength=n_video).astype(np.float64)
    video_pos = np.bincount(
        video, weights=y, minlength=n_video
    ).astype(np.float64)
    global_rate = float(y.mean())
    video_rate = (
        video_pos + 20.0 * global_rate
    ) / (video_count + 20.0)

    author = np.asarray(train.X["author_id"], dtype=np.int64)
    n_author = int(FEATURE_CARDINALITIES["author_id"])
    author_count = np.bincount(
        author, minlength=n_author
    ).astype(np.float64)
    author_pos = np.bincount(
        author, weights=y, minlength=n_author
    ).astype(np.float64)
    author_rate = (
        author_pos + 30.0 * global_rate
    ) / (author_count + 30.0)

    duration = np.asarray(train.X["duration_bucket"], dtype=np.int64)
    n_duration = int(FEATURE_CARDINALITIES["duration_bucket"])
    duration_count = np.bincount(
        duration, minlength=n_duration
    ).astype(np.float64)
    duration_pos = np.bincount(
        duration, weights=y, minlength=n_duration
    ).astype(np.float64)
    duration_rate = (
        duration_pos + 50.0 * global_rate
    ) / (duration_count + 50.0)

    row_hardness = (
        0.60 * video_rate[video]
        + 0.30 * author_rate[author]
        + 0.10 * duration_rate[duration]
    )
    return row_hardness.astype(np.float32)


def prepare_pair_sampler(train):
    users = np.asarray(train.user_id, dtype=np.int64)
    y = np.asarray(train.y, dtype=np.int8)
    user_cardinality = int(FEATURE_CARDINALITIES["user_id"])

    negative_rows = np.flatnonzero(y == 0)
    neg_order = np.argsort(users[negative_rows], kind="stable")
    negative_rows = negative_rows[neg_order]
    negative_users = users[negative_rows]

    negative_count = np.bincount(
        negative_users,
        minlength=user_cardinality,
    ).astype(np.int64)
    negative_start = np.cumsum(
        np.r_[0, negative_count[:-1]]
    ).astype(np.int64)

    positive_rows = np.flatnonzero(y == 1)
    positive_rows = positive_rows[
        negative_count[users[positive_rows]] > 0
    ]

    return {
        "users": users,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "negative_count": negative_count,
        "negative_start": negative_start,
    }


def sample_pairs(sampler, row_hardness, rng):
    users = sampler["users"]
    positives = sampler["positive_rows"]
    negative_rows = sampler["negative_rows"]
    negative_count = sampler["negative_count"]
    negative_start = sampler["negative_start"]

    positive_users = users[positives]
    count = negative_count[positive_users]
    start = negative_start[positive_users]
    n_positive = len(positives)

    # Four logged negatives are sampled for every positive. The highest-prior
    # one is a train-only hard negative; another uniformly sampled negative
    # preserves broad pair coverage.
    random_offsets = (
        rng.random((n_positive, 4)) * count[:, None]
    ).astype(np.int64)
    candidates = negative_rows[start[:, None] + random_offsets]

    candidate_hardness = row_hardness[candidates]
    hard_column = np.argmax(candidate_hardness, axis=1)
    hard_negative = candidates[
        np.arange(n_positive, dtype=np.int64),
        hard_column,
    ]
    random_negative = candidates[:, 0]

    pair_positive = np.repeat(positives, 2)
    pair_negative = np.column_stack((
        random_negative,
        hard_negative,
    )).reshape(-1)

    # Hard-negative pairs receive greater weight because confusing negatives
    # are disproportionately likely to occupy the metric's top five ranks.
    pair_weight = np.tile(
        np.asarray([1.0, 1.5], dtype=np.float32),
        n_positive,
    )

    permutation = rng.permutation(len(pair_positive))
    return (
        pair_positive[permutation],
        pair_negative[permutation],
        pair_weight[permutation],
    )


class FieldEmbeddings(nn.Module):
    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.tables = nn.ModuleList([
            nn.Embedding(
                int(FEATURE_CARDINALITIES[name]),
                dim,
            )
            for name in CAT_FIELDS
        ])
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.02)

    def forward(self, cats):
        return torch.stack([
            table(cats[:, j])
            for j, table in enumerate(self.tables)
        ], dim=1)


class FiBiNETScorer(nn.Module):
    """SENet field weighting followed by explicit bilinear interactions."""

    def __init__(self):
        super().__init__()
        self.fields = FieldEmbeddings()
        n_fields = len(CAT_FIELDS)

        hidden = max(4, n_fields // 3)
        self.squeeze = nn.Sequential(
            nn.Linear(n_fields, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_fields),
            nn.Sigmoid(),
        )

        left, right = np.triu_indices(n_fields, k=1)
        self.register_buffer(
            "pair_left",
            torch.from_numpy(left.astype(np.int64)),
        )
        self.register_buffer(
            "pair_right",
            torch.from_numpy(right.astype(np.int64)),
        )

        n_pairs = len(left)
        input_dim = n_fields * EMBED_DIM + n_pairs + len(NUM_FIELDS)
        self.head = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, cats, nums):
        embedding = self.fields(cats)
        squeeze_value = embedding.mean(dim=2)
        gates = self.squeeze(squeeze_value).unsqueeze(-1)
        weighted = embedding * gates

        interactions = (
            weighted[:, self.pair_left, :]
            * weighted[:, self.pair_right, :]
        ).sum(dim=2)

        x = torch.cat((
            weighted.flatten(1),
            interactions,
            nums,
        ), dim=1)
        return self.head(x).squeeze(1)


class TwoTowerScorer(nn.Module):
    """Separately forms user/context and candidate representations."""

    def __init__(self):
        super().__init__()
        self.fields = FieldEmbeddings()

        self.query_indices = [
            FIELD_INDEX[name] for name in [
                "user_id",
                "tab",
                "hour",
                "user_active_degree",
                "is_video_author",
                "is_live_streamer",
                "onehot_feat3",
                "onehot_feat8",
            ]
        ]
        self.item_indices = [
            FIELD_INDEX[name] for name in [
                "video_id",
                "author_id",
                "tag",
                "duration_bucket",
                "upload_type",
                "music_type",
                "video_type",
            ]
        ]

        query_dim = len(self.query_indices) * EMBED_DIM + 4
        item_dim = len(self.item_indices) * EMBED_DIM + 1

        self.query_tower = nn.Sequential(
            nn.Linear(query_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 48),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 48),
        )
        self.context_bias = nn.Sequential(
            nn.Linear(len(NUM_FIELDS), 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, cats, nums):
        embedding = self.fields(cats)

        query = torch.cat((
            embedding[:, self.query_indices, :].flatten(1),
            nums[:, 1:],
        ), dim=1)
        item = torch.cat((
            embedding[:, self.item_indices, :].flatten(1),
            nums[:, :1],
        ), dim=1)

        query = nn.functional.normalize(
            self.query_tower(query), dim=1
        )
        item = nn.functional.normalize(
            self.item_tower(item), dim=1
        )

        scale = 8.0 * torch.nn.functional.softplus(self.scale)
        return (
            scale * (query * item).sum(dim=1)
            + self.context_bias(nums).squeeze(1)
        )


class NeuralAdditiveScorer(nn.Module):
    """A low-variance additive model with nonlinear numeric shape functions."""

    def __init__(self):
        super().__init__()
        self.cat_terms = nn.ModuleList([
            nn.Embedding(
                int(FEATURE_CARDINALITIES[name]),
                1,
            )
            for name in CAT_FIELDS
        ])
        for table in self.cat_terms:
            nn.init.zeros_(table.weight)

        self.numeric_terms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 16),
                nn.Tanh(),
                nn.Linear(16, 8),
                nn.Tanh(),
                nn.Linear(8, 1),
            )
            for _ in NUM_FIELDS
        ])
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, cats, nums):
        result = self.bias.expand(cats.shape[0])
        for j, table in enumerate(self.cat_terms):
            result = result + table(cats[:, j]).squeeze(1)
        for j, term in enumerate(self.numeric_terms):
            result = result + term(nums[:, j:j + 1]).squeeze(1)
        return result


def train_pairwise_model(
    model,
    train_cats,
    train_nums,
    sampler,
    row_hardness,
    seed_offset,
):
    torch.manual_seed(SEED + seed_offset)
    rng = np.random.default_rng(SEED + seed_offset)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=2e-5,
    )

    losses = []
    hard_pair_margins = []

    for epoch in range(EPOCHS):
        pos_rows, neg_rows, weights = sample_pairs(
            sampler, row_hardness, rng
        )

        model.train()
        total_loss = 0.0
        total_weight = 0.0
        margin_sum = 0.0
        margin_count = 0

        for start in range(0, len(pos_rows), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(pos_rows))
            p = pos_rows[start:end]
            n = neg_rows[start:end]

            pos_cats = torch.from_numpy(train_cats[p]).long()
            neg_cats = torch.from_numpy(train_cats[n]).long()
            pos_nums = torch.from_numpy(train_nums[p])
            neg_nums = torch.from_numpy(train_nums[n])
            batch_weight = torch.from_numpy(weights[start:end])

            pos_score = model(pos_cats, pos_nums)
            neg_score = model(neg_cats, neg_nums)
            margin = pos_score - neg_score

            element_loss = nn.functional.softplus(-margin)
            loss = (
                element_loss * batch_weight
            ).sum() / batch_weight.sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(
                (element_loss * batch_weight).sum().detach()
            )
            total_weight += float(batch_weight.sum())
            margin_sum += float(margin.detach().sum())
            margin_count += len(margin)

        losses.append(total_loss / max(total_weight, 1.0))
        hard_pair_margins.append(
            margin_sum / max(margin_count, 1)
        )

    return {
        "loss": losses,
        "mean_pair_margin": hard_pair_margins,
    }


@torch.no_grad()
def predict_model(model, cats, nums):
    model.eval()
    result = np.empty(len(cats), dtype=np.float64)

    for start in range(0, len(cats), INFER_BATCH_SIZE):
        end = min(start + INFER_BATCH_SIZE, len(cats))
        batch_cats = torch.from_numpy(cats[start:end]).long()
        batch_nums = torch.from_numpy(nums[start:end])
        result[start:end] = model(
            batch_cats, batch_nums
        ).cpu().numpy()

    return result


train = load("train")
valid = load("valid")
test = load("test")

numeric_state = fit_numeric_state(train)
train_cats, train_nums = make_features(train, numeric_state)
valid_cats, valid_nums = make_features(valid, numeric_state)
test_cats, test_nums = make_features(test, numeric_state)

row_hardness = build_hardness_prior(train)
sampler = prepare_pair_sampler(train)

constructors = [
    ("pairwise_fibinet", FiBiNETScorer),
    ("pairwise_two_tower", TwoTowerScorer),
    ("pairwise_neural_additive", NeuralAdditiveScorer),
]

family_valid = {}
family_test = {}
training_findings = {}
failures = {}

for family_index, (name, constructor) in enumerate(constructors):
    try:
        torch.manual_seed(SEED + 1000 * (family_index + 1))
        model = constructor()
        training_findings[name] = train_pairwise_model(
            model=model,
            train_cats=train_cats,
            train_nums=train_nums,
            sampler=sampler,
            row_hardness=row_hardness,
            seed_offset=1000 * (family_index + 1),
        )
        family_valid[name] = predict_model(
            model, valid_cats, valid_nums
        )
        family_test[name] = predict_model(
            model, test_cats, test_nums
        )
        del model
        gc.collect()
    except Exception as exc:
        failures[name] = repr(exc)
        gc.collect()

if not family_valid:
    raise RuntimeError(
        "All pairwise model families failed: " + repr(failures)
    )

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

family_valid_rank = {
    name: rank_percentile(valid.user_id, scores)
    for name, scores in family_valid.items()
}
family_test_rank = {
    name: rank_percentile(test.user_id, scores)
    for name, scores in family_test.items()
}

candidate_valid = {"incumbent": inc_valid}
candidate_test = {"incumbent": inc_test}
candidate_raw = {"incumbent": inc_valid}

for name in family_valid:
    candidate_valid[name + "_standalone"] = family_valid[name]
    candidate_test[name + "_standalone"] = family_test[name]
    candidate_raw[name + "_standalone"] = family_valid[name]

    for alpha in (0.05, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70):
        key = f"{name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * family_valid_rank[name]
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * family_test_rank[name]
        )
        candidate_raw[key] = family_valid[name]

if len(family_valid_rank) >= 2:
    family_names = sorted(family_valid_rank)
    ensemble_valid = np.mean(
        np.stack([
            family_valid_rank[name] for name in family_names
        ]),
        axis=0,
    )
    ensemble_test = np.mean(
        np.stack([
            family_test_rank[name] for name in family_names
        ]),
        axis=0,
    )

    candidate_valid["pairwise_family_ensemble"] = ensemble_valid
    candidate_test["pairwise_family_ensemble"] = ensemble_test
    candidate_raw["pairwise_family_ensemble"] = ensemble_valid

    for alpha in (0.10, 0.20, 0.30, 0.40, 0.55):
        key = f"pairwise_ensemble_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * ensemble_valid
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * ensemble_test
        )
        candidate_raw[key] = ensemble_valid

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, scores)
    for name, scores in candidate_valid.items()
}

best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_valid[best_name]
best_test = candidate_test[best_name]

correlations_with_incumbent = {
    name: float(np.corrcoef(
        inc_valid_rank,
        family_valid_rank[name],
    )[0, 1])
    for name in family_valid_rank
}

pairwise_correlations = {}
family_names = sorted(family_valid_rank)
for i in range(len(family_names)):
    for j in range(i + 1, len(family_names)):
        left = family_names[i]
        right = family_names[j]
        pairwise_correlations[f"{left}__{right}"] = float(
            np.corrcoef(
                family_valid_rank[left],
                family_valid_rank[right],
            )[0, 1]
        )

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "eligible_positive_rows": int(
        len(sampler["positive_rows"])
    ),
    "negative_rows": int(
        len(sampler["negative_rows"])
    ),
    "hardness_mean": float(row_hardness.mean()),
    "hardness_std": float(row_hardness.std()),
    "training": training_findings,
    "failures": failures,
    "rank_correlations_with_incumbent":
        correlations_with_incumbent,
    "pairwise_family_rank_correlations":
        pairwise_correlations,
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_name != "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(
                candidate_raw[best_name],
                dtype=np.float64,
            ),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))