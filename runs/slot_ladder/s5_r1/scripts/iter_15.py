import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260831
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(12, os.cpu_count() or 8))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "onehot_feat3",
    "upload_type",
    "tab",
    "hour",
    "music_type",
    "user_active_degree",
]

PAIR_SAMPLES = 720000
EPOCHS = 2
BATCH_SIZE = 32768
PRED_BATCH_SIZE = 65536
HALF_LIFE_DAYS = 4.0
DEVICE = torch.device("cpu")


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    ranks_sorted = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks_sorted
    return result


def make_offsets():
    cardinalities = np.asarray(
        [FEATURE_CARDINALITIES[f] for f in FIELDS],
        dtype=np.int64,
    )
    offsets = np.r_[0, np.cumsum(cardinalities[:-1])].astype(np.int64)
    return offsets, cardinalities, int(cardinalities.sum())


OFFSETS, CARDINALITIES, TOTAL_CARDINALITY = make_offsets()


def encode_split(split):
    columns = []
    for j, field in enumerate(FIELDS):
        x = np.asarray(split.X[field], dtype=np.int64)
        x = np.minimum(np.maximum(x, 0), CARDINALITIES[j] - 1)
        columns.append(x + OFFSETS[j])
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int64)


def construct_conditional_pairs(train):
    users = np.asarray(train.user_id, dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)
    n_users = int(users.max()) + 1

    # Sorting by (user, label) creates contiguous negative and positive blocks
    # for every user, allowing fully vectorized random pair generation.
    sorted_rows = np.lexsort(
        (np.arange(len(users), dtype=np.int64), labels, users)
    )
    total_count = np.bincount(users, minlength=n_users).astype(np.int64)
    positive_count = np.bincount(
        users, weights=labels, minlength=n_users
    ).astype(np.int64)
    negative_count = total_count - positive_count

    starts = np.r_[0, np.cumsum(total_count[:-1])].astype(np.int64)
    positive_starts = starts + negative_count

    eligible = np.flatnonzero(
        (positive_count > 0) & (negative_count > 0)
    )
    if len(eligible) == 0:
        raise RuntimeError("No users with both labels in training")

    # GAUC weights users by positive count. Sampling users proportionally to
    # positives approximates that aggregation while each sampled comparison
    # remains strictly conditional on one user.
    probabilities = positive_count[eligible].astype(np.float64)
    probabilities /= probabilities.sum()

    rng = np.random.default_rng(SEED)
    sampled_users = rng.choice(
        eligible,
        size=PAIR_SAMPLES,
        replace=True,
        p=probabilities,
    )

    pos_offsets = (
        rng.random(PAIR_SAMPLES) * positive_count[sampled_users]
    ).astype(np.int64)
    neg_offsets = (
        rng.random(PAIR_SAMPLES) * negative_count[sampled_users]
    ).astype(np.int64)

    positive_rows = sorted_rows[
        positive_starts[sampled_users] + pos_offsets
    ]
    negative_rows = sorted_rows[
        starts[sampled_users] + neg_offsets
    ]

    last_date = int(np.max(train.date))
    pos_age = (
        last_date - np.asarray(train.date, dtype=np.int64)[positive_rows]
    ).astype(np.float32)
    neg_age = (
        last_date - np.asarray(train.date, dtype=np.int64)[negative_rows]
    ).astype(np.float32)

    pair_age = 0.5 * (pos_age + neg_age)
    weights = np.power(
        0.5, pair_age / HALF_LIFE_DAYS
    ).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-6)

    return positive_rows, negative_rows, weights, eligible


class AdditiveUtility(nn.Module):
    def __init__(self, total_cardinality):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Embedding(total_cardinality, 1)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.bias + self.linear(x).squeeze(-1).sum(dim=1)


class FMUtility(nn.Module):
    def __init__(self, total_cardinality, dim=12):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.025)

    def forward(self, x):
        linear = self.linear(x).squeeze(-1).sum(dim=1)
        emb = self.embedding(x)
        summed = emb.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


class NonlinearUtility(nn.Module):
    def __init__(self, total_cardinality, n_fields, dim=8):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, dim)
        self.network = nn.Sequential(
            nn.Linear(n_fields * dim, 96),
            nn.SiLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )
        nn.init.normal_(self.embedding.weight, std=0.025)
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        emb = self.embedding(x).reshape(x.shape[0], -1)
        return self.network(emb).squeeze(1)


def train_pairwise(model, x_train, pos_rows, neg_rows, pair_weights, lr):
    model.to(DEVICE)
    model.train()

    x_tensor = torch.from_numpy(x_train)
    pos_tensor = torch.from_numpy(
        np.asarray(pos_rows, dtype=np.int64)
    )
    neg_tensor = torch.from_numpy(
        np.asarray(neg_rows, dtype=np.int64)
    )
    weight_tensor = torch.from_numpy(
        np.asarray(pair_weights, dtype=np.float32)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=2e-6,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + 17)

    n = len(pos_rows)
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for begin in range(0, n, BATCH_SIZE):
            batch = permutation[begin:begin + BATCH_SIZE]
            xp = x_tensor[pos_tensor[batch]].to(DEVICE)
            xn = x_tensor[neg_tensor[batch]].to(DEVICE)
            weights = weight_tensor[batch].to(DEVICE)

            positive_score = model(xp)
            negative_score = model(xn)
            loss = (
                F.softplus(-(positive_score - negative_score)) * weights
            ).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_model(model, x):
    model.eval()
    x_tensor = torch.from_numpy(x)
    result = np.empty(len(x), dtype=np.float64)
    for begin in range(0, len(x), PRED_BATCH_SIZE):
        end = min(begin + PRED_BATCH_SIZE, len(x))
        batch = x_tensor[begin:end].to(DEVICE)
        result[begin:end] = (
            model(batch).detach().cpu().numpy().astype(np.float64)
        )
    return result


train = load("train")
valid = load("valid")

x_train = encode_split(train)
x_valid = encode_split(valid)

positive_rows, negative_rows, pair_weights, eligible_users = (
    construct_conditional_pairs(train)
)

model_specs = {
    "conditional_additive": (
        AdditiveUtility(TOTAL_CARDINALITY),
        0.012,
    ),
    "conditional_fm": (
        FMUtility(TOTAL_CARDINALITY, dim=12),
        0.003,
    ),
    "conditional_nonlinear": (
        NonlinearUtility(
            TOTAL_CARDINALITY,
            len(FIELDS),
            dim=8,
        ),
        0.002,
    ),
}

models = {}
valid_own_ranks = {}

for name, (model, learning_rate) in model_specs.items():
    model = train_pairwise(
        model,
        x_train,
        positive_rows,
        negative_rows,
        pair_weights,
        learning_rate,
    )
    models[name] = model
    logits = predict_model(model, x_valid)
    valid_own_ranks[name] = within_user_rank(valid.user_id, logits)

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not os.path.exists(incumbent_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores missing")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
if len(incumbent_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")

incumbent_valid_rank = within_user_rank(
    valid.user_id, incumbent_valid
)

candidate_scores = {}
candidate_metrics = {}
candidate_recipes = {}

for family, own_rank in valid_own_ranks.items():
    standalone = family + "_standalone"
    candidate_scores[standalone] = own_rank
    candidate_metrics[standalone] = float(
        evaluate(valid.user_id, valid.y, own_rank)["primary"]
    )
    candidate_recipes[standalone] = (family, 1.0, False)

    for own_weight in (0.15, 0.30, 0.45, 0.60):
        candidate_name = (
            f"{family}_incumbent_borda_w{own_weight:.2f}"
        )
        blended = (
            own_weight * own_rank
            + (1.0 - own_weight) * incumbent_valid_rank
        )
        candidate_scores[candidate_name] = blended
        candidate_metrics[candidate_name] = float(
            evaluate(valid.user_id, valid.y, blended)["primary"]
        )
        candidate_recipes[candidate_name] = (
            family,
            own_weight,
            True,
        )

winner = max(candidate_metrics, key=candidate_metrics.get)
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)
winner_family, winner_weight, uses_incumbent = candidate_recipes[winner]

standalone_results = {
    name: candidate_metrics[name + "_standalone"]
    for name in model_specs
}

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "best_standalone": max(
                standalone_results,
                key=standalone_results.get,
            ),
            "eligible_pairwise_users": int(len(eligible_users)),
            "pair_weight_mean": float(pair_weights.mean()),
            "pair_weight_p10": float(
                np.quantile(pair_weights, 0.10)
            ),
            "incumbent_check": float(
                evaluate(
                    valid.user_id,
                    valid.y,
                    incumbent_valid,
                )["primary"]
            ),
        },
        separators=(",", ":"),
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_metrics.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if uses_incumbent:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(
                valid_own_ranks[winner_family],
                dtype=np.float64,
            ),
        )

# Only the selected family is needed for test inference. No test labels are
# accessed, and all fitted parameters came solely from the training split.
test = load("test")
x_test = encode_split(test)
winner_test_logits = predict_model(
    models[winner_family], x_test
)
winner_test_rank = within_user_rank(
    test.user_id, winner_test_logits
)

if uses_incumbent:
    if not os.path.exists(incumbent_test_path):
        raise FileNotFoundError("Trusted incumbent test scores missing")
    incumbent_test = np.asarray(
        np.load(incumbent_test_path), dtype=np.float64
    )
    if len(incumbent_test) != len(test.user_id):
        raise ValueError("Incumbent test length mismatch")
    incumbent_test_rank = within_user_rank(
        test.user_id, incumbent_test
    )
    test_scores = (
        winner_weight * winner_test_rank
        + (1.0 - winner_weight) * incumbent_test_rank
    )
else:
    test_scores = winner_test_rank

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
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)