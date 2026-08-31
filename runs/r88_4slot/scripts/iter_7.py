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
SEED = 19417
DEVICE = torch.device("cpu")
BATCH_SIZE = 8192
CHECKPOINTS = (2, 4)
MAX_EPOCHS = max(CHECKPOINTS)
N_HARD_NEGATIVES = 3

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
]
CARDS = [int(FEATURE_CARDINALITIES[x]) for x in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARD = int(sum(CARDS))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))


def encode(split):
    return np.stack(
        [
            np.asarray(split.X[name], dtype=np.int64) + OFFSETS[j]
            for j, name in enumerate(FIELDS)
        ],
        axis=1,
    )


def date_recency_weights(dates, half_life):
    dates = np.asarray(dates)
    unique_dates = np.unique(dates)
    positions = np.searchsorted(unique_dates, dates).astype(np.float32)
    age = float(len(unique_dates) - 1) - positions
    weights = np.exp2(-age / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-6)
    return weights


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    # A tiny deterministic jitter prevents large empirical-Bayes tie blocks.
    jitter = (np.arange(n, dtype=np.float64) % 104729) * 1e-13
    order = np.lexsort((np.arange(n, dtype=np.int64), scores + jitter, user_ids))
    sorted_users = user_ids[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    repeated_starts = np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)
    position = np.arange(n, dtype=np.float64) - repeated_starts

    ranked_sorted = np.full(n, 0.5, dtype=np.float64)
    mask = repeated_counts > 1
    ranked_sorted[mask] = position[mask] / (repeated_counts[mask] - 1.0)

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


class PairSampler:
    def __init__(self, x, y, dates, half_life):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.int64))
        y = np.asarray(y, dtype=np.int8)
        users = np.asarray(x[:, 0] - OFFSETS[0], dtype=np.int64)
        user_card = CARDS[0]

        neg_rows = np.flatnonzero(y == 0).astype(np.int64)
        neg_order = np.argsort(users[neg_rows], kind="stable")
        self.neg_rows = neg_rows[neg_order]
        neg_users = users[self.neg_rows]

        neg_counts = np.bincount(neg_users, minlength=user_card).astype(np.int64)
        neg_starts = np.zeros(user_card, dtype=np.int64)
        if user_card > 1:
            neg_starts[1:] = np.cumsum(neg_counts[:-1])

        positive_rows = np.flatnonzero((y == 1) & (neg_counts[users] > 0)).astype(
            np.int64
        )

        self.positive_rows = positive_rows
        self.positive_users = users[positive_rows]
        self.neg_counts = neg_counts
        self.neg_starts = neg_starts
        self.pair_weights = date_recency_weights(dates, half_life)[positive_rows]

    def shuffled_positive_rows(self, generator):
        permutation = torch.randperm(len(self.positive_rows), generator=generator)
        return permutation.numpy()

    def sample_candidates(self, local_indices, rng, n_candidates):
        users = self.positive_users[local_indices]
        starts = self.neg_starts[users]
        counts = self.neg_counts[users]

        random_fraction = rng.random(
            (len(local_indices), n_candidates), dtype=np.float64
        )
        offsets = np.floor(random_fraction * counts[:, None]).astype(np.int64)
        return self.neg_rows[starts[:, None] + offsets]


class BPRFieldModel(nn.Module):
    def __init__(self, dim=20):
        super().__init__()
        self.user_embedding = nn.Embedding(CARDS[0], dim)
        self.video_embedding = nn.Embedding(CARDS[1], dim)
        self.author_embedding = nn.Embedding(CARDS[2], dim)
        self.tag_embedding = nn.Embedding(CARDS[3], dim)

        self.video_bias = nn.Embedding(CARDS[1], 1)
        self.author_bias = nn.Embedding(CARDS[2], 1)
        self.tag_bias = nn.Embedding(CARDS[3], 1)
        self.tab_bias = nn.Embedding(CARDS[4], 1)
        self.duration_bias = nn.Embedding(CARDS[5], 1)

        for module in [
            self.user_embedding,
            self.video_embedding,
            self.author_embedding,
            self.tag_embedding,
        ]:
            nn.init.normal_(module.weight, std=0.04)

        for module in [
            self.video_bias,
            self.author_bias,
            self.tag_bias,
            self.tab_bias,
            self.duration_bias,
        ]:
            nn.init.zeros_(module.weight)

    def forward(self, x):
        user = x[:, 0] - OFFSETS[0]
        video = x[:, 1] - OFFSETS[1]
        author = x[:, 2] - OFFSETS[2]
        tag = x[:, 3] - OFFSETS[3]
        tab = x[:, 4] - OFFSETS[4]
        duration = x[:, 5] - OFFSETS[5]

        user_vector = self.user_embedding(user)
        content_vector = (
            self.video_embedding(video)
            + 0.65 * self.author_embedding(author)
            + 0.35 * self.tag_embedding(tag)
        )
        interaction = torch.sum(user_vector * content_vector, dim=1)

        bias = (
            self.video_bias(video).squeeze(1)
            + 0.7 * self.author_bias(author).squeeze(1)
            + 0.4 * self.tag_bias(tag).squeeze(1)
            + self.tab_bias(tab).squeeze(1)
            + self.duration_bias(duration).squeeze(1)
        )
        return interaction + bias


class PairwiseInteractionMLP(nn.Module):
    def __init__(self, dim=10):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, dim)
        nn.init.normal_(self.embedding.weight, std=0.035)

        flat = len(FIELDS) * dim
        self.network = nn.Sequential(
            nn.Linear(flat, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        embedded = self.embedding(x).reshape(x.shape[0], -1)
        deep_score = self.network(embedded).squeeze(1)
        wide_score = self.linear(x).sum(dim=1).squeeze(1)
        return deep_score + wide_score


def predict_neural(model, x):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    x_tensor = torch.from_numpy(np.asarray(x, dtype=np.int64))
    with torch.inference_mode():
        for lo in range(0, len(x), BATCH_SIZE * 2):
            hi = min(lo + BATCH_SIZE * 2, len(x))
            result[lo:hi] = model(x_tensor[lo:hi]).cpu().numpy()
    return result


def train_pairwise(
    model_class,
    x_train,
    y_train,
    dates_train,
    x_valid=None,
    y_valid=None,
    valid_users=None,
    half_life=5.0,
    fixed_epochs=None,
):
    torch.manual_seed(SEED)
    model = model_class().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=2e-6
    )
    sampler = PairSampler(x_train, y_train, dates_train, half_life)

    order_generator = torch.Generator()
    order_generator.manual_seed(SEED + 11)
    negative_rng = np.random.default_rng(SEED + 29)

    if fixed_epochs is None:
        epochs = MAX_EPOCHS
    else:
        epochs = int(fixed_epochs)

    best_epoch = epochs
    best_scores = None
    best_metrics = None
    best_primary = -np.inf

    for epoch in range(1, epochs + 1):
        model.train()
        local_order = sampler.shuffled_positive_rows(order_generator)

        for lo in range(0, len(local_order), BATCH_SIZE):
            local_idx = local_order[lo:min(lo + BATCH_SIZE, len(local_order))]
            positive_rows = sampler.positive_rows[local_idx]
            candidate_rows = sampler.sample_candidates(
                local_idx, negative_rng, N_HARD_NEGATIVES
            )

            positive_x = sampler.x[positive_rows]
            candidate_x = sampler.x[candidate_rows.reshape(-1)]

            # Select the currently highest-scoring sampled negative.
            with torch.no_grad():
                candidate_scores = model(candidate_x).reshape(
                    len(local_idx), N_HARD_NEGATIVES
                )
                hard_choice = torch.argmax(candidate_scores, dim=1)

            row_numbers = torch.arange(len(local_idx))
            candidate_tensor = torch.from_numpy(candidate_rows)
            hard_rows = candidate_tensor[row_numbers, hard_choice]
            negative_x = sampler.x[hard_rows]

            positive_score = model(positive_x)
            negative_score = model(negative_x)
            margin_loss = nn.functional.softplus(
                -(positive_score - negative_score)
            )

            weights = torch.from_numpy(
                sampler.pair_weights[local_idx].astype(np.float32, copy=False)
            )
            loss = (margin_loss * weights).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if (
            fixed_epochs is None
            and epoch in CHECKPOINTS
            and x_valid is not None
        ):
            scores = predict_neural(model, x_valid)
            metrics = evaluate(valid_users, y_valid, scores)
            primary = float(metrics["primary"])
            if primary > best_primary:
                best_primary = primary
                best_epoch = epoch
                best_scores = scores.copy()
                best_metrics = metrics

    if fixed_epochs is not None:
        return model

    return best_epoch, best_scores, best_metrics


def weighted_category_rate(ids, y, weights, cardinality, strength):
    ids = np.asarray(ids, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    global_rate = float(np.sum(weights * y) / np.sum(weights))

    total = np.bincount(ids, weights=weights, minlength=cardinality)
    positive = np.bincount(ids, weights=weights * y, minlength=cardinality)
    rate = (positive + strength * global_rate) / (total + strength)
    return rate.astype(np.float32), global_rate


class EmpiricalBayesModel:
    def __init__(self, half_life=5.0):
        self.half_life = half_life

    def fit(self, split_x, y, dates):
        y = np.asarray(y, dtype=np.float64)
        weights = date_recency_weights(dates, self.half_life).astype(np.float64)
        self.global_rate = float(np.sum(weights * y) / np.sum(weights))

        self.rates = {}
        strengths = {
            "video_id": 25.0,
            "author_id": 35.0,
            "tag": 45.0,
            "tab": 70.0,
            "duration_bucket": 80.0,
        }
        for name, strength in strengths.items():
            rate, _ = weighted_category_rate(
                np.asarray(split_x[name], dtype=np.int64),
                y,
                weights,
                int(FEATURE_CARDINALITIES[name]),
                strength,
            )
            self.rates[name] = rate

        users = np.asarray(split_x["user_id"], dtype=np.int64)
        authors = np.asarray(split_x["author_id"], dtype=np.int64)
        author_card = int(FEATURE_CARDINALITIES["author_id"])
        keys = users * author_card + authors

        unique_keys, inverse = np.unique(keys, return_inverse=True)
        totals = np.bincount(inverse, weights=weights)
        positives = np.bincount(inverse, weights=weights * y)
        parent = self.rates["author_id"][
            (unique_keys % author_card).astype(np.int64)
        ].astype(np.float64)

        pair_strength = 8.0
        pair_rates = (
            positives + pair_strength * parent
        ) / (totals + pair_strength)

        self.ua_keys = unique_keys.astype(np.int64)
        self.ua_rates = pair_rates.astype(np.float32)
        return self

    @staticmethod
    def logit(values):
        values = np.clip(values, 1e-4, 1.0 - 1e-4)
        return np.log(values / (1.0 - values))

    def predict(self, split_x, variant="full"):
        video_rate = self.rates["video_id"][
            np.asarray(split_x["video_id"], dtype=np.int64)
        ]
        author_rate = self.rates["author_id"][
            np.asarray(split_x["author_id"], dtype=np.int64)
        ]
        tag_rate = self.rates["tag"][
            np.asarray(split_x["tag"], dtype=np.int64)
        ]
        tab_rate = self.rates["tab"][
            np.asarray(split_x["tab"], dtype=np.int64)
        ]
        duration_rate = self.rates["duration_bucket"][
            np.asarray(split_x["duration_bucket"], dtype=np.int64)
        ]

        base_logit = self.logit(self.global_rate)
        score = (
            1.00 * (self.logit(video_rate) - base_logit)
            + 0.70 * (self.logit(author_rate) - base_logit)
            + 0.35 * (self.logit(tag_rate) - base_logit)
            + 0.45 * (self.logit(tab_rate) - base_logit)
            + 0.25 * (self.logit(duration_rate) - base_logit)
        )

        if variant == "full":
            users = np.asarray(split_x["user_id"], dtype=np.int64)
            authors = np.asarray(split_x["author_id"], dtype=np.int64)
            author_card = int(FEATURE_CARDINALITIES["author_id"])
            keys = users * author_card + authors
            positions = np.searchsorted(self.ua_keys, keys)
            valid = positions < len(self.ua_keys)
            matched = np.zeros(len(keys), dtype=bool)
            matched[valid] = self.ua_keys[positions[valid]] == keys[valid]

            pair_rate = author_rate.copy()
            pair_rate[matched] = self.ua_rates[positions[matched]]
            score += 1.15 * (
                self.logit(pair_rate) - self.logit(author_rate)
            )

        return np.asarray(score, dtype=np.float32)


train = load("train")
valid = load("valid")

x_train = encode(train)
x_valid = encode(valid)
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)
dates_train = np.asarray(train.date)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path)
inc_valid_rank = within_user_rank(valid_users, inc_valid)

candidate_scores = {}
candidate_epochs = {}
candidate_recipes = {}

# Family 1: low-rank field-aware BPR.
bpr_epoch, bpr_scores, _ = train_pairwise(
    BPRFieldModel,
    x_train,
    y_train,
    dates_train,
    x_valid=x_valid,
    y_valid=y_valid,
    valid_users=valid_users,
    half_life=4.0,
)
candidate_scores["hard_bpr"] = bpr_scores
candidate_epochs["hard_bpr"] = bpr_epoch
candidate_recipes["hard_bpr"] = ("neural", BPRFieldModel, 4.0)

# Family 2: nonlinear pairwise neural interaction ranker.
mlp_epoch, mlp_scores, _ = train_pairwise(
    PairwiseInteractionMLP,
    x_train,
    y_train,
    dates_train,
    x_valid=x_valid,
    y_valid=y_valid,
    valid_users=valid_users,
    half_life=7.0,
)
candidate_scores["pairwise_interaction_mlp"] = mlp_scores
candidate_epochs["pairwise_interaction_mlp"] = mlp_epoch
candidate_recipes["pairwise_interaction_mlp"] = (
    "neural",
    PairwiseInteractionMLP,
    7.0,
)

# Family 3: non-parametric hierarchical empirical Bayes.
eb_models = {}
for half_life in (3.0, 7.0):
    model = EmpiricalBayesModel(half_life=half_life).fit(
        train.X, y_train, train.date
    )
    eb_models[half_life] = model
    for variant in ("marginal", "full"):
        name = "empirical_bayes_h{}_{}".format(int(half_life), variant)
        candidate_scores[name] = model.predict(valid.X, variant=variant)
        candidate_epochs[name] = 0
        candidate_recipes[name] = ("eb", half_life, variant)

recorded = {}
best_primary = -np.inf
best_name = None
best_alpha = None
best_scores = None
best_raw_scores = None
best_metrics = None

alphas = np.linspace(0.0, 1.0, 21)

for name, raw_scores in candidate_scores.items():
    raw_metrics = evaluate(valid_users, y_valid, raw_scores)
    recorded[name + "_standalone"] = float(raw_metrics["primary"])

    raw_rank = within_user_rank(valid_users, raw_scores)
    local_best = -np.inf
    local_alpha = None

    for alpha in alphas:
        blended = (1.0 - alpha) * inc_valid_rank + alpha * raw_rank
        metrics = evaluate(valid_users, y_valid, blended)
        primary = float(metrics["primary"])

        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_alpha = float(alpha)
            best_scores = blended.copy()
            best_raw_scores = raw_scores.copy()
            best_metrics = metrics

    recorded[name + "_best_blend"] = float(local_best)
    recorded[name + "_blend_alpha"] = float(local_alpha)
    if candidate_epochs[name]:
        recorded[name + "_epoch"] = int(candidate_epochs[name])

print("CANDIDATES " + json.dumps(recorded, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(best_raw_scores, dtype=np.float64),
    )

# Refit the exact selected recipe on train + validation.
test = load("test")
inc_test = np.load(inc_test_path)
test_users = np.asarray(test.user_id)

recipe = candidate_recipes[best_name]
if recipe[0] == "neural":
    _, model_class, half_life = recipe
    x_fit = np.concatenate([x_train, x_valid], axis=0)
    y_fit = np.concatenate([y_train, y_valid], axis=0)
    dates_fit = np.concatenate(
        [np.asarray(train.date), np.asarray(valid.date)], axis=0
    )
    selected_model = train_pairwise(
        model_class,
        x_fit,
        y_fit,
        dates_fit,
        half_life=half_life,
        fixed_epochs=candidate_epochs[best_name],
    )
    raw_test_scores = predict_neural(selected_model, encode(test))
else:
    _, half_life, variant = recipe
    fit_x = {}
    for field in train.X.keys():
        fit_x[field] = np.concatenate(
            [
                np.asarray(train.X[field]),
                np.asarray(valid.X[field]),
            ],
            axis=0,
        )
    y_fit = np.concatenate([y_train, y_valid], axis=0)
    dates_fit = np.concatenate(
        [np.asarray(train.date), np.asarray(valid.date)], axis=0
    )
    selected_model = EmpiricalBayesModel(half_life=half_life).fit(
        fit_x, y_fit, dates_fit
    )
    raw_test_scores = selected_model.predict(test.X, variant=variant)

inc_test_rank = within_user_rank(test_users, inc_test)
raw_test_rank = within_user_rank(test_users, raw_test_scores)
test_scores = (
    (1.0 - best_alpha) * inc_test_rank
    + best_alpha * raw_test_rank
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_family": best_name,
            "selected_blend_weight": float(best_alpha),
            "selected_epoch": int(candidate_epochs[best_name]),
            "pairwise_positive_pairs": int(
                np.sum(
                    (y_train == 1)
                    & (
                        np.bincount(
                            np.asarray(train.X["user_id"])[y_train == 0],
                            minlength=CARDS[0],
                        )[np.asarray(train.X["user_id"])]
                        > 0
                    )
                )
            ),
        },
        sort_keys=True,
    )
)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(", ", ": "),
    )
)