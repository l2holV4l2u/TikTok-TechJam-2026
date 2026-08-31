import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 73591
THREADS = min(16, os.cpu_count() or 1)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

DEVICE = torch.device("cpu")
BATCH_SIZE = 6144
EPOCHS = 3
EMBED_DIM = 14
HALF_LIFE = 4.0
HARD_NEGATIVES = 4
RANDOM_NEGATIVES = 8

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "hour",
    "upload_type",
    "music_type",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "is_video_author",
    "is_live_streamer",
    "onehot_feat3",
    "onehot_feat8",
]

CANDIDATE_FIELDS = [
    "video_id",
    "author_id",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "is_video_author",
    "is_live_streamer",
    "onehot_feat3",
    "onehot_feat8",
]

USER_CONTEXT_FIELDS = [
    "user_id",
    "tab",
    "hour",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_range",
]


def categorical_matrix(split):
    return np.ascontiguousarray(
        np.stack([
            np.asarray(split.X[name], dtype=np.int32)
            for name in FIELDS
        ], axis=1),
        dtype=np.int32,
    )


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    ordered_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = ordered_users[1:] != ordered_users[:-1]
    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = ordered_users[:-1] != ordered_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((
        np.asarray([-1], dtype=np.int64),
        end_positions,
    )))
    row_sizes = np.repeat(sizes, sizes)

    within_position = (
        np.arange(n, dtype=np.int64) - start_positions
    )
    ranked = (
        within_position.astype(np.float64) + 0.5
    ) / row_sizes.astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def smoothed_entity_rate(ids, labels, cardinality, prior_strength):
    ids = np.asarray(ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    counts = np.bincount(ids, minlength=cardinality).astype(np.float64)
    positives = np.bincount(
        ids, weights=labels, minlength=cardinality
    ).astype(np.float64)
    prior = float(labels.mean())
    return (
        positives + prior_strength * prior
    ) / (
        counts + prior_strength
    )


def propensity_score_for_mining(train, labels):
    components = []
    specifications = [
        ("video_id", 18.0),
        ("author_id", 24.0),
        ("tag", 80.0),
        ("duration_bucket", 100.0),
        ("upload_type", 100.0),
        ("music_type", 100.0),
        ("onehot_feat3", 45.0),
        ("onehot_feat8", 45.0),
    ]

    for field, strength in specifications:
        ids = np.asarray(train.X[field], dtype=np.int64)
        rates = smoothed_entity_rate(
            ids,
            labels,
            int(FEATURE_CARDINALITIES[field]),
            strength,
        )
        components.append(rates[ids])

    score = np.mean(np.stack(components, axis=1), axis=1)
    return np.asarray(score, dtype=np.float32)


def top_negative_table(user_ids, labels, ordering_score, width):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    ordering_score = np.asarray(ordering_score, dtype=np.float64)

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    table = np.full((n_users, width), -1, dtype=np.int64)

    negatives = np.flatnonzero(labels == 0)
    order = np.lexsort((
        negatives,
        -ordering_score[negatives],
        user_ids[negatives],
    ))
    ordered_rows = negatives[order]
    ordered_users = user_ids[ordered_rows]

    starts = np.empty(len(ordered_rows), dtype=bool)
    if len(ordered_rows) == 0:
        return table
    starts[0] = True
    starts[1:] = ordered_users[1:] != ordered_users[:-1]
    start_positions = np.maximum.accumulate(
        np.where(
            starts,
            np.arange(len(ordered_rows), dtype=np.int64),
            0,
        )
    )
    ranks = (
        np.arange(len(ordered_rows), dtype=np.int64)
        - start_positions
    )

    keep = ranks < width
    table[
        ordered_users[keep],
        ranks[keep],
    ] = ordered_rows[keep]
    return table


def construct_pairs(train, labels):
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    positives = np.flatnonzero(labels == 1).astype(np.int64)

    propensity = propensity_score_for_mining(train, labels)
    hard_table = top_negative_table(
        users, labels, propensity, HARD_NEGATIVES
    )

    random_key = np.random.RandomState(SEED + 17).uniform(
        size=len(labels)
    )
    random_table = top_negative_table(
        users, labels, random_key, RANDOM_NEGATIVES
    )

    positive_users = users[positives]
    serial = np.arange(len(positives), dtype=np.int64)

    hard_rank = (
        serial * np.int64(1103515245)
        + positives * np.int64(12345)
    ) % HARD_NEGATIVES
    random_rank = (
        serial * np.int64(2654435761)
        + positives * np.int64(97)
    ) % RANDOM_NEGATIVES

    hard_negative = hard_table[positive_users, hard_rank]
    random_negative = random_table[positive_users, random_rank]

    hard_fallback = hard_table[positive_users, 0]
    random_fallback = random_table[positive_users, 0]

    hard_negative = np.where(
        hard_negative >= 0, hard_negative, random_fallback
    )
    random_negative = np.where(
        random_negative >= 0, random_negative, hard_fallback
    )

    valid_hard = hard_negative >= 0
    valid_random = random_negative >= 0

    pair_positive = np.concatenate([
        positives[valid_hard],
        positives[valid_random],
    ])
    pair_negative = np.concatenate([
        hard_negative[valid_hard],
        random_negative[valid_random],
    ])

    pair_kind = np.concatenate([
        np.ones(np.sum(valid_hard), dtype=np.float32),
        np.zeros(np.sum(valid_random), dtype=np.float32),
    ])

    dates = np.asarray(train.date, dtype=np.int32)
    day_age = dates.max() - dates[pair_positive]
    recency = np.power(
        0.5,
        day_age.astype(np.float32) / HALF_LIFE,
    )
    recency /= max(float(recency.mean()), 1e-8)

    # Hard negatives get moderately greater emphasis while random
    # negatives preserve broad calibration and avoid collapsing onto a
    # handful of popular difficult impressions.
    pair_weights = recency * (1.0 + 0.35 * pair_kind)
    pair_weights /= max(float(pair_weights.mean()), 1e-8)

    diagnostics = {
        "positive_rows": int(len(positives)),
        "pairs": int(len(pair_positive)),
        "hard_pairs": int(np.sum(valid_hard)),
        "random_pairs": int(np.sum(valid_random)),
        "users_with_hard_negative": int(np.sum(hard_table[:, 0] >= 0)),
        "mining_positive_mean": float(propensity[labels == 1].mean()),
        "mining_negative_mean": float(propensity[labels == 0].mean()),
    }
    print("FINDINGS " + json.dumps(diagnostics, sort_keys=True))

    return (
        np.asarray(pair_positive, dtype=np.int64),
        np.asarray(pair_negative, dtype=np.int64),
        np.asarray(pair_weights, dtype=np.float32),
    )


class EmbeddingScorer(nn.Module):
    def __init__(self, embedding_dim=EMBED_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                int(FEATURE_CARDINALITIES[field]),
                embedding_dim,
            )
            for field in FIELDS
        ])
        self.linear = nn.ModuleList([
            nn.Embedding(
                int(FEATURE_CARDINALITIES[field]),
                1,
            )
            for field in FIELDS
        ])
        self.bias = nn.Parameter(torch.zeros(()))

        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.025)
            with torch.no_grad():
                embedding.weight[0].zero_()
        for linear in self.linear:
            nn.init.zeros_(linear.weight)

    def embedded(self, x):
        return torch.stack([
            embedding(x[:, index])
            for index, embedding in enumerate(self.embeddings)
        ], dim=1)

    def wide(self, x):
        terms = [
            linear(x[:, index]).squeeze(1)
            for index, linear in enumerate(self.linear)
        ]
        return torch.stack(terms, dim=1).sum(dim=1) + self.bias


class PairwiseFM(EmbeddingScorer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        embedded = self.embedded(x)
        summed = embedded.sum(dim=1)
        fm = 0.5 * (
            summed.square().sum(dim=1)
            - embedded.square().sum(dim=(1, 2))
        )
        return self.wide(x) + fm


class PairwiseTwoTower(EmbeddingScorer):
    def __init__(self):
        super().__init__()
        user_indices = [FIELDS.index(field) for field in USER_CONTEXT_FIELDS]
        candidate_indices = [
            FIELDS.index(field) for field in CANDIDATE_FIELDS
        ]
        self.user_indices = user_indices
        self.candidate_indices = candidate_indices

        user_input = len(user_indices) * EMBED_DIM
        candidate_input = len(candidate_indices) * EMBED_DIM

        self.user_tower = nn.Sequential(
            nn.Linear(user_input, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 48),
        )
        self.candidate_tower = nn.Sequential(
            nn.Linear(candidate_input, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 48),
        )
        self.temperature = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        embedded = self.embedded(x)
        user_features = embedded[
            :, self.user_indices, :
        ].flatten(start_dim=1)
        candidate_features = embedded[
            :, self.candidate_indices, :
        ].flatten(start_dim=1)

        user_vector = F.normalize(
            self.user_tower(user_features), dim=1
        )
        candidate_vector = F.normalize(
            self.candidate_tower(candidate_features), dim=1
        )
        metric_score = (
            user_vector * candidate_vector
        ).sum(dim=1) * F.softplus(self.temperature)

        return self.wide(x) + metric_score


class CrossLayer(nn.Module):
    def __init__(self, dimension, rank=48):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=True)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x0, x):
        interaction = self.up(F.silu(self.down(x)))
        return x + x0 * interaction


class PairwiseDCNv2(EmbeddingScorer):
    def __init__(self):
        super().__init__()
        dimension = len(FIELDS) * EMBED_DIM
        self.cross_layers = nn.ModuleList([
            CrossLayer(dimension, 48),
            CrossLayer(dimension, 48),
            CrossLayer(dimension, 32),
        ])
        self.deep = nn.Sequential(
            nn.Linear(dimension, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(192, 64),
            nn.SiLU(),
        )
        self.output = nn.Linear(dimension + 64, 1)

    def forward(self, x):
        x0 = self.embedded(x).flatten(start_dim=1)
        crossed = x0
        for layer in self.cross_layers:
            crossed = layer(x0, crossed)
        deep = self.deep(x0)
        interaction_score = self.output(
            torch.cat([crossed, deep], dim=1)
        ).squeeze(1)
        return self.wide(x) + interaction_score


def train_pairwise(
    model,
    x_train,
    labels,
    positive_rows,
    negative_rows,
    pair_weights,
    seed,
    name,
):
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0022,
        weight_decay=2e-5,
    )

    n_pairs = len(positive_rows)
    labels_tensor = torch.from_numpy(labels.astype(np.float32))

    for epoch in range(EPOCHS):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(seed + 1009 * epoch)
        permutation = torch.randperm(n_pairs, generator=generator)

        total_pair_loss = 0.0
        total_margin = 0.0
        total_rows = 0

        for start in range(0, n_pairs, BATCH_SIZE):
            pair_ids = permutation[
                start:start + BATCH_SIZE
            ].numpy()

            pos_rows = positive_rows[pair_ids]
            neg_rows = negative_rows[pair_ids]
            weights = torch.from_numpy(
                pair_weights[pair_ids]
            ).to(DEVICE)

            pos_x = torch.from_numpy(
                x_train[pos_rows].astype(np.int64, copy=False)
            ).to(DEVICE)
            neg_x = torch.from_numpy(
                x_train[neg_rows].astype(np.int64, copy=False)
            ).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pos_score = model(pos_x)
            neg_score = model(neg_x)
            margin = pos_score - neg_score

            pair_losses = F.softplus(-margin)
            pair_loss = torch.sum(
                pair_losses * weights
            ) / torch.sum(weights)

            # A weak pointwise auxiliary term anchors score magnitudes and
            # field biases without overwhelming the ranking objective.
            point_logits = torch.cat([pos_score, neg_score], dim=0)
            point_targets = torch.cat([
                torch.ones_like(pos_score),
                torch.zeros_like(neg_score),
            ])
            point_loss = F.binary_cross_entropy_with_logits(
                point_logits, point_targets
            )
            loss = pair_loss + 0.12 * point_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()

            batch_rows = len(pair_ids)
            total_pair_loss += float(pair_loss.detach()) * batch_rows
            total_margin += float(margin.detach().mean()) * batch_rows
            total_rows += batch_rows

        print("FINDINGS " + json.dumps({
            "model": name,
            "epoch": epoch + 1,
            "pair_loss": total_pair_loss / max(total_rows, 1),
            "mean_margin": total_margin / max(total_rows, 1),
        }, sort_keys=True))

    return model


@torch.no_grad()
def predict(model, x):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    prediction_batch = BATCH_SIZE * 2

    for start in range(0, len(x), prediction_batch):
        end = min(start + prediction_batch, len(x))
        batch = torch.from_numpy(
            x[start:end].astype(np.int64, copy=False)
        ).to(DEVICE)
        result[start:end] = (
            model(batch).cpu().numpy().astype(np.float32)
        )
    return result


train = load("train")
valid = load("valid")
test = load("test")

x_train = categorical_matrix(train)
x_valid = categorical_matrix(valid)
x_test = categorical_matrix(test)
labels_train = np.asarray(train.y, dtype=np.int8)

positive_rows, negative_rows, pair_weights = construct_pairs(
    train, labels_train
)

model_specs = [
    ("hard_bpr_fm", PairwiseFM, SEED + 11),
    ("hard_bpr_two_tower", PairwiseTwoTower, SEED + 29),
    ("hard_bpr_dcnv2", PairwiseDCNv2, SEED + 47),
]

own_valid = {}
own_test = {}

for name, constructor, seed in model_specs:
    model = constructor()
    model = train_pairwise(
        model,
        x_train,
        labels_train,
        positive_rows,
        negative_rows,
        pair_weights,
        seed,
        name,
    )
    own_valid[name] = predict(model, x_valid)
    own_test[name] = predict(model, x_test)
    del model
    gc.collect()

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

candidate_scores = {
    "trusted_incumbent": inc_valid,
}
candidate_metrics = {
    "trusted_incumbent": evaluate(
        valid.user_id, valid.y, inc_valid
    ),
}
candidate_specs = {
    "trusted_incumbent": ("incumbent", None),
}

valid_ranks = {}
test_ranks = {}

for name in own_valid:
    valid_raw = np.asarray(own_valid[name], dtype=np.float64)
    test_raw = np.asarray(own_test[name], dtype=np.float64)

    valid_ranks[name] = rank_percentile(
        valid.user_id, valid_raw
    )
    test_ranks[name] = rank_percentile(
        test.user_id, test_raw
    )

    candidate_scores[name] = valid_raw
    candidate_metrics[name] = evaluate(
        valid.user_id, valid.y, valid_raw
    )
    candidate_specs[name] = (name, None)

    for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        candidate_name = f"{name}_incumbent_{alpha:.2f}"
        blended = (
            alpha * valid_ranks[name]
            + (1.0 - alpha) * inc_valid_rank
        )
        candidate_scores[candidate_name] = blended
        candidate_metrics[candidate_name] = evaluate(
            valid.user_id, valid.y, blended
        )
        candidate_specs[candidate_name] = (name, alpha)

ensemble_valid = np.mean(
    np.stack([
        valid_ranks[name] for name in own_valid
    ], axis=1),
    axis=1,
)
ensemble_test = np.mean(
    np.stack([
        test_ranks[name] for name in own_test
    ], axis=1),
    axis=1,
)

candidate_scores["hard_pairwise_ensemble"] = ensemble_valid
candidate_metrics["hard_pairwise_ensemble"] = evaluate(
    valid.user_id, valid.y, ensemble_valid
)
candidate_specs["hard_pairwise_ensemble"] = ("ensemble", None)

for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
    candidate_name = f"hard_pairwise_ensemble_incumbent_{alpha:.2f}"
    blended = (
        alpha * ensemble_valid
        + (1.0 - alpha) * inc_valid_rank
    )
    candidate_scores[candidate_name] = blended
    candidate_metrics[candidate_name] = evaluate(
        valid.user_id, valid.y, blended
    )
    candidate_specs[candidate_name] = ("ensemble", alpha)

names = list(own_valid)
for left_index in range(len(names)):
    for right_index in range(left_index + 1, len(names)):
        left = names[left_index]
        right = names[right_index]
        pair_name = f"{left}_{right}_ensemble"
        pair_valid = 0.5 * (
            valid_ranks[left] + valid_ranks[right]
        )
        pair_test = 0.5 * (
            test_ranks[left] + test_ranks[right]
        )

        candidate_scores[pair_name] = pair_valid
        candidate_metrics[pair_name] = evaluate(
            valid.user_id, valid.y, pair_valid
        )
        candidate_specs[pair_name] = (
            f"pair:{left}:{right}", None
        )

        for alpha in (0.10, 0.20, 0.30):
            blended_name = f"{pair_name}_incumbent_{alpha:.2f}"
            blended = (
                alpha * pair_valid
                + (1.0 - alpha) * inc_valid_rank
            )
            candidate_scores[blended_name] = blended
            candidate_metrics[blended_name] = evaluate(
                valid.user_id, valid.y, blended
            )
            candidate_specs[blended_name] = (
                f"pair:{left}:{right}", alpha
            )

best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = np.asarray(
    candidate_scores[best_name], dtype=np.float64
)
best_family, best_alpha = candidate_specs[best_name]

standalone_best = max(
    own_valid,
    key=lambda name: float(
        candidate_metrics[name]["primary"]
    ),
)
raw_valid_for_audit = np.asarray(
    own_valid[standalone_best], dtype=np.float64
)

if best_family == "incumbent":
    best_test = inc_test
elif best_family == "ensemble":
    if best_alpha is None:
        best_test = ensemble_test
    else:
        best_test = (
            best_alpha * ensemble_test
            + (1.0 - best_alpha) * inc_test_rank
        )
elif best_family.startswith("pair:"):
    _, left, right = best_family.split(":")
    pair_test = 0.5 * (
        test_ranks[left] + test_ranks[right]
    )
    if best_alpha is None:
        best_test = pair_test
    else:
        best_test = (
            best_alpha * pair_test
            + (1.0 - best_alpha) * inc_test_rank
        )
else:
    if best_alpha is None:
        best_test = np.asarray(
            own_test[best_family], dtype=np.float64
        )
    else:
        best_test = (
            best_alpha * test_ranks[best_family]
            + (1.0 - best_alpha) * inc_test_rank
        )

correlations = {}
for left_index in range(len(names)):
    for right_index in range(left_index + 1, len(names)):
        left = names[left_index]
        right = names[right_index]
        correlations[f"{left}__{right}"] = float(
            np.corrcoef(
                valid_ranks[left], valid_ranks[right]
            )[0, 1]
        )

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "best_family": best_family,
    "best_blend_alpha": best_alpha,
    "best_standalone": standalone_best,
    "best_standalone_primary": float(
        candidate_metrics[standalone_best]["primary"]
    ),
    "within_user_rank_correlations": correlations,
    "epochs": EPOCHS,
    "embedding_dim": EMBED_DIM,
    "half_life_days": HALF_LIFE,
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
    if best_family == "incumbent" or best_alpha is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            raw_valid_for_audit,
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))