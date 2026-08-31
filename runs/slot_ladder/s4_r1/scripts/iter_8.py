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
SEED = 314159
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 16384

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
n_fields = len(FIELDS)


def make_matrix(split):
    columns = []
    for field, offset, card in zip(FIELDS, offsets, cards):
        values = np.asarray(split.X[field], dtype=np.int64)
        if values.size and (values.min() < 0 or values.max() >= card):
            raise ValueError("Out-of-range IDs in " + field)
        columns.append(values + offset)
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.int64)


x_train = make_matrix(train)
x_valid = make_matrix(valid)
x_test = make_matrix(test)
y_train = np.asarray(train.y, dtype=np.float32)

# The evaluation windows follow immediately after training, while the label
# rate and activity distribution drift substantially. A four-day half-life
# emphasizes behavior closest to the deployment boundary.
train_dates = np.asarray(train.date, dtype=np.int64)
sample_weight = np.exp2(
    (train_dates - train_dates.max()).astype(np.float32) / 4.0
)
sample_weight /= sample_weight.mean()
sample_weight = sample_weight.astype(np.float32)

aux_targets = np.stack(
    [
        y_train,
        np.asarray(train.aux["is_click"], dtype=np.float32),
        np.asarray(train.aux["is_like"], dtype=np.float32),
    ],
    axis=1,
)

rng = np.random.default_rng(SEED)


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


def user_slate_sizes(user_ids):
    _, inverse, counts = np.unique(
        np.asarray(user_ids), return_inverse=True, return_counts=True
    )
    return counts[inverse].astype(np.float64)


@torch.inference_mode()
def predict(model, x, output_index=None):
    model.eval()
    result = np.empty(len(x), dtype=np.float64)
    for lo in range(0, len(x), PRED_BATCH_SIZE):
        hi = min(lo + PRED_BATCH_SIZE, len(x))
        xb = torch.from_numpy(x[lo:hi])
        logits = model(xb)
        if output_index is not None:
            logits = logits[:, output_index]
        result[lo:hi] = logits.detach().cpu().numpy().astype(np.float64)
    return result


def train_binary(model, epochs, learning_rate):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-6
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


def train_ple(model, epochs=3, learning_rate=0.0015):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-6
    )
    task_weights = torch.tensor([1.0, 0.25, 0.12], dtype=torch.float32)
    n = len(y_train)

    for _ in range(epochs):
        model.train()
        order = rng.permutation(n)
        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:lo + BATCH_SIZE]
            xb = torch.from_numpy(x_train[idx])
            target = torch.from_numpy(aux_targets[idx])
            weights = torch.from_numpy(sample_weight[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            task_losses = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            row_losses = (task_losses * task_weights).sum(dim=1)
            loss = (row_losses * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


class FieldAwareFM(nn.Module):
    """Each feature has a distinct embedding for every interacting field."""

    def __init__(self, n_features, fields, rank=10):
        super().__init__()
        self.fields = fields
        self.linear = nn.Embedding(n_features, 1)
        self.latent = nn.Embedding(n_features, fields * rank)
        self.rank = rank
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, std=0.015)

    def forward(self, x):
        linear = self.linear(x).squeeze(-1).sum(dim=1)
        latent = self.latent(x).view(
            x.shape[0], self.fields, self.fields, self.rank
        )

        interaction = torch.zeros_like(linear)
        for i in range(self.fields):
            for j in range(i + 1, self.fields):
                interaction = interaction + (
                    latent[:, i, j, :] * latent[:, j, i, :]
                ).sum(dim=1)

        return self.bias + linear + interaction


class FiBiNET(nn.Module):
    """SENET feature recalibration followed by field-specific bilinear pairs."""

    def __init__(self, n_features, fields, emb_dim=10):
        super().__init__()
        self.fields = fields
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(n_features, emb_dim)

        hidden = max(2, fields // 2)
        self.se_fc1 = nn.Linear(fields, hidden)
        self.se_fc2 = nn.Linear(hidden, fields)

        self.pairs = [
            (i, j) for i in range(fields) for j in range(i + 1, fields)
        ]
        self.bilinear = nn.Parameter(
            torch.empty(len(self.pairs), emb_dim, emb_dim)
        )

        input_dim = fields * emb_dim + len(self.pairs) * emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.bilinear)

    def forward(self, x):
        embeddings = self.embedding(x)
        descriptors = embeddings.mean(dim=2)
        gates = torch.sigmoid(
            self.se_fc2(F.relu(self.se_fc1(descriptors)))
        ).unsqueeze(-1)
        recalibrated = embeddings * gates

        products = []
        for pair_index, (i, j) in enumerate(self.pairs):
            transformed = recalibrated[:, i, :] @ self.bilinear[pair_index]
            products.append(transformed * recalibrated[:, j, :])

        pair_features = torch.cat(products, dim=1)
        flat_embeddings = recalibrated.flatten(1)
        return self.mlp(
            torch.cat([flat_embeddings, pair_features], dim=1)
        ).squeeze(1)


class WideDeep(nn.Module):
    """Memorized category effects plus a nonlinear joint representation."""

    def __init__(self, n_features, fields, emb_dim=12):
        super().__init__()
        self.wide = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, emb_dim)
        dim = fields * emb_dim

        self.deep = nn.Sequential(
            nn.Linear(dim, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.wide.weight)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x):
        wide_score = self.wide(x).squeeze(-1).sum(dim=1)
        deep_score = self.deep(self.embedding(x).flatten(1)).squeeze(1)
        return self.bias + wide_score + deep_score


class PLE(nn.Module):
    """Shared and task-specific experts with separate task gates."""

    def __init__(
        self,
        n_features,
        fields,
        emb_dim=10,
        tasks=3,
        shared_experts=2,
        specific_experts=2,
    ):
        super().__init__()
        self.tasks = tasks
        self.shared_experts_count = shared_experts
        self.specific_experts_count = specific_experts

        self.embedding = nn.Embedding(n_features, emb_dim)
        input_dim = fields * emb_dim
        expert_dim = 32

        def expert():
            return nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, expert_dim),
                nn.ReLU(),
            )

        self.shared_experts = nn.ModuleList(
            [expert() for _ in range(shared_experts)]
        )
        self.task_experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [expert() for _ in range(specific_experts)]
                )
                for _ in range(tasks)
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Linear(
                    input_dim, shared_experts + specific_experts
                )
                for _ in range(tasks)
            ]
        )
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(expert_dim, 24),
                    nn.ReLU(),
                    nn.Linear(24, 1),
                )
                for _ in range(tasks)
            ]
        )

        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x):
        base = self.embedding(x).flatten(1)
        shared_values = [expert(base) for expert in self.shared_experts]

        outputs = []
        for task in range(self.tasks):
            specific_values = [
                expert(base) for expert in self.task_experts[task]
            ]
            all_values = torch.stack(
                shared_values + specific_values, dim=1
            )
            gate = torch.softmax(self.gates[task](base), dim=1).unsqueeze(-1)
            representation = (all_values * gate).sum(dim=1)
            outputs.append(
                self.towers[task](representation).squeeze(1)
            )

        return torch.stack(outputs, dim=1)


valid_predictions = {}
test_predictions = {}

torch.manual_seed(SEED + 1)
ffm = FieldAwareFM(total_cardinality, n_fields, rank=10)
train_binary(ffm, epochs=4, learning_rate=0.0012)
valid_predictions["field_aware_fm"] = predict(ffm, x_valid)
test_predictions["field_aware_fm"] = predict(ffm, x_test)
del ffm

torch.manual_seed(SEED + 2)
fibinet = FiBiNET(total_cardinality, n_fields, emb_dim=10)
train_binary(fibinet, epochs=3, learning_rate=0.0015)
valid_predictions["fibinet"] = predict(fibinet, x_valid)
test_predictions["fibinet"] = predict(fibinet, x_test)
del fibinet

torch.manual_seed(SEED + 3)
wide_deep = WideDeep(total_cardinality, n_fields, emb_dim=12)
train_binary(wide_deep, epochs=3, learning_rate=0.0015)
valid_predictions["wide_deep"] = predict(wide_deep, x_valid)
test_predictions["wide_deep"] = predict(wide_deep, x_test)
del wide_deep

torch.manual_seed(SEED + 4)
ple = PLE(total_cardinality, n_fields)
train_ple(ple, epochs=3, learning_rate=0.0015)
valid_predictions["ple_multitask"] = predict(
    ple, x_valid, output_index=0
)
test_predictions["ple_multitask"] = predict(
    ple, x_test, output_index=0
)
del ple

shared_dir = os.environ.get("SHARED_ARTIFACTS")
if not shared_dir:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared_dir, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared_dir, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

valid_slate_size = user_slate_sizes(valid.user_id)
test_slate_size = user_slate_sizes(test.user_id)

candidate_scores = {}
candidate_valid = {}
candidate_test = {}
candidate_raw = {}
candidate_is_blend = {}

# The grids are deliberately coarse because sub-0.002 changes are noisy.
global_alphas = [0.15, 0.25, 0.35, 0.50]
shrink_alphas = [0.25, 0.40, 0.55]

for family, raw_valid in valid_predictions.items():
    raw_test = test_predictions[family]
    own_valid_rank = within_user_rank(valid.user_id, raw_valid)
    own_test_rank = within_user_rank(test.user_id, raw_test)

    raw_metrics = evaluate(valid.user_id, valid.y, raw_valid)
    candidate_scores[family] = float(raw_metrics["primary"])
    candidate_valid[family] = raw_valid
    candidate_test[family] = raw_test
    candidate_raw[family] = raw_valid
    candidate_is_blend[family] = False

    for alpha in global_alphas:
        name = family + "_blend_" + str(alpha)
        blend_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * own_valid_rank
        )
        blend_test = (
            (1.0 - alpha) * inc_test_rank
            + alpha * own_test_rank
        )
        score = evaluate(
            valid.user_id, valid.y, blend_valid
        )["primary"]

        candidate_scores[name] = float(score)
        candidate_valid[name] = blend_valid
        candidate_test[name] = blend_test
        candidate_raw[name] = raw_valid
        candidate_is_blend[name] = True

    # User-specific shrinkage: for short slates, a complementary model has
    # very little evidence with which to alter the trusted ordering. Its
    # weight rises smoothly and reaches the base weight at eight impressions,
    # the region carrying most of the incumbent's reducible nDCG gap.
    valid_reliability = np.minimum(1.0, valid_slate_size / 8.0)
    test_reliability = np.minimum(1.0, test_slate_size / 8.0)

    for base_alpha in shrink_alphas:
        valid_alpha = base_alpha * valid_reliability
        test_alpha = base_alpha * test_reliability

        name = family + "_slate_shrink_" + str(base_alpha)
        blend_valid = (
            (1.0 - valid_alpha) * inc_valid_rank
            + valid_alpha * own_valid_rank
        )
        blend_test = (
            (1.0 - test_alpha) * inc_test_rank
            + test_alpha * own_test_rank
        )
        score = evaluate(
            valid.user_id, valid.y, blend_valid
        )["primary"]

        candidate_scores[name] = float(score)
        candidate_valid[name] = blend_valid
        candidate_test[name] = blend_test
        candidate_raw[name] = raw_valid
        candidate_is_blend[name] = True

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_valid[winner]
test_scores = candidate_test[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if candidate_is_blend[winner]:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[winner], dtype=np.float64),
        )

family_raw_summary = {
    family: candidate_scores[family]
    for family in valid_predictions
}
print("FINDINGS raw_family_primary=" + json.dumps(
    family_raw_summary, sort_keys=True
))
print("FINDINGS winner=" + winner)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(time.time() - START),
        }
    )
)