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
SEED = 271828
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
]
RANK = 16
DEEP_RANK = 12
BATCH_SIZE = 8192
FM_EPOCHS = 10
DEEP_EPOCHS = 7
LR = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))
DEVICE = torch.device("cpu")

cards = np.asarray(
    [int(FEATURE_CARDINALITIES[f]) for f in FIELDS], dtype=np.int64
)
offsets = np.cumsum(
    np.concatenate([np.zeros(1, dtype=np.int64), cards[:-1]])
)
total_cardinality = int(cards.sum())


def local_x(split):
    return np.ascontiguousarray(
        np.column_stack([split.X[f] for f in FIELDS]).astype(
            np.int64, copy=False
        )
    )


def offset_x(x):
    return np.ascontiguousarray(x + offsets[None, :])


class ExpandedFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, RANK)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


class DeepFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, DEEP_RANK)
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(len(FIELDS) * DEEP_RANK, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        fm = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.mlp(v.flatten(1)).squeeze(1)
        return self.bias + linear + fm + deep


@torch.no_grad()
def predict_torch(model, x):
    model.eval()
    out = np.empty(x.shape[0], dtype=np.float64)
    step = BATCH_SIZE * 2
    for start in range(0, x.shape[0], step):
        end = min(start + step, x.shape[0])
        out[start:end] = (
            model(x[start:end]).detach().cpu().numpy().astype(np.float64)
        )
    return out


def train_neural_family(model_class, x_train, y_train, x_valid,
                        valid_users, valid_y, max_epochs, seed):
    torch.manual_seed(seed)
    model = model_class().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None
    best_state = None
    epoch_scores = []

    n = x_train.shape[0]
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(x_train[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_train[idx]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        scores = predict_torch(model, x_valid)
        metrics = evaluate(valid_users, valid_y, scores)
        primary = float(metrics["primary"])
        epoch_scores.append(primary)
        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch
            best_scores = scores.copy()
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return {
        "scores": best_scores,
        "epoch": best_epoch,
        "state": best_state,
        "epoch_scores": epoch_scores,
        "primary": best_primary,
    }


def fit_fixed_epochs(model_class, x, y, epochs, seed):
    torch.manual_seed(seed)
    model = model_class().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    n = x.shape[0]

    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(x[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y[idx]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def fit_empirical_bayes(x, y, alpha=30.0):
    y64 = np.asarray(y, dtype=np.float64)
    prior = float(y64.mean())
    tables = []
    for j, card in enumerate(cards):
        ids = x[:, j]
        count = np.bincount(ids, minlength=int(card)).astype(np.float64)
        positive = np.bincount(
            ids, weights=y64, minlength=int(card)
        ).astype(np.float64)
        rate = (positive + alpha * prior) / (count + alpha)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        tables.append(np.log(rate / (1.0 - rate)))
    return tables


def predict_empirical_bayes(x, tables):
    score = np.zeros(x.shape[0], dtype=np.float64)
    # Identity fields receive slightly more weight, while all side fields
    # contribute stationary item/context evidence.
    weights = np.asarray(
        [1.5, 1.5, 1.2, 0.8, 1.0, 1.1, 0.8, 0.8, 0.7],
        dtype=np.float64,
    )
    for j, table in enumerate(tables):
        score += weights[j] * table[x[:, j]]
    return score / weights.sum()


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return (x - float(x.mean())) / sd


train = load("train")
valid = load("valid")

x_train_local = local_x(train)
x_valid_local = local_x(valid)
x_train_np = offset_x(x_train_local)
x_valid_np = offset_x(x_valid_local)

y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train_t = torch.from_numpy(x_train_np).to(DEVICE)
x_valid_t = torch.from_numpy(x_valid_np).to(DEVICE)
y_train_t = torch.from_numpy(y_train_np).to(DEVICE)

fm_result = train_neural_family(
    ExpandedFM,
    x_train_t,
    y_train_t,
    x_valid_t,
    valid.user_id,
    y_valid,
    FM_EPOCHS,
    SEED,
)
deep_result = train_neural_family(
    DeepFM,
    x_train_t,
    y_train_t,
    x_valid_t,
    valid.user_id,
    y_valid,
    DEEP_EPOCHS,
    SEED + 1,
)

eb_tables = fit_empirical_bayes(x_train_local, y_train_np, alpha=30.0)
eb_valid = predict_empirical_bayes(x_valid_local, eb_tables)

family_valid = {
    "expanded_fm": fm_result["scores"],
    "deepfm": deep_result["scores"],
    "empirical_bayes": eb_valid,
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

candidate_scores = {}
candidate_underlying = {}
candidate_weight = {}

for family_name, raw_scores in family_valid.items():
    standalone_metrics = evaluate(valid.user_id, y_valid, raw_scores)
    candidate_scores[family_name] = float(standalone_metrics["primary"])
    candidate_underlying[family_name] = family_name
    candidate_weight[family_name] = 1.0

    own_z = standardize(raw_scores)
    inc_z = standardize(inc_valid)
    for weight in (0.2, 0.4, 0.6, 0.8):
        name = "%s_blend_%.1f" % (family_name, weight)
        blended = weight * own_z + (1.0 - weight) * inc_z
        metrics_blend = evaluate(valid.user_id, y_valid, blended)
        candidate_scores[name] = float(metrics_blend["primary"])
        candidate_underlying[name] = family_name
        candidate_weight[name] = float(weight)

inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metrics["primary"])
candidate_underlying["trusted_incumbent"] = "incumbent"
candidate_weight["trusted_incumbent"] = 0.0

winner = max(candidate_scores, key=candidate_scores.get)
winner_family = candidate_underlying[winner]
winner_weight = candidate_weight[winner]

if winner_family == "incumbent":
    valid_scores = inc_valid.copy()
    own_valid_scores = None
elif winner_weight >= 1.0:
    own_valid_scores = family_valid[winner_family]
    valid_scores = own_valid_scores.copy()
else:
    own_valid_scores = family_valid[winner_family]
    valid_scores = (
        winner_weight * standardize(own_valid_scores)
        + (1.0 - winner_weight) * standardize(inc_valid)
    )

metrics = evaluate(valid.user_id, y_valid, valid_scores)

print(
    "FINDINGS "
    + json.dumps(
        {
            "fm_best_epoch": int(fm_result["epoch"]),
            "fm_epoch_primary": [
                round(float(v), 6) for v in fm_result["epoch_scores"]
            ],
            "deepfm_best_epoch": int(deep_result["epoch"]),
            "deepfm_epoch_primary": [
                round(float(v), 6) for v in deep_result["epoch_scores"]
            ],
            "winner": winner,
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: round(float(v), 10) for k, v in candidate_scores.items()},
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if own_valid_scores is not None and winner_weight < 1.0:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(own_valid_scores, dtype=np.float64),
        )

# Refit only the family selected on validation, using train + validation.
test = load("test")
x_test_local = local_x(test)

if winner_family == "incumbent":
    test_scores = np.asarray(np.load(inc_test_path), dtype=np.float64)
else:
    x_combined_local = np.ascontiguousarray(
        np.concatenate([x_train_local, x_valid_local], axis=0)
    )
    y_combined_np = np.ascontiguousarray(
        np.concatenate(
            [y_train_np, y_valid.astype(np.float32)], axis=0
        )
    )

    if winner_family == "empirical_bayes":
        final_tables = fit_empirical_bayes(
            x_combined_local, y_combined_np, alpha=30.0
        )
        own_test_scores = predict_empirical_bayes(
            x_test_local, final_tables
        )
    else:
        x_combined_np = offset_x(x_combined_local)
        x_test_np = offset_x(x_test_local)
        x_combined_t = torch.from_numpy(x_combined_np).to(DEVICE)
        y_combined_t = torch.from_numpy(y_combined_np).to(DEVICE)
        x_test_t = torch.from_numpy(x_test_np).to(DEVICE)

        if winner_family == "expanded_fm":
            final_model = fit_fixed_epochs(
                ExpandedFM,
                x_combined_t,
                y_combined_t,
                int(fm_result["epoch"]),
                SEED,
            )
        else:
            final_model = fit_fixed_epochs(
                DeepFM,
                x_combined_t,
                y_combined_t,
                int(deep_result["epoch"]),
                SEED + 1,
            )
        own_test_scores = predict_torch(final_model, x_test_t)

    if winner_weight >= 1.0:
        test_scores = own_test_scores
    else:
        inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
        test_scores = (
            winner_weight * standardize(own_test_scores)
            + (1.0 - winner_weight) * standardize(inc_test)
        )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)