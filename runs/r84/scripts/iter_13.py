import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 314159
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

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
EB_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
]

CARDS = np.asarray(
    [int(FEATURE_CARDINALITIES[f]) for f in FIELDS],
    dtype=np.int64
)
OFFSETS = np.concatenate([
    np.zeros(1, dtype=np.int64),
    np.cumsum(CARDS[:-1], dtype=np.int64)
])
TOTAL_CARD = int(CARDS.sum())


def make_matrix(splits):
    columns = []
    for field in FIELDS:
        columns.append(np.concatenate([
            np.asarray(s.X[field], dtype=np.int64) for s in splits
        ]))
    return np.column_stack(columns).astype(np.int64, copy=False)


def make_labels(label_arrays):
    return np.concatenate([
        np.asarray(y, dtype=np.float32) for y in label_arrays
    ])


class ExpandedFM(nn.Module):
    def __init__(self, total_cardinality, offsets, rank=16):
        super().__init__()
        self.register_buffer(
            "offsets",
            torch.as_tensor(offsets, dtype=torch.long)
        )
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, rank)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        x = x + self.offsets
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        vectors = self.embedding(x)
        summed = vectors.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - vectors.square().sum(dim=(1, 2))
        )
        return self.bias + linear + interaction


class ExplicitCrossDCN(nn.Module):
    def __init__(self, cards, embedding_dim=8, cross_layers=2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(card), embedding_dim) for card in cards
        ])
        width = len(cards) * embedding_dim

        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(width)) for _ in range(cross_layers)
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)
        ])
        for w in self.cross_w:
            nn.init.normal_(w, std=0.01)

        self.deep = nn.Sequential(
            nn.Linear(width, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU()
        )
        self.output = nn.Linear(width + 48, 1)

        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, x):
        x0 = torch.cat([
            emb(x[:, j]) for j, emb in enumerate(self.embeddings)
        ], dim=1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = torch.sum(cross * w, dim=1, keepdim=True)
            cross = x0 * scalar + b + cross
        deep = self.deep(x0)
        return self.output(torch.cat([cross, deep], dim=1)).squeeze(1)


def train_torch_model(model_kind, X, y):
    torch.manual_seed(SEED)
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    if model_kind == "expanded_fm":
        model = ExpandedFM(TOTAL_CARD, OFFSETS, rank=16)
        epochs = 6
        batch_size = 16384
        lr = 1.0e-3
        weight_decay = 0.0
    elif model_kind == "explicit_cross_dcn":
        model = ExplicitCrossDCN(CARDS, embedding_dim=8, cross_layers=2)
        epochs = 4
        batch_size = 16384
        lr = 1.2e-3
        weight_decay = 1.0e-6
    else:
        raise ValueError(model_kind)

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    n = len(y)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    for epoch in range(epochs):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_tensor[idx])
            loss = criterion(logits, y_tensor[idx])
            loss.backward()
            optimizer.step()

    return model


def predict_torch(model, X, batch_size=65536):
    model.eval()
    X_tensor = torch.from_numpy(X)
    result = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            logits = model(X_tensor[start:stop])
            result[start:stop] = logits.numpy().astype(np.float64)
    return result


def fit_empirical_bayes(splits, labels, smoothing=35.0):
    y = make_labels(labels).astype(np.float64)
    global_rate = float(y.mean())
    global_logit = np.log(
        np.clip(global_rate, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - global_rate, 1e-6, 1.0)
    )

    tables = {}
    for field in EB_FIELDS:
        ids = np.concatenate([
            np.asarray(s.X[field], dtype=np.int64) for s in splits
        ])
        cardinality = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(ids, minlength=cardinality).astype(np.float64)
        positives = np.bincount(
            ids, weights=y, minlength=cardinality
        ).astype(np.float64)
        rates = (
            positives + smoothing * global_rate
        ) / (counts + smoothing)
        logits = np.log(
            np.clip(rates, 1e-6, 1.0 - 1e-6)
            / np.clip(1.0 - rates, 1e-6, 1.0)
        )
        # Center each contribution so fields with many rare categories do not
        # repeatedly add the global intercept.
        tables[field] = logits - global_logit

    return global_logit, tables


def predict_empirical_bayes(model, target):
    global_logit, tables = model
    score = np.full(len(target.user_id), global_logit, dtype=np.float64)

    # Identity/content fields receive more weight; context fields stabilize
    # candidates that have sparse video histories.
    weights = {
        "video_id": 1.00,
        "author_id": 0.75,
        "tag": 0.45,
        "duration_bucket": 0.35,
        "upload_type": 0.25,
        "music_type": 0.20,
        "tab": 0.30,
        "hour": 0.20,
    }
    for field in EB_FIELDS:
        ids = np.asarray(target.X[field], dtype=np.int64)
        score += weights[field] * tables[field][ids]
    return score


def standardized_blend(own_scores, incumbent_scores, own_weight):
    own = np.asarray(own_scores, dtype=np.float64)
    incumbent = np.asarray(incumbent_scores, dtype=np.float64)
    own_z = (own - own.mean()) / max(float(own.std()), 1e-8)
    inc_z = (
        incumbent - incumbent.mean()
    ) / max(float(incumbent.std()), 1e-8)
    return own_weight * own_z + (1.0 - own_weight) * inc_z


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

X_train = make_matrix([train])
X_valid = make_matrix([valid])

shared = os.environ["SHARED_ARTIFACTS"]
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)

raw_predictions = {}

fm_model = train_torch_model("expanded_fm", X_train, y_train)
raw_predictions["expanded_fm"] = predict_torch(fm_model, X_valid)

dcn_model = train_torch_model("explicit_cross_dcn", X_train, y_train)
raw_predictions["explicit_cross_dcn"] = predict_torch(
    dcn_model, X_valid
)

eb_model = fit_empirical_bayes([train], [y_train])
raw_predictions["empirical_bayes"] = predict_empirical_bayes(
    eb_model, valid
)

blend_weights = [0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0]
candidate_results = {}
best_primary = -np.inf
best_family = None
best_weight = None
best_scores = None
best_raw = None
best_metrics = None

for family, raw_scores in raw_predictions.items():
    raw_metrics = evaluate(valid.user_id, y_valid, raw_scores)
    candidate_results[f"{family}_raw"] = float(
        raw_metrics["primary"]
    )

    for weight in blend_weights:
        scores = standardized_blend(raw_scores, inc_valid, weight)
        metrics = evaluate(valid.user_id, y_valid, scores)
        candidate_results[
            f"{family}_blend_{weight:.2f}"
        ] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_family = family
            best_weight = float(weight)
            best_scores = scores.copy()
            best_raw = raw_scores.copy()
            best_metrics = metrics

correlations = {
    "fm_dcn": float(np.corrcoef(
        raw_predictions["expanded_fm"],
        raw_predictions["explicit_cross_dcn"]
    )[0, 1]),
    "fm_eb": float(np.corrcoef(
        raw_predictions["expanded_fm"],
        raw_predictions["empirical_bayes"]
    )[0, 1]),
    "dcn_eb": float(np.corrcoef(
        raw_predictions["explicit_cross_dcn"],
        raw_predictions["empirical_bayes"]
    )[0, 1]),
}

print("CANDIDATES " + json.dumps(
    candidate_results, sort_keys=True
))
print("FINDINGS " + json.dumps({
    "winner_family": best_family,
    "winner_own_weight": best_weight,
    "raw_correlations": correlations
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )
    if best_weight < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64)
        )

# Release train-only models before the train+validation refit.
del fm_model
del dcn_model
del eb_model
del X_train
del X_valid
gc.collect()

test = load("test")
X_fit = None
X_test = None

if best_family in ("expanded_fm", "explicit_cross_dcn"):
    X_fit = make_matrix([train, valid])
    y_fit = make_labels([y_train, y_valid])
    X_test = make_matrix([test])
    final_model = train_torch_model(best_family, X_fit, y_fit)
    raw_test = predict_torch(final_model, X_test)
else:
    final_model = fit_empirical_bayes(
        [train, valid], [y_train, y_valid]
    )
    raw_test = predict_empirical_bayes(final_model, test)

if best_weight < 1.0:
    inc_test = np.load(inc_test_path).astype(np.float64)
    test_scores = standardized_blend(
        raw_test, inc_test, best_weight
    )
else:
    test_scores = raw_test

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed)
}))