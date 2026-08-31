import os
import time
import json
import math
import gc
import copy
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2025
FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour"
]
RANK = 16
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 65536
MAX_EPOCHS = 10
LR = 0.001

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def make_offsets():
    offsets = []
    total = 0
    for field in FIELDS:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[field])
    return np.asarray(offsets, dtype=np.int64), total


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def feature_matrix(split):
    x = np.empty((len(split.user_id), len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[field], dtype=np.int64) + OFFSETS[j]
    return x


class CTRModel(nn.Module):
    def __init__(self, kind, cardinality, rank, initial_rate):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        self.bias = nn.Parameter(torch.tensor(
            math.log(initial_rate / max(1e-6, 1.0 - initial_rate)),
            dtype=torch.float32
        ))

        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.01)

        if kind == "deepfm":
            self.deep = nn.Sequential(
                nn.Linear(len(FIELDS) * rank, 96),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Linear(48, 1)
            )
        elif kind == "nfm":
            self.deep = nn.Sequential(
                nn.Linear(rank, 64),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        elif kind == "fm":
            self.deep = None
        else:
            raise ValueError(kind)

        if self.deep is not None:
            last = [m for m in self.deep.modules() if isinstance(m, nn.Linear)][-1]
            with torch.no_grad():
                last.weight.normal_(0.0, 0.002)
                last.bias.zero_()

    def forward(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        v = e[:, :, 1:]
        summed = v.sum(dim=1)
        bi_vector = 0.5 * (summed.square() - v.square().sum(dim=1))
        fm_interaction = bi_vector.sum(dim=1)

        if self.kind == "fm":
            extra = fm_interaction
        elif self.kind == "deepfm":
            extra = fm_interaction + self.deep(v.reshape(v.shape[0], -1)).squeeze(1)
        else:
            extra = self.deep(bi_vector).squeeze(1)

        return self.bias + linear + extra


def predict_torch(model, x_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    x_tensor = torch.from_numpy(x_np)
    with torch.no_grad():
        for start in range(0, len(x_np), PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, len(x_np))
            result[start:end] = model(x_tensor[start:end]).cpu().numpy()
    return result


def fit_torch(kind, x_train, y_train, epochs, validation=None):
    torch.manual_seed(SEED + {"fm": 1, "deepfm": 2, "nfm": 3}[kind])
    model = CTRModel(kind, TOTAL_CARDINALITY, RANK, float(np.mean(y_train)))

    sparse_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LR
    )
    dense_params = [
        p for name, p in model.named_parameters()
        if name != "embedding.weight"
    ]
    dense_optimizer = torch.optim.Adam(dense_params, lr=LR)

    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train.astype(np.float32, copy=False))
    generator = torch.Generator()
    generator.manual_seed(SEED + 100 + {"fm": 1, "deepfm": 2, "nfm": 3}[kind])

    best_primary = -np.inf
    best_epoch = int(epochs)
    best_scores = None
    best_metrics = None
    best_state = None
    n = len(x_train)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        permutation = torch.randperm(n, generator=generator)

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:min(start + BATCH_SIZE, n)]
            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)

            logits = model(x_tensor[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_tensor[idx]
            )
            loss.backward()
            sparse_optimizer.step()
            dense_optimizer.step()

        if validation is not None:
            x_valid, y_valid, valid_users = validation
            scores = predict_torch(model, x_valid)
            metrics = evaluate(valid_users, y_valid, scores)
            if float(metrics["primary"]) > best_primary:
                best_primary = float(metrics["primary"])
                best_epoch = epoch
                best_scores = scores.copy()
                best_metrics = metrics
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if validation is not None:
        model.load_state_dict(best_state)
        return model, best_epoch, best_scores, best_metrics
    return model


EB_FIELDS = [
    "video_id", "author_id", "tag", "upload_type",
    "music_type", "duration_bucket", "tab"
]
EB_SMOOTHING = {
    "video_id": 25.0,
    "author_id": 35.0,
    "tag": 100.0,
    "upload_type": 150.0,
    "music_type": 150.0,
    "duration_bucket": 120.0,
    "tab": 150.0
}
EB_WEIGHTS = {
    "video_id": 1.0,
    "author_id": 0.8,
    "tag": 0.55,
    "upload_type": 0.30,
    "music_type": 0.25,
    "duration_bucket": 0.35,
    "tab": 0.45
}


def fit_eb(split, labels):
    prior = float(np.mean(labels))
    tables = {}
    for field in EB_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(ids, minlength=cardinality).astype(np.float64)
        positives = np.bincount(
            ids, weights=labels.astype(np.float64, copy=False),
            minlength=cardinality
        )
        smooth = EB_SMOOTHING[field]
        rates = (positives + smooth * prior) / (counts + smooth)
        tables[field] = rates.astype(np.float32)
    return prior, tables


def predict_eb(split, fitted):
    prior, tables = fitted
    eps = 1e-5
    base = math.log((prior + eps) / (1.0 - prior + eps))
    scores = np.full(len(split.user_id), base, dtype=np.float32)
    prior_logit = base

    for field in EB_FIELDS:
        rates = np.clip(
            tables[field][np.asarray(split.X[field], dtype=np.int64)],
            eps, 1.0 - eps
        )
        logits = np.log(rates / (1.0 - rates))
        scores += EB_WEIGHTS[field] * (logits - prior_logit)
    return scores


def standardized_blend(own, incumbent, weight, own_mu, own_sd, inc_mu, inc_sd):
    own_z = (own - own_mu) / own_sd
    inc_z = (incumbent - inc_mu) / inc_sd
    return weight * own_z + (1.0 - weight) * inc_z


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
x_train = feature_matrix(train)
x_valid = feature_matrix(valid)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

family_scores = {}
family_epochs = {}
candidate_scores = {}
candidate_metrics = {}

for kind in ["fm", "deepfm", "nfm"]:
    model, best_epoch, scores, metrics = fit_torch(
        kind, x_train, y_train, MAX_EPOCHS,
        validation=(x_valid, y_valid, valid.user_id)
    )
    family_scores[kind] = np.asarray(scores, dtype=np.float64)
    family_epochs[kind] = int(best_epoch)
    candidate_scores[kind] = float(metrics["primary"])
    candidate_metrics[kind] = metrics
    del model
    gc.collect()

eb_fit = fit_eb(train, y_train)
eb_scores = np.asarray(predict_eb(valid, eb_fit), dtype=np.float64)
eb_metrics = evaluate(valid.user_id, y_valid, eb_scores)
family_scores["empirical_bayes"] = eb_scores
candidate_scores["empirical_bayes"] = float(eb_metrics["primary"])
candidate_metrics["empirical_bayes"] = eb_metrics

inc_mu = float(np.mean(inc_valid))
inc_sd = max(float(np.std(inc_valid)), 1e-6)

best_name = None
best_family = None
best_weight = 1.0
best_scores = None
best_metrics = None
blend_metadata = {}

for family, raw_scores in family_scores.items():
    raw_metrics = evaluate(valid.user_id, y_valid, raw_scores)
    if best_metrics is None or float(raw_metrics["primary"]) > float(best_metrics["primary"]):
        best_name = family
        best_family = family
        best_weight = 1.0
        best_scores = raw_scores.copy()
        best_metrics = raw_metrics

    own_mu = float(np.mean(raw_scores))
    own_sd = max(float(np.std(raw_scores)), 1e-6)
    family_best_primary = -np.inf
    family_best_weight = None
    family_best_scores = None
    family_best_metrics = None

    for weight in np.linspace(0.20, 0.95, 16):
        blended = standardized_blend(
            raw_scores, inc_valid, float(weight),
            own_mu, own_sd, inc_mu, inc_sd
        )
        metrics = evaluate(valid.user_id, y_valid, blended)
        if float(metrics["primary"]) > family_best_primary:
            family_best_primary = float(metrics["primary"])
            family_best_weight = float(weight)
            family_best_scores = blended.copy()
            family_best_metrics = metrics

    blend_name = family + "_blend"
    candidate_scores[blend_name] = family_best_primary
    candidate_metrics[blend_name] = family_best_metrics
    blend_metadata[family] = (
        family_best_weight, own_mu, own_sd, inc_mu, inc_sd
    )

    if family_best_primary > float(best_metrics["primary"]):
        best_name = blend_name
        best_family = family
        best_weight = family_best_weight
        best_scores = family_best_scores
        best_metrics = family_best_metrics

print("CANDIDATES " + json.dumps(
    {k: float(v) for k, v in candidate_scores.items()},
    sort_keys=True
))
print("FINDINGS " + json.dumps({
    "selected_family": best_family,
    "selected_candidate": best_name,
    "blend_own_weight": float(best_weight),
    "selected_epochs": family_epochs.get(best_family, None)
}))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )
    if best_weight < 0.999999:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(family_scores[best_family], dtype=np.float64)
        )

# Refit the selected family on train + validation, then score test.
test = load("test")
y_combined = np.concatenate([
    y_train,
    y_valid.astype(np.float32, copy=False)
])

if best_family == "empirical_bayes":
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.user_id = np.concatenate([train.user_id, valid.user_id])
    combined.X = {
        field: np.concatenate([
            np.asarray(train.X[field], dtype=np.int64),
            np.asarray(valid.X[field], dtype=np.int64)
        ])
        for field in EB_FIELDS
    }
    refit = fit_eb(combined, y_combined)
    own_test_scores = np.asarray(predict_eb(test, refit), dtype=np.float64)
else:
    x_combined = np.concatenate([x_train, x_valid], axis=0)
    epochs = family_epochs[best_family]
    refit_model = fit_torch(
        best_family, x_combined, y_combined, epochs, validation=None
    )
    x_test = feature_matrix(test)
    own_test_scores = np.asarray(
        predict_torch(refit_model, x_test), dtype=np.float64
    )

if best_weight < 0.999999:
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    _, own_mu, own_sd, saved_inc_mu, saved_inc_sd = blend_metadata[best_family]
    test_scores = standardized_blend(
        own_test_scores, inc_test, best_weight,
        own_mu, own_sd, saved_inc_mu, saved_inc_sd
    )
else:
    test_scores = own_test_scores

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