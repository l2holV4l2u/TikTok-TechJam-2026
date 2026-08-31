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
SEED = 1729
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
BATCH_SIZE = 4096
EMBED_DIM = 16
LR = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
device = torch.device("cpu")

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
n_fields = len(FIELDS)


def make_features(split):
    a = np.stack(
        [
            np.asarray(split.X[f], dtype=np.int64) + offsets[j]
            for j, f in enumerate(FIELDS)
        ],
        axis=1,
    )
    return torch.from_numpy(a)


def temporal_weights(dates, half_life):
    if half_life is None:
        return torch.ones(len(dates), dtype=torch.float32)
    d = np.asarray(dates, dtype=np.float32)
    age = float(np.max(d)) - d
    w = np.exp2(-age / float(half_life)).astype(np.float32)
    w /= max(float(w.mean()), 1e-8)
    return torch.from_numpy(w)


class ExpandedFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        sv = v.sum(dim=1)
        fm = 0.5 * (sv.square() - v.square().sum(dim=1)).sum(dim=1)
        return self.bias + linear + fm


class DeepFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.bias = nn.Parameter(torch.zeros(1))
        dim = n_fields * EMBED_DIM
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        sv = v.sum(dim=1)
        fm = 0.5 * (sv.square() - v.square().sum(dim=1)).sum(dim=1)
        deep = self.deep(v.flatten(1)).squeeze(-1)
        return self.bias + linear + fm + deep


class DCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.bias = nn.Parameter(torch.zeros(1))
        dim = n_fields * EMBED_DIM

        self.cross_w = nn.ParameterList(
            [nn.Parameter(torch.empty(dim)) for _ in range(2)]
        )
        self.cross_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in range(2)]
        )
        for w in self.cross_w:
            nn.init.normal_(w, std=0.01)

        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 64, 1)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        x0 = self.embedding(x).flatten(1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = (cross * w).sum(dim=1, keepdim=True)
            cross = x0 * scalar + b + cross
        deep = self.deep(x0)
        interaction = self.output(torch.cat([cross, deep], dim=1)).squeeze(-1)
        return self.bias + linear + interaction


SPECS = {
    "fm_uniform": {
        "kind": "fm",
        "epochs": 5,
        "half_life": None,
    },
    "fm_recency4": {
        "kind": "fm",
        "epochs": 5,
        "half_life": 4.0,
    },
    "deepfm": {
        "kind": "deepfm",
        "epochs": 3,
        "half_life": None,
    },
    "dcn": {
        "kind": "dcn",
        "epochs": 3,
        "half_life": None,
    },
}


def build_model(kind):
    if kind == "fm":
        return ExpandedFM()
    if kind == "deepfm":
        return DeepFM()
    if kind == "dcn":
        return DCN()
    raise ValueError(kind)


def fit_model(x, y, dates, spec, seed):
    torch.manual_seed(seed)
    model = build_model(spec["kind"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    weights = temporal_weights(dates, spec["half_life"])

    n = x.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 10000)

    model.train()
    for _ in range(int(spec["epochs"])):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = x[idx].to(device)
            yb = y[idx].to(device)
            wb = weights[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (losses * wb).mean()
            loss.backward()
            optimizer.step()
    return model


@torch.inference_mode()
def predict(model, x, batch_size=16384):
    model.eval()
    out = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        logits = model(x[start:end].to(device))
        out[start:end] = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    return out


train = load("train")
valid = load("valid")

x_train = make_features(train)
x_valid = make_features(valid)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path):
    raise RuntimeError("Trusted incumbent validation predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

valid_predictions = {}
candidate_predictions = {}
candidate_recipes = {}
candidate_scores = {}

inc_metrics = evaluate(valid.user_id, valid.y, inc_valid)
candidate_predictions["incumbent"] = inc_valid
candidate_recipes["incumbent"] = {"incumbent": 1.0, "models": {}}
candidate_scores["incumbent"] = float(inc_metrics["primary"])

# Fit all structurally different families on train only.
for i, (name, spec) in enumerate(SPECS.items()):
    model = fit_model(
        x_train,
        y_train,
        train.date,
        spec,
        SEED + 101 * i,
    )
    pred = predict(model, x_valid)
    valid_predictions[name] = pred

    met = evaluate(valid.user_id, valid.y, pred)
    candidate_predictions[name] = pred
    candidate_recipes[name] = {
        "incumbent": 0.0,
        "models": {name: 1.0},
    }
    candidate_scores[name] = float(met["primary"])
    del model

# Blend each family with the trusted incumbent. The blend coefficient is
# selected exclusively on validation and later applied unchanged to test.
for name, pred in valid_predictions.items():
    for alpha in (0.2, 0.4, 0.6, 0.8):
        cname = "%s_inc%.1f" % (name, 1.0 - alpha)
        blended = alpha * pred + (1.0 - alpha) * inc_valid
        met = evaluate(valid.user_id, valid.y, blended)
        candidate_predictions[cname] = blended
        candidate_recipes[cname] = {
            "incumbent": 1.0 - alpha,
            "models": {name: alpha},
        }
        candidate_scores[cname] = float(met["primary"])

# Also test cross-family averages, both alone and lightly anchored by the
# trusted incumbent. This can preserve complementary pair orderings.
family_names = list(SPECS.keys())
for i in range(len(family_names)):
    for j in range(i + 1, len(family_names)):
        a = family_names[i]
        b = family_names[j]
        pair = 0.5 * valid_predictions[a] + 0.5 * valid_predictions[b]

        cname = "%s_%s_avg" % (a, b)
        met = evaluate(valid.user_id, valid.y, pair)
        candidate_predictions[cname] = pair
        candidate_recipes[cname] = {
            "incumbent": 0.0,
            "models": {a: 0.5, b: 0.5},
        }
        candidate_scores[cname] = float(met["primary"])

        cname2 = "%s_%s_inc20" % (a, b)
        triple = 0.4 * valid_predictions[a] + 0.4 * valid_predictions[b] + 0.2 * inc_valid
        met2 = evaluate(valid.user_id, valid.y, triple)
        candidate_predictions[cname2] = triple
        candidate_recipes[cname2] = {
            "incumbent": 0.2,
            "models": {a: 0.4, b: 0.4},
        }
        candidate_scores[cname2] = float(met2["primary"])

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = np.asarray(candidate_predictions[winner], dtype=np.float64)
metrics = evaluate(valid.user_id, valid.y, valid_scores)
winner_recipe = candidate_recipes[winner]

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS winner=%s uniform_fm=%.6f recency4_fm=%.6f delta_recency=%+.6f"
    % (
        winner,
        candidate_scores["fm_uniform"],
        candidate_scores["fm_recency4"],
        candidate_scores["fm_recency4"] - candidate_scores["fm_uniform"],
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores,
    )

# Refit precisely the selected recipe on train + validation, then score test.
test = load("test")
x_test = make_features(test)

test_scores = np.zeros(x_test.shape[0], dtype=np.float64)

inc_weight = float(winner_recipe["incumbent"])
if inc_weight != 0.0:
    if not os.path.exists(inc_test_path):
        raise RuntimeError("Trusted incumbent test predictions are unavailable")
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    test_scores += inc_weight * inc_test

if winner_recipe["models"]:
    y_valid = torch.from_numpy(np.asarray(valid.y, dtype=np.float32))
    x_combined = torch.cat([x_train, x_valid], dim=0)
    y_combined = torch.cat([y_train, y_valid], dim=0)
    combined_dates = np.concatenate(
        [
            np.asarray(train.date),
            np.asarray(valid.date),
        ]
    )

    for name, coefficient in winner_recipe["models"].items():
        spec_index = family_names.index(name)
        final_model = fit_model(
            x_combined,
            y_combined,
            combined_dates,
            SPECS[name],
            SEED + 101 * spec_index,
        )
        test_scores += float(coefficient) * predict(final_model, x_test)
        del final_model

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