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
SEED = 20260829
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
MAX_EPOCHS = 6
EMBED_DIM = 16

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

torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.manual_seed(SEED)
np.random.seed(SEED)

CARDINALITIES = np.asarray(
    [int(FEATURE_CARDINALITIES[f]) for f in FIELDS],
    dtype=np.int64,
)
OFFSETS = np.concatenate(
    [np.zeros(1, dtype=np.int64), np.cumsum(CARDINALITIES[:-1])],
)
TOTAL_CARDINALITY = int(CARDINALITIES.sum())
N_FIELDS = len(FIELDS)


def make_matrix(split):
    local = np.column_stack(
        [np.asarray(split.X[f], dtype=np.int64) for f in FIELDS]
    )
    return np.ascontiguousarray(local + OFFSETS[None, :], dtype=np.int64)


def initial_logit(y):
    p = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
    return float(np.log(p / (1.0 - p)))


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if not np.isfinite(std) or std < 1e-10:
        std = 1.0
    return (x - mean) / std


class BaseCTR(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, EMBED_DIM)
        self.global_bias = nn.Parameter(
            torch.tensor(float(bias), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def linear_score(self, x):
        return self.linear(x).sum(dim=1).squeeze(-1)

    def embedded(self, x):
        return self.embedding(x)

    @staticmethod
    def bi_interaction(e):
        summed = e.sum(dim=1)
        return 0.5 * (
            summed.square() - e.square().sum(dim=1)
        )


class FM(BaseCTR):
    def __init__(self, bias):
        super().__init__(bias)

    def forward(self, x):
        e = self.embedded(x)
        fm = self.bi_interaction(e).sum(dim=1)
        return self.global_bias + self.linear_score(x) + fm


class NFM(BaseCTR):
    def __init__(self, bias):
        super().__init__(bias)
        self.interaction_net = nn.Sequential(
            nn.BatchNorm1d(EMBED_DIM),
            nn.Linear(EMBED_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self._init_dense()

    def _init_dense(self):
        for module in self.interaction_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        e = self.embedded(x)
        bi = self.bi_interaction(e)
        nonlinear = self.interaction_net(bi).squeeze(1)
        return self.global_bias + self.linear_score(x) + nonlinear


class DeepFM(BaseCTR):
    def __init__(self, bias):
        super().__init__(bias)
        dim = N_FIELDS * EMBED_DIM
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self._init_dense()

    def _init_dense(self):
        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        e = self.embedded(x)
        fm = self.bi_interaction(e).sum(dim=1)
        deep = self.deep(e.flatten(start_dim=1)).squeeze(1)
        return self.global_bias + self.linear_score(x) + fm + deep


class WideDeep(BaseCTR):
    def __init__(self, bias):
        super().__init__(bias)
        dim = N_FIELDS * EMBED_DIM
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self._init_dense()

    def _init_dense(self):
        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        e = self.embedded(x)
        deep = self.deep(e.flatten(start_dim=1)).squeeze(1)
        return self.global_bias + self.linear_score(x) + deep


MODEL_BUILDERS = {
    "expanded_fm": lambda bias: FM(bias),
    "nfm": lambda bias: NFM(bias),
    "deepfm": lambda bias: DeepFM(bias),
    "wide_deep": lambda bias: WideDeep(bias),
}


@torch.no_grad()
def predict(model, x_np):
    model.eval()
    x_tensor = torch.from_numpy(x_np)
    out = np.empty(len(x_np), dtype=np.float64)
    for start in range(0, len(x_np), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(x_np))
        out[start:end] = (
            model(x_tensor[start:end])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
    return out


def build_model(name, y, seed):
    torch.manual_seed(seed)
    return MODEL_BUILDERS[name](initial_logit(y))


def fit_with_validation(name, x_train, y_train, x_valid, y_valid, users):
    family_index = list(MODEL_BUILDERS).index(name)
    seed = SEED + 1009 * family_index
    model = build_model(name, y_train, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(
        np.ascontiguousarray(y_train, dtype=np.float32)
    )
    generator = torch.Generator()
    generator.manual_seed(seed + 37)

    n = len(y_train)
    best_primary = -np.inf
    best_scores = None
    best_epoch = 1

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        total_loss = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = xt.index_select(0, idx)
            yb = yt.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * len(idx)

        scores = predict(model, x_valid)
        metrics = evaluate(users, y_valid, scores)
        primary = float(metrics["primary"])

        print(
            "family=%s epoch=%d loss=%.6f primary=%.6f "
            "gauc=%.6f ndcg5=%.6f"
            % (
                name,
                epoch,
                total_loss / n,
                primary,
                float(metrics["gauc"]),
                float(metrics["ndcg@5"]),
            ),
            flush=True,
        )

        if primary > best_primary:
            best_primary = primary
            best_scores = scores.copy()
            best_epoch = epoch

    del optimizer, model, xt, yt
    gc.collect()
    return best_scores, best_epoch


def fit_fixed_epochs(name, x_train, y_train, epochs):
    family_index = list(MODEL_BUILDERS).index(name)
    seed = SEED + 1009 * family_index
    model = build_model(name, y_train, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(
        np.ascontiguousarray(y_train, dtype=np.float32)
    )
    generator = torch.Generator()
    generator.manual_seed(seed + 37)

    n = len(y_train)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        total_loss = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = xt.index_select(0, idx)
            yb = yt.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)

        print(
            "refit_family=%s epoch=%d loss=%.6f"
            % (name, epoch, total_loss / n),
            flush=True,
        )

    del optimizer, xt, yt
    gc.collect()
    return model


train = load("train")
valid = load("valid")

x_train = make_matrix(train)
x_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

artifact_dir = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifact_dir, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_valid_z = standardize(inc_valid)

family_scores = {}
family_epochs = {}

for family in MODEL_BUILDERS:
    scores, epoch = fit_with_validation(
        family,
        x_train,
        y_train,
        x_valid,
        y_valid,
        valid_users,
    )
    family_scores[family] = np.asarray(scores, dtype=np.float64)
    family_epochs[family] = int(epoch)

candidate_values = {}
candidate_arrays = {}
candidate_recipes = {}

inc_metrics = evaluate(valid_users, y_valid, inc_valid)
candidate_values["incumbent"] = float(inc_metrics["primary"])
candidate_arrays["incumbent"] = inc_valid.copy()
candidate_recipes["incumbent"] = ("incumbent", 0.0)

blend_weights = np.linspace(0.0, 1.0, 21)

for family, raw in family_scores.items():
    raw_metrics = evaluate(valid_users, y_valid, raw)
    candidate_values[family] = float(raw_metrics["primary"])
    candidate_arrays[family] = raw.copy()
    candidate_recipes[family] = (family, 1.0)

    raw_z = standardize(raw)
    best_blend_value = -np.inf
    best_blend_array = None
    best_weight = 1.0

    for weight in blend_weights:
        blended = weight * raw_z + (1.0 - weight) * inc_valid_z
        metrics = evaluate(valid_users, y_valid, blended)
        value = float(metrics["primary"])
        if value > best_blend_value:
            best_blend_value = value
            best_blend_array = blended.copy()
            best_weight = float(weight)

    blend_name = "blend_" + family
    candidate_values[blend_name] = best_blend_value
    candidate_arrays[blend_name] = best_blend_array
    candidate_recipes[blend_name] = (family, best_weight)

winner = max(candidate_values, key=candidate_values.get)
winner_family, winner_weight = candidate_recipes[winner]
valid_scores = np.asarray(candidate_arrays[winner], dtype=np.float64)
valid_metrics = evaluate(valid_users, y_valid, valid_scores)

print(
    "FINDINGS best_epochs="
    + json.dumps(family_epochs, sort_keys=True)
    + " winner=%s family=%s model_weight=%.2f"
    % (winner, winner_family, winner_weight),
    flush=True,
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_values.items()},
        sort_keys=True,
    ),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores.astype(np.float64, copy=False),
    )

inc_test = np.asarray(
    np.load(os.path.join(artifact_dir, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

if winner_family == "incumbent":
    test_scores = inc_test.copy()
else:
    test = load("test")
    x_test = make_matrix(test)

    y_combined = np.ascontiguousarray(
        np.concatenate(
            [y_train, np.asarray(valid.y, dtype=np.float32)]
        ),
        dtype=np.float32,
    )
    x_combined = np.ascontiguousarray(
        np.concatenate([x_train, x_valid], axis=0),
        dtype=np.int64,
    )

    final_model = fit_fixed_epochs(
        winner_family,
        x_combined,
        y_combined,
        family_epochs[winner_family],
    )
    family_test = predict(final_model, x_test)

    if winner_weight >= 1.0 - 1e-12:
        test_scores = family_test
    elif winner_weight <= 1e-12:
        test_scores = inc_test.copy()
    else:
        test_scores = (
            winner_weight * standardize(family_test)
            + (1.0 - winner_weight) * standardize(inc_test)
        )

    del final_model
    gc.collect()

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(valid_metrics["primary"]),
            "gauc": float(valid_metrics["gauc"]),
            "ndcg@5": float(valid_metrics["ndcg@5"]),
            "gpu_seconds": elapsed,
        }
    ),
    flush=True,
)