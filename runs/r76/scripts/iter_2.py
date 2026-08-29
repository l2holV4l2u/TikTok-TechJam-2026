import os
import time
import json
import copy
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 2026
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
MAX_EPOCHS = 7

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

CARDINALITIES = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.asarray(
    [0] + list(np.cumsum(CARDINALITIES[:-1], dtype=np.int64)),
    dtype=np.int64,
)
TOTAL_CARDINALITY = int(sum(CARDINALITIES))


def make_local_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([np.asarray(split.X[f], dtype=np.int64) for f in FIELDS]),
        dtype=np.int64,
    )


def offset_matrix(x):
    return np.ascontiguousarray(x + OFFSETS[None, :], dtype=np.int64)


def initial_logit(y):
    p = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
    return float(np.log(p / (1.0 - p)))


class WideAdditive(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.bias + self.linear(x).sum(dim=1).squeeze(1)


class FactorizationMachine(nn.Module):
    def __init__(self, bias, rank=16):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.latent = nn.Embedding(TOTAL_CARDINALITY, rank)
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        v = self.latent(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


class DeepFM(nn.Module):
    def __init__(self, bias, rank=12):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.latent = nn.Embedding(TOTAL_CARDINALITY, rank)
        dim = len(FIELDS) * rank
        self.mlp = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        v = self.latent(x)
        summed = v.sum(dim=1)
        fm = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.mlp(v.flatten(start_dim=1)).squeeze(1)
        return self.bias + linear + fm + deep


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x0, x):
        scalar = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scalar + self.bias + x


class DCN(nn.Module):
    def __init__(self, bias, rank=8):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, rank)
        dim = len(FIELDS) * rank
        self.cross1 = CrossLayer(dim)
        self.cross2 = CrossLayer(dim)
        self.deep = nn.Sequential(
            nn.Linear(dim, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 32, 1)
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x0 = self.embedding(x).flatten(start_dim=1)
        cross = self.cross1(x0, x0)
        cross = self.cross2(x0, cross)
        deep = self.deep(x0)
        return self.bias + self.output(torch.cat([cross, deep], dim=1)).squeeze(1)


MODEL_BUILDERS = {
    "wide_additive": lambda bias: WideAdditive(bias),
    "fm_expanded": lambda bias: FactorizationMachine(bias, rank=16),
    "deepfm": lambda bias: DeepFM(bias, rank=12),
    "dcn": lambda bias: DCN(bias, rank=8),
}


@torch.no_grad()
def predict(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    out = np.empty(len(x_np), dtype=np.float64)
    for start in range(0, len(x_np), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(x_np))
        out[start:end] = (
            model(x[start:end]).detach().cpu().numpy().astype(np.float64, copy=False)
        )
    return out


def fit_with_validation(
    name,
    x_train,
    y_train,
    x_valid,
    y_valid,
    valid_users,
):
    seed = SEED + list(MODEL_BUILDERS).index(name) * 101
    torch.manual_seed(seed)

    model = MODEL_BUILDERS[name](initial_logit(y_train))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(seed + 19)

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None
    best_state = None
    n = len(y_train)

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=generator)
        total_loss = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = xt.index_select(0, idx)
            yb = yt.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)

        scores = predict(model, x_valid)
        metrics = evaluate(valid_users, y_valid, scores)
        primary = float(metrics["primary"])
        print(
            "family=%s epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
            % (
                name,
                epoch + 1,
                total_loss / n,
                primary,
                float(metrics["gauc"]),
                float(metrics["ndcg@5"]),
            ),
            flush=True,
        )

        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch + 1
            best_scores = scores.copy()
            best_state = copy.deepcopy(model.state_dict())

    del model, optimizer, xt, yt
    gc.collect()
    return best_scores, best_epoch, best_state


def fit_fixed(name, x, y, epochs):
    seed = SEED + list(MODEL_BUILDERS).index(name) * 101
    torch.manual_seed(seed)
    model = MODEL_BUILDERS[name](initial_logit(y))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(seed + 19)
    n = len(y)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        total_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
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
            % (name, epoch + 1, total_loss / n),
            flush=True,
        )

    return model


def target_stat_scores(x_fit_local, y_fit, x_score_local, smoothing=25.0):
    prior = float(np.clip(np.mean(y_fit), 1e-5, 1.0 - 1e-5))
    result = np.zeros(len(x_score_local), dtype=np.float64)

    # Every family receives the same fields. The user contribution is constant
    # within a user and therefore does not directly alter within-user ranking,
    # while the remaining fields provide item/context empirical priors.
    for j, cardinality in enumerate(CARDINALITIES):
        ids = x_fit_local[:, j]
        counts = np.bincount(ids, minlength=cardinality).astype(np.float64)
        positives = np.bincount(
            ids,
            weights=y_fit.astype(np.float64, copy=False),
            minlength=cardinality,
        )
        rates = (positives + smoothing * prior) / (counts + smoothing)
        rates = np.clip(rates, 1e-5, 1.0 - 1e-5)
        logits = np.log(rates / (1.0 - rates))
        result += logits[x_score_local[:, j]]

    result /= float(len(FIELDS))
    return result


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    scale = float(np.std(x))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (x - float(np.mean(x))) / scale


train = load("train")
valid = load("valid")

x_train_local = make_local_matrix(train)
x_valid_local = make_local_matrix(valid)
x_train = offset_matrix(x_train_local)
x_valid = offset_matrix(x_valid_local)

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

artifact_dir = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifact_dir, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_valid_z = standardize(inc_valid)

family_valid = {}
family_epochs = {}

# Non-parametric family.
family_valid["empirical_bayes"] = target_stat_scores(
    x_train_local,
    y_train,
    x_valid_local,
)
family_epochs["empirical_bayes"] = 0

# Four genuinely different learned prediction forms.
for family in MODEL_BUILDERS:
    scores, best_epoch, _ = fit_with_validation(
        family,
        x_train,
        y_train,
        x_valid,
        y_valid,
        valid_users,
    )
    family_valid[family] = scores
    family_epochs[family] = best_epoch
    print(
        "selected_family_epoch %s %d" % (family, best_epoch),
        flush=True,
    )

candidate_scores = {}
candidate_arrays = {}
candidate_recipe = {}

inc_metric = evaluate(valid_users, y_valid, inc_valid)
candidate_scores["incumbent"] = float(inc_metric["primary"])
candidate_arrays["incumbent"] = inc_valid
candidate_recipe["incumbent"] = ("incumbent", 0.0)

weights = np.linspace(0.0, 1.0, 11)

for family, raw_scores in family_valid.items():
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    standalone_metrics = evaluate(valid_users, y_valid, raw_scores)
    candidate_scores[family] = float(standalone_metrics["primary"])
    candidate_arrays[family] = raw_scores
    candidate_recipe[family] = (family, 1.0)

    z = standardize(raw_scores)
    best_blend_primary = -np.inf
    best_blend = None
    best_weight = None

    for w in weights:
        blended = w * z + (1.0 - w) * inc_valid_z
        metrics = evaluate(valid_users, y_valid, blended)
        primary = float(metrics["primary"])
        if primary > best_blend_primary:
            best_blend_primary = primary
            best_blend = blended.copy()
            best_weight = float(w)

    key = "blend_" + family
    candidate_scores[key] = best_blend_primary
    candidate_arrays[key] = best_blend
    candidate_recipe[key] = (family, best_weight)

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = np.asarray(candidate_arrays[winner], dtype=np.float64)
winner_family, winner_weight = candidate_recipe[winner]
valid_metrics = evaluate(valid_users, y_valid, valid_scores)

print(
    "FINDINGS winner=%s family=%s candidate_weight=%.2f epoch=%d"
    % (
        winner,
        winner_family,
        winner_weight,
        family_epochs.get(winner_family, 0),
    ),
    flush=True,
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_scores.items()},
        sort_keys=True,
        separators=(", ", ": "),
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

# Create test scores using the selected recipe refit on train+validation.
inc_test = np.asarray(
    np.load(os.path.join(artifact_dir, "incumbent_test_scores.npy")),
    dtype=np.float64,
)
test = load("test")
x_test_local = make_local_matrix(test)

if winner_family == "incumbent":
    test_scores = inc_test.copy()
else:
    y_combined = np.ascontiguousarray(
        np.concatenate([y_train, np.asarray(valid.y, dtype=np.float32)]),
        dtype=np.float32,
    )
    x_combined_local = np.ascontiguousarray(
        np.concatenate([x_train_local, x_valid_local], axis=0),
        dtype=np.int64,
    )

    if winner_family == "empirical_bayes":
        candidate_test = target_stat_scores(
            x_combined_local,
            y_combined,
            x_test_local,
        )
    else:
        x_combined = offset_matrix(x_combined_local)
        x_test = offset_matrix(x_test_local)
        refit_model = fit_fixed(
            winner_family,
            x_combined,
            y_combined,
            family_epochs[winner_family],
        )
        candidate_test = predict(refit_model, x_test)
        del refit_model, x_combined, x_test
        gc.collect()

    if winner_weight >= 1.0 - 1e-12:
        test_scores = np.asarray(candidate_test, dtype=np.float64)
    elif winner_weight <= 1e-12:
        test_scores = inc_test.copy()
    else:
        test_scores = (
            winner_weight * standardize(candidate_test)
            + (1.0 - winner_weight) * standardize(inc_test)
        )

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
        },
        separators=(", ", ": "),
    )
)