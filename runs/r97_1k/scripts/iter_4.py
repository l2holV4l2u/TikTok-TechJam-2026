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
SEED = 314159
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "onehot_feat3",
    "onehot_feat8",
    "upload_type",
    "duration_bucket",
    "user_active_degree",
]
NUM_FIELDS = [
    "duration_ms",
    "user_follow_user_num",
    "user_fans_user_num",
    "user_friend_user_num",
    "user_register_days",
]

CARDS = [int(FEATURE_CARDINALITIES[x]) for x in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1]).astype(np.int64)
TOTAL_CARD = int(sum(CARDS))
N_FIELDS = len(FIELDS)
N_NUM = len(NUM_FIELDS)

EMBED_DIM = 8
BATCH_SIZE = 32768
PRED_BATCH_SIZE = 131072
EPOCHS = 3
LR_SPARSE = 0.004
LR_DENSE = 0.002


def make_numeric_stats(split):
    means = []
    scales = []
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        z = np.log1p(np.maximum(np.nan_to_num(x, nan=0.0), 0.0))
        means.append(float(z.mean()))
        sd = float(z.std())
        scales.append(max(sd, 1e-3))
    return np.asarray(means, np.float32), np.asarray(scales, np.float32)


def categorical_batch(split, indices=None, start=None, end=None):
    columns = []
    for name, offset in zip(FIELDS, OFFSETS):
        if indices is None:
            x = np.asarray(split.X[name][start:end], dtype=np.int64)
        else:
            x = np.asarray(split.X[name][indices], dtype=np.int64)
        columns.append(torch.from_numpy(x) + int(offset))
    return torch.stack(columns, dim=1)


def numeric_batch(split, means, scales, indices=None, start=None, end=None):
    columns = []
    for j, name in enumerate(NUM_FIELDS):
        if indices is None:
            x = np.asarray(split.num[name][start:end], dtype=np.float32)
        else:
            x = np.asarray(split.num[name][indices], dtype=np.float32)
        x = np.log1p(np.maximum(np.nan_to_num(x, nan=0.0), 0.0))
        x = (x - means[j]) / scales[j]
        columns.append(torch.from_numpy(x))
    return torch.stack(columns, dim=1)


class FwFM(nn.Module):
    """Field-weighted FM: every field pair has a separately learned strength."""

    def __init__(self, initial_bias):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM + 1, sparse=True)
        self.pair_weight = nn.Parameter(torch.ones(N_FIELDS, N_FIELDS))
        self.numeric_linear = nn.Linear(N_NUM, 1, bias=False)
        self.intercept = nn.Parameter(torch.tensor(float(initial_bias)))
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.02)
            self.numeric_linear.weight.zero_()

        ii, jj = torch.triu_indices(N_FIELDS, N_FIELDS, offset=1)
        self.register_buffer("pair_i", ii)
        self.register_buffer("pair_j", jj)

    def forward(self, cat, num):
        all_e = self.embedding(cat)
        linear = all_e[:, :, 0].sum(dim=1)
        e = all_e[:, :, 1:]
        products = (e[:, self.pair_i, :] * e[:, self.pair_j, :]).sum(dim=2)
        weights = self.pair_weight[self.pair_i, self.pair_j]
        interaction = (products * weights).sum(dim=1)
        numeric = self.numeric_linear(num).squeeze(1)
        return self.intercept + linear + interaction + numeric

    def sparse_parameters(self):
        return [self.embedding.weight]


class ProductNetwork(nn.Module):
    """PNN: explicit pairwise inner products feed a nonlinear prediction tower."""

    def __init__(self, initial_bias):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM, sparse=True)
        ii, jj = torch.triu_indices(N_FIELDS, N_FIELDS, offset=1)
        self.register_buffer("pair_i", ii)
        self.register_buffer("pair_j", jj)
        input_dim = N_FIELDS * EMBED_DIM + len(ii) + N_NUM
        self.tower = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.ReLU(),
            nn.LayerNorm(160),
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Linear(80, 1),
        )
        self.intercept = nn.Parameter(torch.tensor(float(initial_bias)))
        with torch.no_grad():
            self.embedding.weight.normal_(0.0, 0.025)
            self.tower[-1].weight.mul_(0.1)
            self.tower[-1].bias.zero_()

    def forward(self, cat, num):
        e = self.embedding(cat)
        products = (e[:, self.pair_i, :] * e[:, self.pair_j, :]).sum(dim=2)
        z = torch.cat([e.flatten(1), products, num], dim=1)
        return self.intercept + self.tower(z).squeeze(1)

    def sparse_parameters(self):
        return [self.embedding.weight]


class DeepCrossNetwork(nn.Module):
    """DCN: bounded-depth explicit feature crosses plus a nonlinear tower."""

    def __init__(self, initial_bias):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM, sparse=True)
        dim = N_FIELDS * EMBED_DIM + N_NUM
        self.cross_w = nn.ParameterList(
            [nn.Parameter(torch.empty(dim)) for _ in range(3)]
        )
        self.cross_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in range(3)]
        )
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 64, 1)
        self.intercept = nn.Parameter(torch.tensor(float(initial_bias)))

        with torch.no_grad():
            self.embedding.weight.normal_(0.0, 0.025)
            for w in self.cross_w:
                w.normal_(0.0, 0.02)
            self.output.weight.mul_(0.1)
            self.output.bias.zero_()

    def forward(self, cat, num):
        e = self.embedding(cat).flatten(1)
        x0 = torch.cat([e, num], dim=1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = (cross * w).sum(dim=1, keepdim=True)
            cross = cross + x0 * scalar + b
        deep = self.deep(x0)
        return self.intercept + self.output(
            torch.cat([cross, deep], dim=1)
        ).squeeze(1)

    def sparse_parameters(self):
        return [self.embedding.weight]


def train_model(model, train, y_tensor, means, scales, seed_offset):
    sparse_params = list(model.sparse_parameters())
    sparse_ids = {id(p) for p in sparse_params}
    dense_params = [p for p in model.parameters() if id(p) not in sparse_ids]

    sparse_opt = torch.optim.SparseAdam(sparse_params, lr=LR_SPARSE)
    dense_opt = torch.optim.AdamW(
        dense_params, lr=LR_DENSE, weight_decay=2e-6
    )

    n = len(train.user_id)
    generator = torch.Generator()
    generator.manual_seed(SEED + seed_offset)
    model.train()

    for epoch in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        rows_seen = 0

        for begin in range(0, n, BATCH_SIZE):
            ids = permutation[begin:begin + BATCH_SIZE]
            ids_np = ids.numpy()
            cat = categorical_batch(train, indices=ids_np)
            num = numeric_batch(train, means, scales, indices=ids_np)
            target = y_tensor[ids]

            sparse_opt.zero_grad(set_to_none=True)
            dense_opt.zero_grad(set_to_none=True)
            logits = model(cat, num)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_params, 5.0)
            sparse_opt.step()
            dense_opt.step()

            count = int(ids.numel())
            loss_sum += float(loss.detach()) * count
            rows_seen += count

        print(
            "TRAIN family=%s epoch=%d logloss=%.6f"
            % (
                model.__class__.__name__,
                epoch + 1,
                loss_sum / max(rows_seen, 1),
            ),
            flush=True,
        )


@torch.inference_mode()
def predict(model, split, means, scales):
    model.eval()
    n = len(split.user_id)
    result = np.empty(n, dtype=np.float32)
    for begin in range(0, n, PRED_BATCH_SIZE):
        end = min(begin + PRED_BATCH_SIZE, n)
        cat = categorical_batch(split, start=begin, end=end)
        num = numeric_batch(split, means, scales, start=begin, end=end)
        result[begin:end] = model(cat, num).cpu().numpy()
    return result


train = load("train")
numeric_means, numeric_scales = make_numeric_stats(train)
train_y = torch.from_numpy(np.asarray(train.y, dtype=np.float32).copy())
positive_rate = float(train_y.mean())
initial_bias = float(np.log(positive_rate / (1.0 - positive_rate)))

models = {
    "fwfm": FwFM(initial_bias),
    "pnn": ProductNetwork(initial_bias),
    "dcn": DeepCrossNetwork(initial_bias),
}

for model_index, (name, model) in enumerate(models.items()):
    train_model(
        model,
        train,
        train_y,
        numeric_means,
        numeric_scales,
        seed_offset=1000 * (model_index + 1),
    )

del train_y
del train
gc.collect()

valid = load("valid")
valid_predictions = {}
standalone_metrics = {}

for name, model in models.items():
    scores = predict(model, valid, numeric_means, numeric_scales)
    valid_predictions[name] = scores
    standalone_metrics[name] = evaluate(valid.user_id, valid.y, scores)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
has_incumbent = os.path.exists(inc_valid_path) and os.path.exists(inc_test_path)

candidate_primary = {}
candidate_specs = []

for name in models:
    metric = standalone_metrics[name]
    candidate_primary[name] = float(metric["primary"])
    candidate_specs.append(
        {
            "candidate_name": name,
            "family": name,
            "weight": 1.0,
            "scores": valid_predictions[name],
            "metrics": metric,
        }
    )

if has_incumbent:
    incumbent_valid = np.asarray(
        np.load(inc_valid_path, mmap_mode="r"), dtype=np.float32
    )
    # The trusted-incumbent contract explicitly permits selecting these blend
    # weights on validation and applying the same fixed weight to test.
    for name in models:
        own = valid_predictions[name]
        for own_weight in (0.25, 0.50, 0.75):
            blend = (
                own_weight * own
                + (1.0 - own_weight) * incumbent_valid
            ).astype(np.float32)
            metric = evaluate(valid.user_id, valid.y, blend)
            candidate_name = "%s_blend_%.2f" % (name, own_weight)
            candidate_primary[candidate_name] = float(metric["primary"])
            candidate_specs.append(
                {
                    "candidate_name": candidate_name,
                    "family": name,
                    "weight": float(own_weight),
                    "scores": blend,
                    "metrics": metric,
                }
            )

winner = max(candidate_specs, key=lambda x: x["metrics"]["primary"])
winner_name = winner["candidate_name"]
winner_family = winner["family"]
winner_weight = float(winner["weight"])
valid_scores = np.asarray(winner["scores"], dtype=np.float32)
metrics = winner["metrics"]
own_winner_valid = valid_predictions[winner_family]

print(
    "FINDINGS winner=%s standalone_primary=%.6f blend_weight=%.2f"
    % (
        winner_name,
        standalone_metrics[winner_family]["primary"],
        winner_weight,
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_primary, sort_keys=True),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner_weight < 1.0:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(own_winner_valid, dtype=np.float64),
        )

del valid
del incumbent_valid if has_incumbent else models
# Keep only the selected model for test inference.
winner_model = models[winner_family]
for name in list(models.keys()):
    if name != winner_family:
        del models[name]
gc.collect()

test = load("test")
own_test_scores = predict(
    winner_model, test, numeric_means, numeric_scales
)

if winner_weight < 1.0:
    incumbent_test = np.asarray(
        np.load(inc_test_path, mmap_mode="r"), dtype=np.float32
    )
    test_scores = (
        winner_weight * own_test_scores
        + (1.0 - winner_weight) * incumbent_test
    ).astype(np.float32)
else:
    test_scores = own_test_scores

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
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": elapsed,
        }
    )
)