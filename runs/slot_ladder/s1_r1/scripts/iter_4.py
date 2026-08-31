import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
THREADS = min(8, os.cpu_count() or 1)
BATCH_SIZE = 4096
EPOCHS = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

CAT_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
n_train = len(train.user_id)

dates = np.asarray(train.date, dtype=np.int64)
age = dates.max() - dates
w_train = np.power(0.5, age.astype(np.float32) / 4.0).astype(np.float32)
w_train /= w_train.mean()

cards = [int(FEATURE_CARDINALITIES[name]) for name in CAT_FIELDS]
offsets = np.zeros(len(cards), dtype=np.int64)
offsets[1:] = np.cumsum(cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def make_categorical(split):
    result = np.empty((len(split.user_id), len(CAT_FIELDS)), dtype=np.int64)
    for j, name in enumerate(CAT_FIELDS):
        result[:, j] = np.asarray(split.X[name], dtype=np.int64) + offsets[j]
    return result


xcat_train = make_categorical(train)
xcat_valid = make_categorical(valid)
xcat_test = make_categorical(test)


def make_raw_numeric(split_name, split):
    columns = []
    for name in NUM_FIELDS:
        values = np.asarray(split.num[name], dtype=np.float32)
        if name != "user_register_days":
            values = np.log1p(np.maximum(values, 0.0))
        columns.append(values)

    for key in ["video_id", "author_id"]:
        hist = historical_features(split_name, key=key)
        for name in sorted(hist):
            columns.append(np.asarray(hist[name], dtype=np.float32))

    return np.column_stack(columns).astype(np.float32)


raw_train = make_raw_numeric("train", train)
raw_valid = make_raw_numeric("valid", valid)
raw_test = make_raw_numeric("test", test)

finite_train = np.where(np.isfinite(raw_train), raw_train, np.nan)
num_mean = np.nanmean(finite_train, axis=0).astype(np.float32)
num_mean = np.where(np.isfinite(num_mean), num_mean, 0.0).astype(np.float32)
num_std = np.nanstd(finite_train, axis=0).astype(np.float32)
num_std = np.where(
    np.isfinite(num_std) & (num_std > 1e-5), num_std, 1.0
).astype(np.float32)


def normalize_numeric(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.where(np.isfinite(values), values, num_mean[None, :])
    values = (values - num_mean[None, :]) / num_std[None, :]
    return np.clip(values, -8.0, 8.0).astype(np.float32)


xnum_train = normalize_numeric(raw_train)
xnum_valid = normalize_numeric(raw_valid)
xnum_test = normalize_numeric(raw_test)
n_num = xnum_train.shape[1]

del raw_train, raw_valid, raw_test, finite_train

base_rate = float(np.clip(np.average(y_train, weights=w_train), 1e-6, 1 - 1e-6))
base_logit = float(np.log(base_rate / (1.0 - base_rate)))

aux_click = np.asarray(train.aux["is_click"], dtype=np.float32)
aux_like = np.asarray(train.aux["is_like"], dtype=np.float32)


class XDeepFM(nn.Module):
    def __init__(self, rank=12, cin_sizes=(16, 16)):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, rank)
        self.linear = nn.Embedding(total_cardinality, 1)
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

        self.cin_layers = nn.ModuleList()
        previous_fields = len(CAT_FIELDS)
        for size in cin_sizes:
            self.cin_layers.append(
                nn.Conv1d(len(CAT_FIELDS) * previous_fields, size, kernel_size=1)
            )
            previous_fields = size

        deep_dim = len(CAT_FIELDS) * rank + n_num
        self.deep = nn.Sequential(
            nn.Linear(deep_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(sum(cin_sizes) + 64 + n_num, 1)
        self.bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))

    def forward(self, xcat, xnum):
        x0 = self.embedding(xcat)
        xk = x0
        cin_outputs = []

        for layer in self.cin_layers:
            outer = torch.einsum("bfd,bhd->bfhd", x0, xk)
            outer = outer.reshape(
                outer.shape[0], outer.shape[1] * outer.shape[2], outer.shape[3]
            )
            xk = F.relu(layer(outer)).transpose(1, 2)
            cin_outputs.append(xk.sum(dim=2))

        deep_input = torch.cat([x0.flatten(1), xnum], dim=1)
        deep_output = self.deep(deep_input)
        cin_output = torch.cat(cin_outputs, dim=1)
        linear = self.linear(xcat).sum(dim=1).squeeze(1)

        combined = torch.cat([cin_output, deep_output, xnum], dim=1)
        return self.bias + linear + self.output(combined).squeeze(1)


class ProductNeuralNetwork(nn.Module):
    def __init__(self, rank=14):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, rank)
        self.linear = nn.Embedding(total_cardinality, 1)
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

        n_pairs = len(CAT_FIELDS) * (len(CAT_FIELDS) - 1) // 2
        input_dim = len(CAT_FIELDS) * rank + n_pairs + n_num
        self.network = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.ReLU(),
            nn.Dropout(0.06),
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Linear(80, 1),
        )
        self.bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))

    def forward(self, xcat, xnum):
        embedding = self.embedding(xcat)
        products = []
        for i in range(len(CAT_FIELDS)):
            for j in range(i + 1, len(CAT_FIELDS)):
                products.append((embedding[:, i] * embedding[:, j]).sum(1))
        products = torch.stack(products, dim=1)

        features = torch.cat(
            [embedding.flatten(1), products, xnum], dim=1
        )
        linear = self.linear(xcat).sum(dim=1).squeeze(1)
        return self.bias + linear + self.network(features).squeeze(1)


class FiBiNET(nn.Module):
    def __init__(self, rank=12):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, rank)
        self.linear = nn.Embedding(total_cardinality, 1)
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

        n_fields = len(CAT_FIELDS)
        self.squeeze = nn.Sequential(
            nn.Linear(n_fields, 8),
            nn.ReLU(),
            nn.Linear(8, n_fields),
            nn.Sigmoid(),
        )

        self.pair_transforms = nn.ModuleList()
        for _ in range(n_fields * (n_fields - 1) // 2):
            self.pair_transforms.append(nn.Linear(rank, rank, bias=False))

        interaction_dim = (
            n_fields * (n_fields - 1) // 2
        ) * rank
        self.network = nn.Sequential(
            nn.Linear(interaction_dim + n_fields * rank + n_num, 144),
            nn.ReLU(),
            nn.Dropout(0.06),
            nn.Linear(144, 72),
            nn.ReLU(),
            nn.Linear(72, 1),
        )
        self.bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))

    def forward(self, xcat, xnum):
        embedding = self.embedding(xcat)
        squeeze_stat = embedding.mean(dim=2)
        field_weight = self.squeeze(squeeze_stat).unsqueeze(2)
        recalibrated = embedding * field_weight

        interactions = []
        pair_index = 0
        for i in range(len(CAT_FIELDS)):
            for j in range(i + 1, len(CAT_FIELDS)):
                transformed = self.pair_transforms[pair_index](
                    recalibrated[:, i]
                )
                interactions.append(transformed * recalibrated[:, j])
                pair_index += 1

        interaction_features = torch.cat(interactions, dim=1)
        features = torch.cat(
            [recalibrated.flatten(1), interaction_features, xnum], dim=1
        )
        linear = self.linear(xcat).sum(dim=1).squeeze(1)
        return self.bias + linear + self.network(features).squeeze(1)


class PLE(nn.Module):
    def __init__(self, rank=10, n_shared=3, n_specific=2):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, rank)
        nn.init.normal_(self.embedding.weight, std=0.025)

        input_dim = len(CAT_FIELDS) * rank + n_num
        expert_dim = 48
        self.n_shared = n_shared
        self.n_specific = n_specific

        def expert():
            return nn.Sequential(
                nn.Linear(input_dim, 96),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(96, expert_dim),
                nn.ReLU(),
            )

        self.shared_experts = nn.ModuleList([expert() for _ in range(n_shared)])
        self.specific_experts = nn.ModuleList(
            [
                nn.ModuleList([expert() for _ in range(n_specific)])
                for _ in range(3)
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Linear(input_dim, n_shared + n_specific)
                for _ in range(3)
            ]
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(expert_dim, 24),
                    nn.ReLU(),
                    nn.Linear(24, 1),
                )
                for _ in range(3)
            ]
        )
        self.long_bias = nn.Parameter(
            torch.tensor(base_logit, dtype=torch.float32)
        )

    def forward(self, xcat, xnum):
        features = torch.cat(
            [self.embedding(xcat).flatten(1), xnum], dim=1
        )
        shared = [expert(features) for expert in self.shared_experts]
        outputs = []

        for task in range(3):
            specific = [
                expert(features) for expert in self.specific_experts[task]
            ]
            expert_stack = torch.stack(shared + specific, dim=1)
            gate = torch.softmax(self.gates[task](features), dim=1).unsqueeze(2)
            representation = (expert_stack * gate).sum(dim=1)
            output = self.heads[task](representation).squeeze(1)
            if task == 0:
                output = output + self.long_bias
            outputs.append(output)

        return tuple(outputs)


def train_model(model, seed, multitask=False):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.2e-3, weight_decay=2e-6
    )

    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(n_train)
        running_loss = 0.0
        running_weight = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            cat = torch.from_numpy(xcat_train[idx])
            num = torch.from_numpy(xnum_train[idx])
            target = torch.from_numpy(y_train[idx])
            weight = torch.from_numpy(w_train[idx])

            optimizer.zero_grad(set_to_none=True)
            output = model(cat, num)

            if multitask:
                long_logit, click_logit, like_logit = output
                long_loss = F.binary_cross_entropy_with_logits(
                    long_logit, target, reduction="none"
                )
                click_target = torch.from_numpy(aux_click[idx])
                like_target = torch.from_numpy(aux_like[idx])
                click_loss = F.binary_cross_entropy_with_logits(
                    click_logit, click_target, reduction="none"
                )
                like_loss = F.binary_cross_entropy_with_logits(
                    like_logit, like_target, reduction="none"
                )
                row_loss = long_loss + 0.16 * click_loss + 0.10 * like_loss
            else:
                row_loss = F.binary_cross_entropy_with_logits(
                    output, target, reduction="none"
                )

            loss = (row_loss * weight).sum() / weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += float((row_loss.detach() * weight).sum())
            running_weight += float(weight.sum())

        print(
            "TRAIN family=%s epoch=%d loss=%.6f"
            % (
                model.__class__.__name__,
                epoch + 1,
                running_loss / running_weight,
            ),
            flush=True,
        )

    return model


def predict_model(model, xcat, xnum, multitask=False):
    scores = np.empty(xcat.shape[0], dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, xcat.shape[0], 16384):
            end = min(start + 16384, xcat.shape[0])
            output = model(
                torch.from_numpy(xcat[start:end]),
                torch.from_numpy(xnum[start:end]),
            )
            if multitask:
                output = output[0]
            scores[start:end] = output.cpu().numpy()
    return scores


def smoothed_entity_logit(train_ids, query_ids, cardinality, strength):
    train_ids = np.asarray(train_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)

    counts = np.bincount(
        train_ids, weights=w_train, minlength=cardinality
    ).astype(np.float64)
    positives = np.bincount(
        train_ids, weights=w_train * y_train, minlength=cardinality
    ).astype(np.float64)

    probability = (
        positives + strength * base_rate
    ) / (counts + strength)
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    logits = np.log(probability / (1.0 - probability))
    return logits[query_ids]


def empirical_bayes_scores(split):
    video_score = smoothed_entity_logit(
        train.X["video_id"],
        split.X["video_id"],
        FEATURE_CARDINALITIES["video_id"],
        24.0,
    )
    author_score = smoothed_entity_logit(
        train.X["author_id"],
        split.X["author_id"],
        FEATURE_CARDINALITIES["author_id"],
        45.0,
    )
    duration_score = smoothed_entity_logit(
        train.X["duration_bucket"],
        split.X["duration_bucket"],
        FEATURE_CARDINALITIES["duration_bucket"],
        120.0,
    )
    tab_score = smoothed_entity_logit(
        train.X["tab"],
        split.X["tab"],
        FEATURE_CARDINALITIES["tab"],
        180.0,
    )
    return (
        0.62 * video_score
        + 0.27 * author_score
        + 0.08 * duration_score
        + 0.03 * tab_score
    ).astype(np.float64)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    sorted_rank = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)
    denominator = np.maximum(np.repeat(lengths - 1, lengths), 1)
    percentile = sorted_rank.astype(np.float64) / denominator

    result = np.empty(n, dtype=np.float64)
    result[order] = percentile
    return result


shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

families = [
    ("xdeepfm", lambda: XDeepFM(rank=12, cin_sizes=(16, 16)), False),
    ("pnn", lambda: ProductNeuralNetwork(rank=14), False),
    ("fibinet", lambda: FiBiNET(rank=12), False),
    ("ple", lambda: PLE(rank=10, n_shared=3, n_specific=2), True),
]

predictions = {}
candidate_scores = {}

for family_index, (name, constructor, multitask) in enumerate(families):
    torch.manual_seed(SEED + 100 * family_index)
    model = constructor()
    model = train_model(
        model, SEED + 1000 * family_index, multitask=multitask
    )
    valid_scores = predict_model(
        model, xcat_valid, xnum_valid, multitask=multitask
    )
    test_scores = predict_model(
        model, xcat_test, xnum_test, multitask=multitask
    )
    predictions[name] = (valid_scores, test_scores)
    raw_metric = evaluate(valid.user_id, y_valid, valid_scores)
    candidate_scores[name + "_raw"] = float(raw_metric["primary"])
    del model

eb_valid = empirical_bayes_scores(valid)
eb_test = empirical_bayes_scores(test)
predictions["empirical_bayes"] = (eb_valid, eb_test)
eb_metric = evaluate(valid.user_id, y_valid, eb_valid)
candidate_scores["empirical_bayes_raw"] = float(eb_metric["primary"])

inc_metric = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metric["primary"])

blend_alphas = [0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
best_primary = -np.inf
best_metric = None
best_valid_scores = None
best_test_scores = None
best_raw_valid = None
best_name = None
best_alpha = None

for name, (own_valid, own_test) in predictions.items():
    own_valid_rank = within_user_rank(valid.user_id, own_valid)
    own_test_rank = within_user_rank(test.user_id, own_test)

    for alpha in blend_alphas:
        blend_valid = (
            (1.0 - alpha) * inc_valid_rank + alpha * own_valid_rank
        )
        metric = evaluate(valid.user_id, y_valid, blend_valid)
        key = "%s_blend_a%.2f" % (name, alpha)
        candidate_scores[key] = float(metric["primary"])

        if metric["primary"] > best_primary:
            best_primary = float(metric["primary"])
            best_metric = metric
            best_valid_scores = blend_valid.copy()
            best_test_scores = (
                (1.0 - alpha) * inc_test_rank + alpha * own_test_rank
            ).copy()
            best_raw_valid = own_valid.copy()
            best_name = name
            best_alpha = alpha

print(
    "FINDINGS selected_family=%s blend_alpha=%.2f raw_primary=%.6f incumbent_primary=%.6f"
    % (
        best_name,
        best_alpha,
        candidate_scores[best_name + "_raw"],
        candidate_scores["trusted_incumbent"],
    ),
    flush=True,
)

print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metric["primary"]),
            "gauc": float(best_metric["gauc"]),
            "ndcg@5": float(best_metric["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)