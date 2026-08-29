import os
import time
import math
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
]
RANK = 12
SVD_RANK = 20
BATCH_SIZE = 8192
PRED_BATCH = 32768
MMOE_EPOCHS = 2

OFFSETS = {}
TOTAL_CARDINALITY = 0
for field in FIELDS:
    OFFSETS[field] = TOTAL_CARDINALITY
    TOTAL_CARDINALITY += int(FEATURE_CARDINALITIES[field])


def make_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[field], dtype=np.int64) + OFFSETS[field]
    return x


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def rank_transform(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_positions = np.where(starts, np.arange(n, dtype=np.int64), 0)
    start_positions = np.maximum.accumulate(start_positions)
    within = np.arange(n, dtype=np.int64) - start_positions

    ends = np.r_[np.flatnonzero(starts)[1:], n]
    group_sizes = np.repeat(ends - np.flatnonzero(starts), ends - np.flatnonzero(starts))

    ranked_sorted = np.where(
        group_sizes > 1,
        within.astype(np.float64) / (group_sizes.astype(np.float64) - 1.0),
        0.5,
    )
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


class EmpiricalBayesModel:
    def __init__(self):
        self.global_rate = 0.5
        self.tables = {}

    @staticmethod
    def _fit_single(keys, y):
        keys = np.asarray(keys, dtype=np.int64)
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        counts = np.bincount(inverse).astype(np.float64)
        positives = np.bincount(inverse, weights=y).astype(np.float64)
        return unique_keys, counts, positives

    def fit(self, split, y):
        y = np.asarray(y, dtype=np.float64)
        self.global_rate = float(np.mean(y))
        user = np.asarray(split.X["user_id"], dtype=np.int64)

        specs = {
            "video": (np.asarray(split.X["video_id"], dtype=np.int64), 35.0),
            "author": (np.asarray(split.X["author_id"], dtype=np.int64), 60.0),
            "tag": (np.asarray(split.X["tag"], dtype=np.int64), 80.0),
            "user_author": (
                user * int(FEATURE_CARDINALITIES["author_id"])
                + np.asarray(split.X["author_id"], dtype=np.int64),
                10.0,
            ),
            "user_tag": (
                user * int(FEATURE_CARDINALITIES["tag"])
                + np.asarray(split.X["tag"], dtype=np.int64),
                12.0,
            ),
            "user_tab": (
                user * int(FEATURE_CARDINALITIES["tab"])
                + np.asarray(split.X["tab"], dtype=np.int64),
                15.0,
            ),
        }

        for name, (keys, prior) in specs.items():
            unique_keys, counts, positives = self._fit_single(keys, y)
            rates = (
                positives + prior * self.global_rate
            ) / (counts + prior)
            self.tables[name] = (unique_keys, rates, prior)
        return self

    def _lookup(self, name, keys, fallback):
        unique_keys, rates, _ = self.tables[name]
        keys = np.asarray(keys, dtype=np.int64)
        positions = np.searchsorted(unique_keys, keys)
        found = positions < unique_keys.size
        clipped = np.minimum(positions, max(unique_keys.size - 1, 0))
        found &= unique_keys[clipped] == keys

        result = np.asarray(fallback, dtype=np.float64).copy()
        result[found] = rates[clipped[found]]
        return result

    def predict_components(self, split):
        user = np.asarray(split.X["user_id"], dtype=np.int64)
        video = np.asarray(split.X["video_id"], dtype=np.int64)
        author = np.asarray(split.X["author_id"], dtype=np.int64)
        tag = np.asarray(split.X["tag"], dtype=np.int64)
        tab = np.asarray(split.X["tab"], dtype=np.int64)

        base = np.full(video.size, self.global_rate, dtype=np.float64)
        video_rate = self._lookup("video", video, base)
        author_rate = self._lookup("author", author, base)
        tag_rate = self._lookup("tag", tag, base)

        ua_key = user * int(FEATURE_CARDINALITIES["author_id"]) + author
        ut_key = user * int(FEATURE_CARDINALITIES["tag"]) + tag
        ub_key = user * int(FEATURE_CARDINALITIES["tab"]) + tab

        ua_rate = self._lookup("user_author", ua_key, author_rate)
        ut_rate = self._lookup("user_tag", ut_key, tag_rate)
        ub_rate = self._lookup("user_tab", ub_key, base)

        global_content = (
            0.50 * safe_logit(video_rate)
            + 0.35 * safe_logit(author_rate)
            + 0.15 * safe_logit(tag_rate)
        )
        personalized = (
            0.55 * safe_logit(ua_rate)
            + 0.30 * safe_logit(ut_rate)
            + 0.15 * safe_logit(ub_rate)
        )
        hybrid = (
            0.25 * safe_logit(video_rate)
            + 0.15 * safe_logit(author_rate)
            + 0.40 * safe_logit(ua_rate)
            + 0.15 * safe_logit(ut_rate)
            + 0.05 * safe_logit(ub_rate)
        )
        return {
            "eb_global_content": global_content,
            "eb_personalized": personalized,
            "eb_hybrid": hybrid,
        }


class LatentSVDModel:
    def __init__(self, rank=SVD_RANK):
        self.rank = rank
        self.user_factors = None
        self.item_factors = None
        self.global_rate = 0.5
        self.user_rate = None
        self.item_rate = None

    def fit(self, split, y):
        users = np.asarray(split.X["user_id"], dtype=np.int64)
        items = np.asarray(split.X["video_id"], dtype=np.int64)
        y = np.asarray(y, dtype=np.float64)

        n_users = int(FEATURE_CARDINALITIES["user_id"])
        n_items = int(FEATURE_CARDINALITIES["video_id"])
        self.global_rate = float(np.mean(y))

        user_count = np.bincount(users, minlength=n_users).astype(np.float64)
        user_pos = np.bincount(
            users, weights=y, minlength=n_users
        ).astype(np.float64)
        item_count = np.bincount(items, minlength=n_items).astype(np.float64)
        item_pos = np.bincount(
            items, weights=y, minlength=n_items
        ).astype(np.float64)

        self.user_rate = (
            user_pos + 20.0 * self.global_rate
        ) / (user_count + 20.0)
        self.item_rate = (
            item_pos + 30.0 * self.global_rate
        ) / (item_count + 30.0)

        pair_code = users * n_items + items
        unique_pair, inverse = np.unique(pair_code, return_inverse=True)
        pair_count = np.bincount(inverse).astype(np.float64)
        pair_pos = np.bincount(inverse, weights=y).astype(np.float64)

        pair_users = unique_pair // n_items
        pair_items = unique_pair % n_items
        pair_mean = pair_pos / pair_count
        residual = (
            pair_mean
            - self.user_rate[pair_users]
            - self.item_rate[pair_items]
            + self.global_rate
        )

        matrix = sparse.csr_matrix(
            (residual.astype(np.float32), (pair_users, pair_items)),
            shape=(n_users, n_items),
        )

        k = min(self.rank, min(matrix.shape) - 1)
        u, s, vt = svds(
            matrix,
            k=k,
            which="LM",
            return_singular_vectors=True,
            random_state=SEED,
        )
        order = np.argsort(s)[::-1]
        s = s[order]
        u = u[:, order]
        vt = vt[order]

        root_s = np.sqrt(np.maximum(s, 0.0))
        self.user_factors = (u * root_s[None, :]).astype(np.float32)
        self.item_factors = (vt.T * root_s[None, :]).astype(np.float32)
        return self

    def predict(self, split):
        users = np.asarray(split.X["user_id"], dtype=np.int64)
        items = np.asarray(split.X["video_id"], dtype=np.int64)
        interaction = np.sum(
            self.user_factors[users] * self.item_factors[items], axis=1
        )
        additive = (
            self.user_rate[users]
            + self.item_rate[items]
            - self.global_rate
        )
        return safe_logit(additive) + 2.0 * interaction.astype(np.float64)


class MMoEModel(nn.Module):
    def __init__(self, intercepts):
        super().__init__()
        self.embedding = nn.Embedding(
            TOTAL_CARDINALITY, RANK, sparse=True
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)

        dim = len(FIELDS) * RANK
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            for _ in range(4)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(dim, 4) for _ in range(3)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(32, 1) for _ in range(3)
        ])
        self.register_buffer(
            "intercepts",
            torch.tensor(intercepts, dtype=torch.float32),
        )

    def forward(self, x):
        flat = self.embedding(x).reshape(x.shape[0], -1)
        expert_values = torch.stack(
            [expert(flat) for expert in self.experts], dim=1
        )

        outputs = []
        for task in range(3):
            gate = torch.softmax(self.gates[task](flat), dim=1)
            representation = torch.sum(
                expert_values * gate.unsqueeze(2), dim=1
            )
            output = self.heads[task](representation).squeeze(1)
            outputs.append(output + self.intercepts[task])
        return torch.stack(outputs, dim=1)


def intercept_for(y):
    p = float(np.mean(y))
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def get_multitask_targets(split, long_y):
    click = (
        np.asarray(split.aux["is_click"]).reshape(-1) > 0
    ).astype(np.float32)
    like = (
        np.asarray(split.aux["is_like"]).reshape(-1) > 0
    ).astype(np.float32)
    return np.column_stack([
        np.asarray(long_y, dtype=np.float32),
        click,
        like,
    ]).astype(np.float32)


def fit_mmoe(split, long_y, seed):
    torch.manual_seed(seed)
    x_np = make_matrix(split)
    targets_np = get_multitask_targets(split, long_y)
    intercepts = [
        intercept_for(targets_np[:, i]) for i in range(3)
    ]
    model = MMoEModel(intercepts)

    sparse_optimizer = torch.optim.SparseAdam(
        model.embedding.parameters(), lr=0.0015
    )
    dense_parameters = [
        p for name, p in model.named_parameters()
        if not name.startswith("embedding.")
    ]
    dense_optimizer = torch.optim.AdamW(
        dense_parameters, lr=0.0012, weight_decay=1e-5
    )

    x = torch.from_numpy(np.ascontiguousarray(x_np))
    targets = torch.from_numpy(targets_np)
    n = x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(seed + 101)

    model.train()
    task_weights = torch.tensor(
        [1.0, 0.30, 0.20], dtype=torch.float32
    )

    for _ in range(MMOE_EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(x[idx])
            losses = F.binary_cross_entropy_with_logits(
                logits, targets[idx], reduction="none"
            )
            loss = torch.mean(
                torch.sum(losses * task_weights[None, :], dim=1)
            )

            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_parameters, 5.0)
            sparse_optimizer.step()
            dense_optimizer.step()

    return model


@torch.no_grad()
def predict_mmoe(model, split):
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(make_matrix(split)))
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], PRED_BATCH):
        end = min(start + PRED_BATCH, x.shape[0])
        result[start:end] = (
            model(x[start:end])[:, 0].cpu().numpy().astype(np.float64)
        )
    return result


class JoinedSplit:
    pass


def join_splits(a, b):
    result = JoinedSplit()
    result.X = {
        field: np.concatenate([
            np.asarray(a.X[field]),
            np.asarray(b.X[field]),
        ])
        for field in a.X
    }
    result.user_id = np.concatenate([
        np.asarray(a.user_id), np.asarray(b.user_id)
    ])
    result.video_id = np.concatenate([
        np.asarray(a.video_id), np.asarray(b.video_id)
    ])
    result.date = np.concatenate([
        np.asarray(a.date), np.asarray(b.date)
    ])
    result.time_ms = np.concatenate([
        np.asarray(a.time_ms), np.asarray(b.time_ms)
    ])
    result.num = {
        field: np.concatenate([
            np.asarray(a.num[field]),
            np.asarray(b.num[field]),
        ])
        for field in a.num
    }
    common_aux = set(a.aux.keys()).intersection(b.aux.keys())
    result.aux = {
        field: np.concatenate([
            np.asarray(a.aux[field]),
            np.asarray(b.aux[field]),
        ])
        for field in common_aux
    }
    return result


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

artifacts = os.environ.get("RUN_ARTIFACTS", "")
inc_valid_path = os.path.join(artifacts, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_rank = rank_transform(valid.user_id, inc_valid)

family_valid = {}

eb_model = EmpiricalBayesModel().fit(train, y_train)
family_valid.update(eb_model.predict_components(valid))

svd_model = LatentSVDModel().fit(train, y_train)
family_valid["latent_svd"] = svd_model.predict(valid)

mmoe_model = fit_mmoe(train, y_train, SEED + 700)
family_valid["mmoe_multitask"] = predict_mmoe(mmoe_model, valid)

candidate_scores = {}
inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metrics["primary"])

best_name = "trusted_incumbent"
best_family = "incumbent"
best_alpha = 0.0
best_component = None
best_scores = inc_valid.copy()
best_metrics = inc_metrics

blend_weights = [0.25, 0.50, 0.75]

for name, raw_scores in family_valid.items():
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    metrics = evaluate(valid.user_id, y_valid, raw_scores)
    candidate_scores[name] = float(metrics["primary"])

    if float(metrics["primary"]) > float(best_metrics["primary"]):
        best_name = name
        best_family = name
        best_alpha = 1.0
        best_component = name
        best_scores = raw_scores.copy()
        best_metrics = metrics

    family_rank = rank_transform(valid.user_id, raw_scores)
    for alpha in blend_weights:
        blended = alpha * family_rank + (1.0 - alpha) * inc_valid_rank
        blend_name = "%s_rankblend_%.2f" % (name, alpha)
        metrics = evaluate(valid.user_id, y_valid, blended)
        candidate_scores[blend_name] = float(metrics["primary"])

        if float(metrics["primary"]) > float(best_metrics["primary"]):
            best_name = blend_name
            best_family = name
            best_alpha = float(alpha)
            best_component = name
            best_scores = blended.copy()
            best_metrics = metrics

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

test = load("test")

if best_family == "incumbent":
    test_scores = np.asarray(np.load(inc_test_path), dtype=np.float64)
else:
    y_joint = np.concatenate([
        y_train,
        np.asarray(valid.y, dtype=np.float32),
    ])
    joint = join_splits(train, valid)

    if best_family.startswith("eb_"):
        final_model = EmpiricalBayesModel().fit(joint, y_joint)
        component_map = final_model.predict_components(test)
        new_test_raw = component_map[best_family]
    elif best_family == "latent_svd":
        final_model = LatentSVDModel().fit(joint, y_joint)
        new_test_raw = final_model.predict(test)
    elif best_family == "mmoe_multitask":
        final_model = fit_mmoe(joint, y_joint, SEED + 700)
        new_test_raw = predict_mmoe(final_model, test)
    else:
        raise ValueError("Unknown selected family: " + best_family)

    if best_alpha < 1.0:
        inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
        new_test_rank = rank_transform(test.user_id, new_test_raw)
        inc_test_rank = rank_transform(test.user_id, inc_test)
        test_scores = (
            best_alpha * new_test_rank
            + (1.0 - best_alpha) * inc_test_rank
        )
    else:
        test_scores = np.asarray(new_test_raw, dtype=np.float64)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s family=%s incumbent_rankblend_new_weight=%.2f "
    "families=empirical_bayes,latent_svd,multitask_mmoe"
    % (best_name, best_family, best_alpha)
)

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.3f}'
    % (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)