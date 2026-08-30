import os
import time
import json
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7429
THREADS = max(1, min(8, os.cpu_count() or 1))
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

GBDT_CAT_FIELDS = [
    "video_id", "author_id", "tab", "tag", "duration_bucket",
    "upload_type", "music_type", "onehot_feat3", "onehot_feat8",
    "onehot_feat1", "onehot_feat7", "user_active_degree",
    "register_days_bucket", "register_days_range",
    "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "hour", "is_live_streamer",
    "is_video_author", "video_type",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]

PLE_CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type",
    "onehot_feat3", "onehot_feat8", "onehot_feat1",
    "onehot_feat7", "user_active_degree",
    "register_days_bucket", "hour",
]
AUX_KEYS = [
    "is_click", "is_like", "is_follow",
    "is_comment", "is_forward", "is_profile_enter",
]

EMBED_DIM = 8
BATCH_SIZE = 8192
PLE_EPOCHS = 2
AUX_WEIGHT = 0.16
SVD_RANK = 24
BLEND_ALPHAS = [0.10, 0.20, 0.30, 0.40, 0.50]


def concatenate_field(parts, source, name, dtype):
    if len(parts) == 1:
        return np.asarray(getattr(parts[0], source)[name], dtype=dtype)
    return np.concatenate([
        np.asarray(getattr(p, source)[name], dtype=dtype) for p in parts
    ])


def make_gbdt_matrix(parts):
    cols = []
    for f in GBDT_CAT_FIELDS:
        cols.append(concatenate_field(parts, "X", f, np.int32))

    for f in NUM_FIELDS:
        x = concatenate_field(parts, "num", f, np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        cols.append(x)

    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def fit_gbdt(X, y, num_rounds=240):
    dataset = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        categorical_feature=list(range(len(GBDT_CAT_FIELDS))),
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 300,
        "min_sum_hessian_in_leaf": 3.0,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.84,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 2.0,
        "max_cat_threshold": 64,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "max_bin": 127,
        "num_threads": THREADS,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=num_rounds)


def aggregate_sparse(rows, cols, labels, n_rows, n_cols):
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.float32)

    sums = sparse.coo_matrix(
        (labels, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float32
    ).tocsr()
    counts = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_rows, n_cols),
        dtype=np.float32,
    ).tocsr()
    sums.sum_duplicates()
    counts.sum_duplicates()

    if (
        sums.nnz != counts.nnz
        or not np.array_equal(sums.indptr, counts.indptr)
        or not np.array_equal(sums.indices, counts.indices)
    ):
        raise RuntimeError("Sparse aggregation patterns do not match")

    sums.data /= np.maximum(counts.data, 1.0)
    return sums


def fit_svd_entity(user_ids, entity_ids, labels, entity_cardinality, rank=SVD_RANK):
    user_cardinality = int(FEATURE_CARDINALITIES["user_id"])
    global_rate = float(np.mean(labels))
    centered = np.asarray(labels, dtype=np.float32) - np.float32(global_rate)

    mat = aggregate_sparse(
        user_ids, entity_ids, centered,
        user_cardinality, int(entity_cardinality)
    )

    k = min(rank, min(mat.shape) - 1)
    try:
        u, s, vt = svds(
            mat.astype(np.float32),
            k=k,
            which="LM",
            return_singular_vectors=True,
            random_state=SEED,
            maxiter=500,
            tol=2e-3,
        )
    except TypeError:
        np.random.seed(SEED)
        u, s, vt = svds(
            mat.astype(np.float32),
            k=k,
            which="LM",
            return_singular_vectors=True,
            maxiter=500,
            tol=2e-3,
        )

    order = np.argsort(s)[::-1]
    s = s[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order, :].astype(np.float32)
    user_factors = np.ascontiguousarray(u * s[None, :], dtype=np.float32)
    return user_factors, np.ascontiguousarray(vt, dtype=np.float32)


def score_svd(user_factors, entity_factors, users, entities):
    users = np.asarray(users, dtype=np.int64)
    entities = np.asarray(entities, dtype=np.int64)
    return np.einsum(
        "ij,ij->i",
        user_factors[users],
        entity_factors[:, entities].T,
        optimize=True,
    ).astype(np.float64)


ple_cards = [int(FEATURE_CARDINALITIES[f]) for f in PLE_CAT_FIELDS]
ple_offsets = np.cumsum([0] + ple_cards[:-1], dtype=np.int64)
ple_total_cardinality = int(sum(ple_cards))
ple_n_fields = len(PLE_CAT_FIELDS)
n_tasks = 1 + len(AUX_KEYS)


def make_ple_cat(parts):
    cols = []
    for f, off in zip(PLE_CAT_FIELDS, ple_offsets):
        x = concatenate_field(parts, "X", f, np.int64)
        cols.append(x + off)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


def numeric_stats(parts):
    centers = []
    scales = []
    for f in NUM_FIELDS:
        x = concatenate_field(parts, "num", f, np.float32)
        x = np.log1p(np.maximum(
            np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0
        ))
        med = float(np.median(x))
        q25, q75 = np.percentile(x, [25.0, 75.0])
        centers.append(med)
        scales.append(max(float(q75 - q25), 0.25))
    return np.asarray(centers, np.float32), np.asarray(scales, np.float32)


def make_ple_num(parts, centers, scales):
    cols = []
    for j, f in enumerate(NUM_FIELDS):
        x = concatenate_field(parts, "num", f, np.float32)
        x = np.log1p(np.maximum(
            np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0
        ))
        x = np.clip((x - centers[j]) / scales[j], -6.0, 6.0)
        cols.append(x.astype(np.float32))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def training_targets(split):
    main = np.asarray(split.y, dtype=np.float32)
    auxiliary = []
    for key in AUX_KEYS:
        x = np.asarray(split.aux[key], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        auxiliary.append((x > 0).astype(np.float32))
    return np.ascontiguousarray(
        np.column_stack([main] + auxiliary), dtype=np.float32
    )


class PLEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(ple_total_cardinality, EMBED_DIM)
        self.wide = nn.Embedding(ple_total_cardinality, 1)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.015)
        nn.init.zeros_(self.wide.weight)

        input_dim = ple_n_fields * EMBED_DIM + len(NUM_FIELDS)
        expert_dim = 56
        n_shared = 2
        n_specific = 2

        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 96),
                nn.ReLU(),
                nn.Linear(96, expert_dim),
                nn.ReLU(),
            )
            for _ in range(n_shared)
        ])
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 80),
                    nn.ReLU(),
                    nn.Linear(80, expert_dim),
                    nn.ReLU(),
                )
                for _ in range(n_specific)
            ])
            for _ in range(n_tasks)
        ])
        self.task_gates = nn.ModuleList([
            nn.Linear(input_dim, n_shared + n_specific)
            for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            for _ in range(n_tasks)
        ])
        self.main_bias = nn.Parameter(torch.zeros(1))

    def forward(self, cat, num):
        emb = self.embedding(cat).reshape(cat.shape[0], -1)
        x = torch.cat([emb, num], dim=1)
        wide = self.wide(cat).squeeze(-1).sum(dim=1)

        shared = [expert(x) for expert in self.shared_experts]
        outputs = []
        for task in range(n_tasks):
            specific = [expert(x) for expert in self.task_experts[task]]
            expert_stack = torch.stack(shared + specific, dim=1)
            gate = torch.softmax(self.task_gates[task](x), dim=1)
            representation = torch.sum(
                expert_stack * gate[:, :, None], dim=1
            )
            outputs.append(self.towers[task](representation))

        logits = torch.cat(outputs, dim=1)
        main = logits[:, 0] + wide + self.main_bias
        return torch.cat([main[:, None], logits[:, 1:]], dim=1)


def auxiliary_positive_weights(targets):
    result = np.ones(targets.shape[1], dtype=np.float32)
    for j in range(1, targets.shape[1]):
        p = float(np.mean(targets[:, j]))
        if 0.0 < p < 1.0:
            result[j] = np.float32(np.clip((1.0 - p) / p, 1.0, 8.0))
    return result


def fit_ple(cat, num, targets, aux_mask, seed):
    torch.manual_seed(seed)
    model = PLEModel()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0021, weight_decay=3e-6
    )

    cat_t = torch.from_numpy(cat)
    num_t = torch.from_numpy(num)
    target_t = torch.from_numpy(targets)
    mask_t = torch.from_numpy(np.asarray(aux_mask, dtype=np.float32))
    pos_weights = torch.from_numpy(auxiliary_positive_weights(targets))

    n = len(cat)
    for epoch in range(PLE_EPOCHS):
        generator = torch.Generator()
        generator.manual_seed(seed + 101 * epoch)
        order = torch.randperm(n, generator=generator)

        for st in range(0, n, BATCH_SIZE):
            idx = order[st:st + BATCH_SIZE]
            logits = model(cat_t[idx], num_t[idx])

            main_loss = nn.functional.binary_cross_entropy_with_logits(
                logits[:, 0], target_t[idx, 0]
            )
            aux_loss_matrix = nn.functional.binary_cross_entropy_with_logits(
                logits[:, 1:],
                target_t[idx, 1:],
                reduction="none",
                pos_weight=pos_weights[1:],
            )
            row_mask = mask_t[idx, None]
            denom = torch.clamp(
                row_mask.sum() * aux_loss_matrix.shape[1], min=1.0
            )
            aux_loss = (aux_loss_matrix * row_mask).sum() / denom
            loss = main_loss + AUX_WEIGHT * aux_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict_ple(model, cat, num):
    model.eval()
    cat_t = torch.from_numpy(cat)
    num_t = torch.from_numpy(num)
    pred = np.empty(len(cat), dtype=np.float64)
    with torch.no_grad():
        for st in range(0, len(cat), BATCH_SIZE * 2):
            en = min(st + BATCH_SIZE * 2, len(cat))
            pred[st:en] = model(
                cat_t[st:en], num_t[st:en]
            )[:, 0].cpu().numpy()
    return pred


def standardized_blend(anchor, raw, alpha, scale=None):
    if scale is None:
        raw_std = max(float(np.std(raw)), 1e-8)
        anchor_std = max(float(np.std(anchor)), 1e-8)
        scale = anchor_std / raw_std
    return (1.0 - alpha) * anchor + alpha * raw * scale, float(scale)


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

raw_predictions = {}

# Family 1: stationary side-feature pointwise gradient boosting.
X_train_gbdt = make_gbdt_matrix([train])
X_valid_gbdt = make_gbdt_matrix([valid])
gbdt_model = fit_gbdt(X_train_gbdt, y_train)
raw_predictions["gbdt_stationary"] = gbdt_model.predict(
    X_valid_gbdt
).astype(np.float64)
del gbdt_model, X_train_gbdt, X_valid_gbdt
gc.collect()

# Family 2: supervised low-rank user-video and user-author affinities.
uv_u, uv_v = fit_svd_entity(
    train.user_id, train.video_id, y_train,
    FEATURE_CARDINALITIES["video_id"]
)
ua_u, ua_v = fit_svd_entity(
    train.user_id, train.X["author_id"], y_train,
    FEATURE_CARDINALITIES["author_id"]
)
svd_video_valid = score_svd(
    uv_u, uv_v, valid.user_id, valid.video_id
)
svd_author_valid = score_svd(
    ua_u, ua_v, valid.user_id, valid.X["author_id"]
)
raw_predictions["svd_video"] = svd_video_valid
raw_predictions["svd_video_author"] = (
    0.70 * svd_video_valid + 0.30 * svd_author_valid
)
del uv_u, uv_v, ua_u, ua_v, svd_video_valid, svd_author_valid
gc.collect()

# Family 3: PLE with task-specific experts trained from train-only outcomes.
center, scale = numeric_stats([train])
cat_train = make_ple_cat([train])
num_train = make_ple_num([train], center, scale)
cat_valid = make_ple_cat([valid])
num_valid = make_ple_num([valid], center, scale)
targets_train = training_targets(train)
mask_train = np.ones(len(y_train), dtype=np.float32)

ple_model = fit_ple(
    cat_train, num_train, targets_train, mask_train, SEED + 300
)
raw_predictions["ple_multitask"] = predict_ple(
    ple_model, cat_valid, num_valid
)
del ple_model, cat_train, num_train, cat_valid, num_valid
gc.collect()

inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_log = {"incumbent": float(inc_metrics["primary"])}

best_name = "incumbent"
best_scores = inc_valid.copy()
best_metrics = inc_metrics
best_spec = {
    "family": "incumbent",
    "alpha": 0.0,
    "scale": 1.0,
}
best_raw = None

for family, raw in raw_predictions.items():
    metrics = evaluate(valid.user_id, y_valid, raw)
    candidate_log[family] = float(metrics["primary"])

    if metrics["primary"] > best_metrics["primary"]:
        best_name = family
        best_scores = raw.copy()
        best_metrics = metrics
        best_spec = {"family": family, "alpha": 1.0, "scale": 1.0}
        best_raw = raw.copy()

    for alpha in BLEND_ALPHAS:
        blended, score_scale = standardized_blend(
            inc_valid, raw, alpha
        )
        metrics = evaluate(valid.user_id, y_valid, blended)
        name = "%s_blend_%.2f" % (family, alpha)
        candidate_log[name] = float(metrics["primary"])

        if metrics["primary"] > best_metrics["primary"]:
            best_name = name
            best_scores = blended.copy()
            best_metrics = metrics
            best_spec = {
                "family": family,
                "alpha": float(alpha),
                "scale": float(score_scale),
            }
            best_raw = raw.copy()

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True), flush=True)
print(
    "FINDINGS structurally_distinct=true "
    "gbdt=%.6f svd_video=%.6f svd_combined=%.6f ple=%.6f selected=%s"
    % (
        candidate_log["gbdt_stationary"],
        candidate_log["svd_video"],
        candidate_log["svd_video_author"],
        candidate_log["ple_multitask"],
        best_name,
    ),
    flush=True,
)

# Refit the selected identical recipe on train + validation, then score test.
test = load("test")
selected_family = best_spec["family"]
selected_alpha = best_spec["alpha"]
selected_scale = best_spec["scale"]

if selected_family == "incumbent":
    test_scores = inc_test.copy()

elif selected_family == "gbdt_stationary":
    y_tv = np.concatenate([
        y_train, y_valid.astype(np.float32)
    ])
    X_tv = make_gbdt_matrix([train, valid])
    X_test = make_gbdt_matrix([test])
    model_tv = fit_gbdt(X_tv, y_tv)
    test_raw = model_tv.predict(X_test).astype(np.float64)

    if selected_alpha >= 1.0:
        test_scores = test_raw
    else:
        test_scores, _ = standardized_blend(
            inc_test, test_raw, selected_alpha, selected_scale
        )

elif selected_family in ("svd_video", "svd_video_author"):
    y_tv = np.concatenate([
        y_train, y_valid.astype(np.float32)
    ])
    user_tv = np.concatenate([
        np.asarray(train.user_id), np.asarray(valid.user_id)
    ])
    video_tv = np.concatenate([
        np.asarray(train.video_id), np.asarray(valid.video_id)
    ])

    uv_u, uv_v = fit_svd_entity(
        user_tv, video_tv, y_tv,
        FEATURE_CARDINALITIES["video_id"]
    )
    test_video = score_svd(
        uv_u, uv_v, test.user_id, test.video_id
    )

    if selected_family == "svd_video_author":
        author_tv = np.concatenate([
            np.asarray(train.X["author_id"]),
            np.asarray(valid.X["author_id"]),
        ])
        ua_u, ua_v = fit_svd_entity(
            user_tv, author_tv, y_tv,
            FEATURE_CARDINALITIES["author_id"]
        )
        test_author = score_svd(
            ua_u, ua_v, test.user_id, test.X["author_id"]
        )
        test_raw = 0.70 * test_video + 0.30 * test_author
    else:
        test_raw = test_video

    if selected_alpha >= 1.0:
        test_scores = test_raw
    else:
        test_scores, _ = standardized_blend(
            inc_test, test_raw, selected_alpha, selected_scale
        )

elif selected_family == "ple_multitask":
    y_tv = np.concatenate([
        y_train, y_valid.astype(np.float32)
    ])
    n_train = len(y_train)
    n_valid = len(y_valid)

    center_tv, scale_tv = numeric_stats([train, valid])
    cat_tv = make_ple_cat([train, valid])
    num_tv = make_ple_num([train, valid], center_tv, scale_tv)
    cat_test = make_ple_cat([test])
    num_test = make_ple_num([test], center_tv, scale_tv)

    targets_tv = np.zeros(
        (n_train + n_valid, n_tasks), dtype=np.float32
    )
    targets_tv[:n_train] = targets_train
    targets_tv[:, 0] = y_tv
    mask_tv = np.concatenate([
        np.ones(n_train, dtype=np.float32),
        np.zeros(n_valid, dtype=np.float32),
    ])

    model_tv = fit_ple(
        cat_tv, num_tv, targets_tv, mask_tv, SEED + 300
    )
    test_raw = predict_ple(model_tv, cat_test, num_test)

    if selected_alpha >= 1.0:
        test_scores = test_raw
    else:
        test_scores, _ = standardized_blend(
            inc_test, test_raw, selected_alpha, selected_scale
        )
else:
    raise RuntimeError("Unknown selected family: " + selected_family)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if best_raw is not None and selected_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }),
    flush=True,
)