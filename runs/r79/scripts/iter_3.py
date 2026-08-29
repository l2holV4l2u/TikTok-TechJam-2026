import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260829
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
EMBED_DIM = 12
N_EXPERTS = 4
BATCH_SIZE = 4096
MMOE_EPOCHS = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))

ART = os.environ["RUN_ARTIFACTS"]
OUT = os.environ.get("ITER_OUT")
if OUT:
    os.makedirs(OUT, exist_ok=True)

train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

inc_valid = np.asarray(
    np.load(os.path.join(ART, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test_path = os.path.join(ART, "incumbent_test_scores.npy")


def make_x(split):
    return np.ascontiguousarray(
        np.column_stack([np.asarray(split.X[f]) for f in FIELDS]),
        dtype=np.int64,
    )


x_train = make_x(train)
x_valid = make_x(valid)

CARDS = np.asarray(
    [int(FEATURE_CARDINALITIES[f]) for f in FIELDS],
    dtype=np.int64,
)
OFFSETS = np.cumsum(
    np.concatenate([np.zeros(1, dtype=np.int64), CARDS[:-1]])
)
TOTAL_CARD = int(CARDS.sum())


def within_user_rank(user_ids, scores):
    u = np.asarray(user_ids)
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if n == 0:
        return s.copy()

    order = np.lexsort((np.arange(n, dtype=np.int64), s, u))
    sorted_u = u[order]

    start_flag = np.empty(n, dtype=bool)
    start_flag[0] = True
    start_flag[1:] = sorted_u[1:] != sorted_u[:-1]
    starts = np.maximum.accumulate(
        np.where(start_flag, np.arange(n), 0)
    )

    end_flag = np.empty(n, dtype=bool)
    end_flag[-1] = True
    end_flag[:-1] = sorted_u[:-1] != sorted_u[1:]
    ends = np.minimum.accumulate(
        np.where(end_flag, np.arange(n), n - 1)[::-1]
    )[::-1]

    denom = ends - starts
    sorted_rank = np.full(n, 0.5, dtype=np.float64)
    mask = denom > 0
    sorted_rank[mask] = (
        np.arange(n, dtype=np.float64)[mask] - starts[mask]
    ) / denom[mask]

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_rank
    return result


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def dense_smoothed_rate(fit_ids, fit_y, pred_ids, card, prior, base_rate):
    count = np.bincount(fit_ids, minlength=card).astype(np.float64)
    pos = np.bincount(
        fit_ids, weights=fit_y, minlength=card
    ).astype(np.float64)
    rate = (pos + prior * base_rate) / (count + prior)
    return rate[pred_ids]


def sparse_pair_rate(
    fit_a,
    fit_b,
    fit_y,
    pred_a,
    pred_b,
    b_card,
    prior,
    base_rate,
):
    fit_key = (
        np.asarray(fit_a, dtype=np.int64) * np.int64(b_card)
        + np.asarray(fit_b, dtype=np.int64)
    )
    pred_key = (
        np.asarray(pred_a, dtype=np.int64) * np.int64(b_card)
        + np.asarray(pred_b, dtype=np.int64)
    )

    unique_key, inverse = np.unique(fit_key, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    pos = np.bincount(
        inverse, weights=np.asarray(fit_y, dtype=np.float64)
    ).astype(np.float64)
    rate = (pos + prior * base_rate) / (count + prior)

    location = np.searchsorted(unique_key, pred_key)
    found = location < len(unique_key)
    safe_location = np.minimum(location, len(unique_key) - 1)
    found &= unique_key[safe_location] == pred_key

    result = np.full(len(pred_key), base_rate, dtype=np.float64)
    result[found] = rate[safe_location[found]]
    return result


def empirical_bayes_predict(fit_x, fit_y, pred_x):
    base = float(np.mean(fit_y))
    user_col = 0
    video_col = 1
    author_col = 2
    duration_col = 4
    tag_col = 5
    upload_col = 6
    music_col = 7

    video_rate = dense_smoothed_rate(
        fit_x[:, video_col], fit_y, pred_x[:, video_col],
        int(CARDS[video_col]), 55.0, base
    )
    author_rate = dense_smoothed_rate(
        fit_x[:, author_col], fit_y, pred_x[:, author_col],
        int(CARDS[author_col]), 65.0, base
    )
    tag_rate = dense_smoothed_rate(
        fit_x[:, tag_col], fit_y, pred_x[:, tag_col],
        int(CARDS[tag_col]), 180.0, base
    )

    uv = sparse_pair_rate(
        fit_x[:, user_col], fit_x[:, video_col], fit_y,
        pred_x[:, user_col], pred_x[:, video_col],
        int(CARDS[video_col]), 7.0, base
    )
    ua = sparse_pair_rate(
        fit_x[:, user_col], fit_x[:, author_col], fit_y,
        pred_x[:, user_col], pred_x[:, author_col],
        int(CARDS[author_col]), 9.0, base
    )
    ut = sparse_pair_rate(
        fit_x[:, user_col], fit_x[:, tag_col], fit_y,
        pred_x[:, user_col], pred_x[:, tag_col],
        int(CARDS[tag_col]), 13.0, base
    )
    ud = sparse_pair_rate(
        fit_x[:, user_col], fit_x[:, duration_col], fit_y,
        pred_x[:, user_col], pred_x[:, duration_col],
        int(CARDS[duration_col]), 15.0, base
    )
    uu = sparse_pair_rate(
        fit_x[:, user_col], fit_x[:, upload_col], fit_y,
        pred_x[:, user_col], pred_x[:, upload_col],
        int(CARDS[upload_col]), 18.0, base
    )
    um = sparse_pair_rate(
        fit_x[:, user_col], fit_x[:, music_col], fit_y,
        pred_x[:, user_col], pred_x[:, music_col],
        int(CARDS[music_col]), 20.0, base
    )

    score = (
        0.75 * logit(video_rate)
        + 0.65 * logit(author_rate)
        + 0.30 * logit(tag_rate)
        + 0.65 * logit(uv)
        + 1.15 * logit(ua)
        + 0.80 * logit(ut)
        + 0.42 * logit(ud)
        + 0.32 * logit(uu)
        + 0.20 * logit(um)
    )
    return np.asarray(score, dtype=np.float64)


def fit_svd_predict(fit_x, fit_y, pred_x, rank=32):
    n_users = int(CARDS[0])
    n_videos = int(CARDS[1])
    base = float(np.mean(fit_y))

    # Centering observed outcomes removes much of the global exposure signal.
    values = np.asarray(fit_y, dtype=np.float64) - base
    matrix = sparse.coo_matrix(
        (
            values,
            (
                np.asarray(fit_x[:, 0], dtype=np.int64),
                np.asarray(fit_x[:, 1], dtype=np.int64),
            ),
        ),
        shape=(n_users, n_videos),
        dtype=np.float64,
    ).tocsr()

    u, singular, vt = randomized_svd(
        matrix,
        n_components=rank,
        n_iter=3,
        random_state=SEED,
    )
    user_factors = u * singular[None, :]
    pred = np.sum(
        user_factors[pred_x[:, 0]] * vt[:, pred_x[:, 1]].T,
        axis=1,
    )

    # A small item prior stabilizes candidates weakly represented by the SVD.
    item_rate = dense_smoothed_rate(
        fit_x[:, 1], fit_y, pred_x[:, 1], n_videos, 70.0, base
    )
    pred = pred + 0.18 * logit(item_rate)
    return np.asarray(pred, dtype=np.float64)


def auxiliary_targets(split, long_target):
    available = set(split.aux.keys())
    preferred = ["is_click", "is_like", "is_follow"]
    chosen = [name for name in preferred if name in available][:2]
    targets = [np.asarray(long_target, dtype=np.float32)]
    for name in chosen:
        targets.append(np.asarray(split.aux[name], dtype=np.float32))
    return np.ascontiguousarray(np.column_stack(targets), dtype=np.float32), chosen


class MMoE(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        input_dim = len(FIELDS) * EMBED_DIM

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 96),
                nn.ReLU(),
                nn.Linear(96, 48),
                nn.ReLU(),
            )
            for _ in range(N_EXPERTS)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, N_EXPERTS)
            for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(n_tasks)
        ])
        self.linear = nn.Embedding(TOTAL_CARD, n_tasks)
        self.bias = nn.Parameter(torch.zeros(n_tasks))
        self.register_buffer(
            "offsets", torch.from_numpy(OFFSETS.copy()).long()
        )

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        ids = x + self.offsets
        emb = self.embedding(ids).reshape(x.shape[0], -1)
        expert_stack = torch.stack(
            [expert(emb) for expert in self.experts],
            dim=1,
        )
        outputs = []
        linear = self.linear(ids).sum(dim=1) + self.bias
        for task_index, (gate, tower) in enumerate(
            zip(self.gates, self.towers)
        ):
            weights = torch.softmax(gate(emb), dim=1).unsqueeze(-1)
            mixed = torch.sum(weights * expert_stack, dim=1)
            outputs.append(
                tower(mixed).squeeze(1) + linear[:, task_index]
            )
        return torch.stack(outputs, dim=1)


def predict_mmoe(model, x_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float64)
    x_tensor = torch.from_numpy(x_np)
    with torch.no_grad():
        for start in range(0, len(x_np), 65536):
            end = min(start + 65536, len(x_np))
            logits = model(x_tensor[start:end])[:, 0]
            result[start:end] = logits.cpu().numpy()
    return result


def fit_mmoe(
    fit_x,
    target_matrix,
    epochs,
    valid_x=None,
    valid_user=None,
    valid_y=None,
):
    torch.manual_seed(SEED + 401)
    model = MMoE(target_matrix.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0012,
        weight_decay=2e-6,
    )

    x_tensor = torch.from_numpy(fit_x)
    target_tensor = torch.from_numpy(target_matrix)
    generator = torch.Generator().manual_seed(SEED + 403)

    task_weights = torch.ones(target_matrix.shape[1], dtype=torch.float32)
    if target_matrix.shape[1] > 1:
        task_weights[1:] = 0.45

    best_primary = -np.inf
    best_epoch = epochs
    best_state = None
    best_prediction = None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(fit_x), generator=generator)
        total_loss = 0.0

        for start in range(0, len(fit_x), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[idx])
            element_loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                target_tensor[idx],
                reduction="none",
            )
            loss = (
                element_loss.mean(dim=0) * task_weights
            ).sum() / task_weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)

        if valid_x is not None:
            pred = predict_mmoe(model, valid_x)
            metric = evaluate(valid_user, valid_y, pred)
            print(
                "mmoe epoch=%d loss=%.6f primary=%.6f"
                % (
                    epoch,
                    total_loss / len(fit_x),
                    metric["primary"],
                ),
                flush=True,
            )
            if metric["primary"] > best_primary:
                best_primary = float(metric["primary"])
                best_epoch = epoch
                best_prediction = pred.copy()
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }

    if valid_x is not None:
        model.load_state_dict(best_state)
        return model, best_prediction, best_epoch
    return model, None, epochs


# Family 1: hierarchical non-parametric personalized target statistics.
eb_valid = empirical_bayes_predict(x_train, y_train, x_valid)

# Family 2: low-rank collaborative reconstruction.
svd_valid = fit_svd_predict(x_train, y_train, x_valid, rank=32)

# Family 3: auxiliary-supervised multi-gate mixture of experts.
mmoe_train_targets, aux_names = auxiliary_targets(train, y_train)
mmoe_model, mmoe_valid, mmoe_best_epoch = fit_mmoe(
    x_train,
    mmoe_train_targets,
    MMOE_EPOCHS,
    valid_x=x_valid,
    valid_user=valid.user_id,
    valid_y=y_valid,
)
del mmoe_model
gc.collect()

families = {
    "hierarchical_empirical_bayes": eb_valid,
    "latent_svd": svd_valid,
    "auxiliary_mmoe": mmoe_valid,
}

candidate_scores = {}
candidate_predictions = {}
candidate_recipes = {}

inc_metric = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores["incumbent"] = float(inc_metric["primary"])
candidate_predictions["incumbent"] = inc_valid
candidate_recipes["incumbent"] = ("incumbent", 1.0)

inc_rank = within_user_rank(valid.user_id, inc_valid)
blend_alphas = [0.20, 0.35, 0.50, 0.65, 0.80]

for family_name, prediction in families.items():
    prediction = np.asarray(prediction, dtype=np.float64)
    metric = evaluate(valid.user_id, y_valid, prediction)
    candidate_scores[family_name] = float(metric["primary"])
    candidate_predictions[family_name] = prediction
    candidate_recipes[family_name] = (family_name, 1.0)

    family_rank = within_user_rank(valid.user_id, prediction)
    for alpha in blend_alphas:
        name = "%s_blend_%02d" % (
            family_name,
            int(round(alpha * 100)),
        )
        # alpha is the new-family contribution.
        blended = alpha * family_rank + (1.0 - alpha) * inc_rank
        blended_metric = evaluate(valid.user_id, y_valid, blended)
        candidate_scores[name] = float(blended_metric["primary"])
        candidate_predictions[name] = blended
        candidate_recipes[name] = (family_name, alpha)

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = np.asarray(
    candidate_predictions[winner],
    dtype=np.float64,
)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print(
    "FINDINGS auxiliary_tasks=%s mmoe_best_epoch=%d winner=%s"
    % (",".join(aux_names), mmoe_best_epoch, winner),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(
        candidate_scores,
        sort_keys=True,
        separators=(",", ":"),
    ),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )

# Produce test scores with the selected recipe refit on train + validation.
test = load("test")
x_test = make_x(test)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

selected_family, selected_alpha = candidate_recipes[winner]

if selected_family == "incumbent":
    family_test = inc_test
else:
    x_combined = np.ascontiguousarray(
        np.concatenate([x_train, x_valid], axis=0),
        dtype=np.int64,
    )
    y_combined = np.ascontiguousarray(
        np.concatenate([
            y_train,
            y_valid.astype(np.float32),
        ]),
        dtype=np.float32,
    )

    if selected_family == "hierarchical_empirical_bayes":
        family_test = empirical_bayes_predict(
            x_combined,
            y_combined,
            x_test,
        )
    elif selected_family == "latent_svd":
        family_test = fit_svd_predict(
            x_combined,
            y_combined,
            x_test,
            rank=32,
        )
    elif selected_family == "auxiliary_mmoe":
        valid_aux_targets, valid_aux_names = auxiliary_targets(
            valid,
            y_valid.astype(np.float32),
        )
        if valid_aux_names != aux_names:
            raise RuntimeError(
                "Auxiliary task mismatch between train and validation"
            )
        combined_targets = np.ascontiguousarray(
            np.concatenate(
                [mmoe_train_targets, valid_aux_targets],
                axis=0,
            ),
            dtype=np.float32,
        )
        final_mmoe, _, _ = fit_mmoe(
            x_combined,
            combined_targets,
            max(1, mmoe_best_epoch),
        )
        family_test = predict_mmoe(final_mmoe, x_test)
        del final_mmoe
        gc.collect()
    else:
        raise RuntimeError("Unknown selected family: " + selected_family)

if selected_family == "incumbent":
    test_scores = inc_test
elif selected_alpha >= 0.999:
    test_scores = np.asarray(family_test, dtype=np.float64)
else:
    test_scores = (
        selected_alpha
        * within_user_rank(test.user_id, family_test)
        + (1.0 - selected_alpha)
        * within_user_rank(test.user_id, inc_test)
    )

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    ),
    flush=True,
)