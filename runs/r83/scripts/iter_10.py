import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 18473
FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour"
]
DIM = 12
BATCH_SIZE = 4096
PRED_BATCH = 32768
LR = 0.003
EPOCHS = {
    "autoint": 3,
    "xdeepfm": 3,
    "mmoe": 4,
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

cards = np.asarray(
    [int(FEATURE_CARDINALITIES[name]) for name in FIELDS],
    dtype=np.int64
)
offsets = np.cumsum(np.r_[0, cards[:-1]], dtype=np.int64)
TOTAL_CARDINALITY = int(cards.sum())
N_FIELDS = len(FIELDS)


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int64) + offsets[j]
            for j, name in enumerate(FIELDS)
        ]),
        dtype=np.int64
    )


class FeatureBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, DIM)
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.field_embedding = nn.Parameter(
            torch.zeros(1, N_FIELDS, DIM)
        )
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.field_embedding, std=0.01)

    def embedded(self, x):
        return self.embedding(x) + self.field_embedding

    def wide(self, x):
        return self.linear(x).sum(dim=1).squeeze(1)


class AutoIntModel(FeatureBase):
    """Self-attention forms candidate-conditioned field interactions."""

    def __init__(self):
        super().__init__()
        self.q1 = nn.Linear(DIM, DIM, bias=False)
        self.k1 = nn.Linear(DIM, DIM, bias=False)
        self.v1 = nn.Linear(DIM, DIM, bias=False)
        self.o1 = nn.Linear(DIM, DIM, bias=False)

        self.q2 = nn.Linear(DIM, DIM, bias=False)
        self.k2 = nn.Linear(DIM, DIM, bias=False)
        self.v2 = nn.Linear(DIM, DIM, bias=False)
        self.o2 = nn.Linear(DIM, DIM, bias=False)

        self.norm1 = nn.LayerNorm(DIM)
        self.norm2 = nn.LayerNorm(DIM)
        self.output = nn.Sequential(
            nn.Linear(N_FIELDS * DIM, 48),
            nn.ReLU(),
            nn.Linear(48, 1)
        )

    @staticmethod
    def attend(z, q_layer, k_layer, v_layer, o_layer):
        q = q_layer(z)
        k = k_layer(z)
        v = v_layer(z)
        attention = torch.softmax(
            torch.matmul(q, k.transpose(1, 2)) / (DIM ** 0.5),
            dim=-1
        )
        return o_layer(torch.matmul(attention, v))

    def forward(self, x):
        z = self.embedded(x)
        a1 = self.attend(z, self.q1, self.k1, self.v1, self.o1)
        z = self.norm1(z + F.relu(a1))
        a2 = self.attend(z, self.q2, self.k2, self.v2, self.o2)
        z = self.norm2(z + F.relu(a2))
        return self.wide(x) + self.output(z.flatten(1)).squeeze(1)


class CINLayer(nn.Module):
    def __init__(self, field0, field_in, field_out):
        super().__init__()
        self.field0 = field0
        self.field_in = field_in
        self.conv = nn.Conv1d(
            field0 * field_in, field_out, kernel_size=1
        )

    def forward(self, x0, xk):
        # For every embedding coordinate, explicitly cross original fields
        # with the previous CIN layer's feature maps.
        crossed = torch.einsum("bfd,bhd->bfhd", x0, xk)
        crossed = crossed.reshape(
            crossed.shape[0],
            self.field0 * self.field_in,
            crossed.shape[-1]
        )
        return F.relu(self.conv(crossed))


class XDeepFMModel(FeatureBase):
    """CIN explicitly compresses bounded-order field crosses."""

    def __init__(self):
        super().__init__()
        self.cin1 = CINLayer(N_FIELDS, N_FIELDS, 16)
        self.cin2 = CINLayer(N_FIELDS, 16, 16)
        self.cin_output = nn.Linear(32, 1)
        self.deep = nn.Sequential(
            nn.Linear(N_FIELDS * DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x0 = self.embedded(x)
        h1 = self.cin1(x0, x0)
        h2 = self.cin2(x0, h1)
        cin_summary = torch.cat(
            [h1.sum(dim=2), h2.sum(dim=2)], dim=1
        )
        cin_score = self.cin_output(cin_summary).squeeze(1)
        deep_score = self.deep(x0.flatten(1)).squeeze(1)
        return self.wide(x) + cin_score + deep_score


class MMoEModel(FeatureBase):
    """Task-specific gates mix shared nonlinear experts."""

    def __init__(self, n_tasks=3, n_experts=4):
        super().__init__()
        input_dim = N_FIELDS * DIM
        self.n_tasks = n_tasks
        self.n_experts = n_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, n_experts)
            for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x, all_tasks=False):
        flat = self.embedded(x).flatten(1)
        experts = torch.stack(
            [expert(flat) for expert in self.experts], dim=1
        )

        outputs = []
        for task in range(self.n_tasks):
            gate = torch.softmax(self.gates[task](flat), dim=1)
            mixed = (experts * gate.unsqueeze(2)).sum(dim=1)
            score = self.towers[task](mixed).squeeze(1)
            if task == 0:
                score = score + self.wide(x)
            outputs.append(score)

        if all_tasks:
            return outputs
        return outputs[0]


def build_model(family):
    if family == "autoint":
        return AutoIntModel()
    if family == "xdeepfm":
        return XDeepFMModel()
    if family == "mmoe":
        return MMoEModel()
    raise ValueError("Unknown family: " + family)


def fit_model(family, X, y, auxiliary=None):
    family_seed = {
        "autoint": SEED + 101,
        "xdeepfm": SEED + 211,
        "mmoe": SEED + 307,
    }[family]
    torch.manual_seed(family_seed)

    model = build_model(family)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=2e-6
    )

    xt = torch.from_numpy(X)
    yt = torch.from_numpy(
        np.asarray(y, dtype=np.float32)
    )

    if family == "mmoe":
        if auxiliary is None:
            raise ValueError("MMoE requires auxiliary training targets")
        click_t = torch.from_numpy(
            np.asarray(auxiliary[0], dtype=np.float32)
        )
        like_t = torch.from_numpy(
            np.asarray(auxiliary[1], dtype=np.float32)
        )

    rng = np.random.default_rng(family_seed + 1000)

    for epoch in range(EPOCHS[family]):
        order = rng.permutation(len(X))
        model.train()

        for start in range(0, len(order), BATCH_SIZE):
            idx_np = order[start:start + BATCH_SIZE]
            idx = torch.from_numpy(idx_np)
            xb = xt[idx]

            if family == "mmoe":
                logits = model(xb, all_tasks=True)
                main_loss = F.binary_cross_entropy_with_logits(
                    logits[0], yt[idx]
                )
                click_loss = F.binary_cross_entropy_with_logits(
                    logits[1], click_t[idx]
                )
                like_loss = F.binary_cross_entropy_with_logits(
                    logits[2], like_t[idx]
                )
                # Main relevance remains dominant; auxiliary outcomes are
                # training targets only and are never row-level inputs.
                loss = main_loss + 0.20 * click_loss + 0.15 * like_loss
            else:
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(
                    logits, yt[idx]
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict(model, X):
    model.eval()
    xt = torch.from_numpy(X)
    result = np.empty(len(X), dtype=np.float32)

    for start in range(0, len(X), PRED_BATCH):
        end = min(start + PRED_BATCH, len(X))
        result[start:end] = model(xt[start:end]).cpu().numpy()

    return result


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - mean) / std


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids
    ))
    sorted_users = user_ids[order]

    starts_mask = np.r_[
        True,
        sorted_users[1:] != sorted_users[:-1]
    ]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    group_start_per_row = np.repeat(starts, lengths)
    group_length_per_row = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.int64) - group_start_per_row

    ranked = positions / np.maximum(group_length_per_row - 1, 1)
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def make_variants(family, new_scores, incumbent_scores, users):
    variants = [
        (
            family + "_standalone",
            np.asarray(new_scores, dtype=np.float64),
            (family, "standalone", 0.0)
        )
    ]

    new_z = zscore(new_scores)
    incumbent_z = zscore(incumbent_scores)
    new_rank = within_user_rank(users, new_scores)
    incumbent_rank = within_user_rank(users, incumbent_scores)

    for incumbent_weight in (0.25, 0.50, 0.75):
        variants.append((
            family + "_zblend_inc%.2f" % incumbent_weight,
            incumbent_weight * incumbent_z
            + (1.0 - incumbent_weight) * new_z,
            (family, "zblend", incumbent_weight)
        ))
        variants.append((
            family + "_rankblend_inc%.2f" % incumbent_weight,
            incumbent_weight * incumbent_rank
            + (1.0 - incumbent_weight) * new_rank,
            (family, "rankblend", incumbent_weight)
        ))

    return variants


train = load("train")
valid = load("valid")

X_train = make_matrix(train)
X_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

train_aux = (
    np.asarray(train.aux["is_click"], dtype=np.float32),
    np.asarray(train.aux["is_like"], dtype=np.float32)
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)

family_predictions = {}
for family in ("autoint", "xdeepfm", "mmoe"):
    auxiliary = train_aux if family == "mmoe" else None
    model = fit_model(
        family, X_train, y_train, auxiliary=auxiliary
    )
    family_predictions[family] = predict(
        model, X_valid
    ).astype(np.float64)
    del model
    gc.collect()

all_candidates = [
    (
        "trusted_incumbent",
        inc_valid,
        ("incumbent", "standalone", 1.0)
    )
]

for family, scores in family_predictions.items():
    all_candidates.extend(
        make_variants(
            family, scores, inc_valid, valid_users
        )
    )

candidate_scores = {}
candidate_details = {}
best_name = None
best_scores = None
best_spec = None
best_primary = -np.inf

for name, scores, spec in all_candidates:
    result = evaluate(valid_users, y_valid, scores)
    primary = float(result["primary"])
    candidate_scores[name] = primary
    candidate_details[name] = {
        "primary": primary,
        "gauc": float(result["gauc"]),
        "ndcg@5": float(result["ndcg@5"])
    }

    if primary > best_primary:
        best_primary = primary
        best_name = name
        best_scores = np.asarray(scores, dtype=np.float64)
        best_spec = spec

metrics = evaluate(valid_users, y_valid, best_scores)

print("CANDIDATES " + json.dumps(
    candidate_scores, sort_keys=True
))
print("FINDINGS " + json.dumps({
    "winner": best_name,
    "trusted_incumbent": candidate_details["trusted_incumbent"],
    "autoint_standalone": candidate_details["autoint_standalone"],
    "xdeepfm_standalone": candidate_details["xdeepfm_standalone"],
    "mmoe_standalone": candidate_details["mmoe_standalone"]
}, sort_keys=True))

test = load("test")
test_users = np.asarray(test.user_id)
inc_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)

if best_spec[0] == "incumbent":
    test_scores = inc_test.copy()
else:
    family, transform, incumbent_weight = best_spec

    X_combined = np.concatenate(
        [X_train, X_valid], axis=0
    )
    y_combined = np.concatenate(
        [y_train, y_valid], axis=0
    )
    X_test = make_matrix(test)

    combined_aux = None
    if family == "mmoe":
        combined_aux = (
            np.concatenate([
                train_aux[0],
                np.asarray(
                    valid.aux["is_click"], dtype=np.float32
                )
            ]),
            np.concatenate([
                train_aux[1],
                np.asarray(
                    valid.aux["is_like"], dtype=np.float32
                )
            ])
        )

    final_model = fit_model(
        family,
        X_combined,
        y_combined,
        auxiliary=combined_aux
    )
    new_test = predict(
        final_model, X_test
    ).astype(np.float64)

    if transform == "standalone":
        test_scores = new_test
    elif transform == "zblend":
        test_scores = (
            float(incumbent_weight) * zscore(inc_test)
            + (1.0 - float(incumbent_weight)) * zscore(new_test)
        )
    elif transform == "rankblend":
        test_scores = (
            float(incumbent_weight)
            * within_user_rank(test_users, inc_test)
            + (1.0 - float(incumbent_weight))
            * within_user_rank(test_users, new_test)
        )
    else:
        raise RuntimeError("Unknown transform: " + transform)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

wall = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(wall)
}))