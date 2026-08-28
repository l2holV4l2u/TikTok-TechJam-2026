import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FM_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
FIBI_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]

FM_RANK = 16
FM_LR = 0.001
FM_BATCH = 4096
FM_EPOCHS = 12

EMBED_DIM = 8
FIBI_LR = 0.001
FIBI_BATCH = 8192
FIBI_EPOCHS = 7
BLEND_ALPHAS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


def make_offset_matrix(split, fields):
    cards = [int(FEATURE_CARDINALITIES[name]) for name in fields]
    offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
    x = np.stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in fields],
        axis=1,
    )
    return np.ascontiguousarray(x + offsets[None, :]), int(sum(cards))


class FactorizationMachine(nn.Module):
    def __init__(self, n_tokens, rank):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, rank + 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.01)

    def forward(self, x):
        z = self.embedding(x)
        linear = z[:, :, 0].sum(dim=1)
        factors = z[:, :, 1:]
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


class FiBiNET(nn.Module):
    def __init__(self, n_tokens, n_fields, embed_dim):
        super().__init__()
        self.n_fields = n_fields
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(n_tokens, embed_dim)
        self.wide = nn.Embedding(n_tokens, 1)
        self.bias = nn.Parameter(torch.zeros(()))

        reduction_hidden = max(3, n_fields // 3)
        self.senet = nn.Sequential(
            nn.Linear(n_fields, reduction_hidden),
            nn.ReLU(),
            nn.Linear(reduction_hidden, n_fields),
            nn.Sigmoid(),
        )

        # Shared bilinear transform is applied to the left member of each
        # field pair before its elementwise interaction with the right member.
        self.bilinear = nn.Linear(embed_dim, embed_dim, bias=False)

        left = []
        right = []
        for i in range(n_fields):
            for j in range(i + 1, n_fields):
                left.append(i)
                right.append(j)
        self.register_buffer("pair_left", torch.tensor(left, dtype=torch.long))
        self.register_buffer("pair_right", torch.tensor(right, dtype=torch.long))

        n_pairs = len(left)
        deep_input = (n_fields + n_pairs) * embed_dim
        self.deep = nn.Sequential(
            nn.Linear(deep_input, 96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 1),
        )

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.03)
        nn.init.zeros_(self.wide.weight)
        nn.init.xavier_uniform_(self.bilinear.weight)

    def forward(self, x):
        e = self.embedding(x)

        # FiBiNET SENET squeeze: summarize each field embedding and use all
        # fields jointly to produce an impression-specific importance scale.
        squeeze = e.mean(dim=2)
        field_scale = 2.0 * self.senet(squeeze)
        se = e * field_scale.unsqueeze(2)

        transformed = self.bilinear(se)
        pair_vectors = (
            transformed.index_select(1, self.pair_left)
            * se.index_select(1, self.pair_right)
        )

        deep_input = torch.cat(
            [se.flatten(start_dim=1), pair_vectors.flatten(start_dim=1)],
            dim=1,
        )
        deep_logit = self.deep(deep_input).squeeze(1)
        wide_logit = self.wide(x).squeeze(2).sum(dim=1)
        return self.bias + wide_logit + deep_logit


def predict(model, x_np, batch_size=32768):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, x_np.shape[0], batch_size):
            end = min(start + batch_size, x_np.shape[0])
            xb = torch.from_numpy(x_np[start:end])
            result[start:end] = (
                model(xb).detach().cpu().numpy().astype(np.float64)
            )
    return result


def metric_primary(user_id, y, scores):
    return float(evaluate(user_id, y, scores)["primary"])


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
n_train = len(y_train)
criterion = nn.BCEWithLogitsLoss()

xtr_fm, fm_cardinality = make_offset_matrix(train, FM_FIELDS)
xva_fm, _ = make_offset_matrix(valid, FM_FIELDS)

# First reproduce and checkpoint the reliable FM component.
fm = FactorizationMachine(fm_cardinality, FM_RANK)
fm_sparse_opt = torch.optim.SparseAdam(
    [fm.embedding.weight], lr=FM_LR
)
fm_bias_opt = torch.optim.Adam([fm.bias], lr=FM_LR)

rng_fm = np.random.default_rng(SEED)
best_fm_primary = -np.inf
best_fm_epoch = -1
best_fm_state = None
best_fm_valid = None

for epoch in range(FM_EPOCHS):
    fm.train()
    permutation = rng_fm.permutation(n_train)
    total_loss = 0.0

    for start in range(0, n_train, FM_BATCH):
        idx = permutation[start:start + FM_BATCH]
        xb = torch.from_numpy(xtr_fm[idx])
        yb = torch.from_numpy(y_train[idx])

        fm_sparse_opt.zero_grad(set_to_none=True)
        fm_bias_opt.zero_grad(set_to_none=True)
        logits = fm(xb)
        loss = criterion(logits, yb)
        loss.backward()
        fm_sparse_opt.step()
        fm_bias_opt.step()

        total_loss += float(loss.detach()) * len(idx)

    va_scores = predict(fm, xva_fm)
    va_metrics = evaluate(valid.user_id, valid.y, va_scores)
    primary = float(va_metrics["primary"])
    print(
        "fm_epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            epoch + 1,
            total_loss / n_train,
            primary,
            float(va_metrics["gauc"]),
            float(va_metrics["ndcg@5"]),
        ),
        flush=True,
    )

    if primary > best_fm_primary:
        best_fm_primary = primary
        best_fm_epoch = epoch + 1
        best_fm_valid = va_scores.copy()
        best_fm_state = {
            "embedding": fm.embedding.weight.detach().clone(),
            "bias": fm.bias.detach().clone(),
        }

with torch.no_grad():
    fm.embedding.weight.copy_(best_fm_state["embedding"])
    fm.bias.copy_(best_fm_state["bias"])

fm_valid_scores = predict(fm, xva_fm)
fm_metrics = evaluate(valid.user_id, valid.y, fm_valid_scores)

# Train the feature-importance and bilinear interaction ranker independently.
xtr_fibi, fibi_cardinality = make_offset_matrix(train, FIBI_FIELDS)
xva_fibi, _ = make_offset_matrix(valid, FIBI_FIELDS)

torch.manual_seed(SEED + 17)
fibi = FiBiNET(fibi_cardinality, len(FIBI_FIELDS), EMBED_DIM)
fibi_opt = torch.optim.AdamW(
    fibi.parameters(),
    lr=FIBI_LR,
    weight_decay=1e-6,
)

rng_fibi = np.random.default_rng(SEED + 17)
best_blend_primary = float(fm_metrics["primary"])
best_alpha = 0.0
best_fibi_epoch = 0
best_fibi_state = None
best_standalone_primary = -np.inf

for epoch in range(FIBI_EPOCHS):
    fibi.train()
    permutation = rng_fibi.permutation(n_train)
    total_loss = 0.0

    for start in range(0, n_train, FIBI_BATCH):
        idx = permutation[start:start + FIBI_BATCH]
        xb = torch.from_numpy(xtr_fibi[idx])
        yb = torch.from_numpy(y_train[idx])

        fibi_opt.zero_grad(set_to_none=True)
        logits = fibi(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fibi.parameters(), max_norm=5.0)
        fibi_opt.step()
        total_loss += float(loss.detach()) * len(idx)

    fibi_valid = predict(fibi, xva_fibi)
    epoch_candidates = {}
    epoch_best = -np.inf
    epoch_best_alpha = None

    for alpha in BLEND_ALPHAS:
        blended = (1.0 - alpha) * fm_valid_scores + alpha * fibi_valid
        score = metric_primary(valid.user_id, valid.y, blended)
        epoch_candidates[alpha] = score
        if score > epoch_best:
            epoch_best = score
            epoch_best_alpha = alpha

    best_standalone_primary = max(
        best_standalone_primary, epoch_candidates[1.0]
    )

    print(
        "fibi_epoch=%d loss=%.6f standalone=%.6f best_blend=%.6f alpha=%.2f"
        % (
            epoch + 1,
            total_loss / n_train,
            epoch_candidates[1.0],
            epoch_best,
            epoch_best_alpha,
        ),
        flush=True,
    )

    if epoch_best > best_blend_primary:
        best_blend_primary = epoch_best
        best_alpha = float(epoch_best_alpha)
        best_fibi_epoch = epoch + 1
        best_fibi_state = {
            key: value.detach().clone()
            for key, value in fibi.state_dict().items()
        }

# If no FiBiNET blend exceeded the FM, retain the FM exactly.
if best_alpha > 0.0 and best_fibi_state is not None:
    fibi.load_state_dict(best_fibi_state)
    selected_fibi_valid = predict(fibi, xva_fibi)
    final_valid_scores = (
        (1.0 - best_alpha) * fm_valid_scores
        + best_alpha * selected_fibi_valid
    )
else:
    final_valid_scores = fm_valid_scores.copy()

final_metrics = evaluate(valid.user_id, valid.y, final_valid_scores)

candidate_summary = {
    "fm": float(fm_metrics["primary"]),
    "best_fibinet_standalone": float(best_standalone_primary),
    "selected_blend": float(final_metrics["primary"]),
}
print(
    "CANDIDATES " + json.dumps(candidate_summary, separators=(",", ":")),
    flush=True,
)
print(
    "FINDINGS selected_fm_epoch=%d selected_fibi_epoch=%d selected_alpha=%.2f"
    % (best_fm_epoch, best_fibi_epoch, best_alpha),
    flush=True,
)

# Score the hidden test split using exactly the validation-selected models
# and blend coefficient, without reading or using test labels.
test = load("test")
xte_fm, _ = make_offset_matrix(test, FM_FIELDS)
fm_test_scores = predict(fm, xte_fm)

if best_alpha > 0.0 and best_fibi_state is not None:
    xte_fibi, _ = make_offset_matrix(test, FIBI_FIELDS)
    fibi_test_scores = predict(fibi, xte_fibi)
    test_scores = (
        (1.0 - best_alpha) * fm_test_scores
        + best_alpha * fibi_test_scores
    )
else:
    test_scores = fm_test_scores

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": 0.0,
        },
        separators=(",", ":"),
    )
)