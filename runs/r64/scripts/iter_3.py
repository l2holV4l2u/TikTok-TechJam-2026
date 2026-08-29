#!/usr/bin/env python3
import os
import time
import numpy as np
import torch
import torch.nn as nn
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

# Start timer
t0 = time.time()

# Settings
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
EPOCHS = 3
BATCH_SIZE = 8192
SEED = 12345

torch.manual_seed(SEED)
np.random.seed(SEED)

# Load data
train = load("train")
valid = load("valid")

# Quick checks
for f in FIELDS:
    if f not in train.X:
        raise RuntimeError(f"Expected field {f} in data but it is missing.")

# Prepare categorical arrays for chosen fields
X_train_cat = np.vstack([train.X[f].astype(np.int64) for f in FIELDS]).T  # (N, F)
y_train = train.y.astype(np.float32)

X_valid_cat = np.vstack([valid.X[f].astype(np.int64) for f in FIELDS]).T
y_valid = valid.y.astype(np.float32)

# Build an author-level smoothed long_view rate from the TRAINING table.
# For valid/test we use full-train counts; for train rows we compute a leave-one-out value.
# This is safe (no leakage) because we only derive from train split.
author_train_ids = train.X["author_id"].astype(np.int64)
max_author_id = int(author_train_ids.max()) if author_train_ids.size > 0 else 0
# Make arrays length at least max seen across train/valid to allow indexing by id present later.
# But we only need counts from train, so size = FEATURE_CARDINALITIES["author_id"] (safer).
author_card = int(FEATURE_CARDINALITIES.get("author_id", max_author_id + 1))
# Count totals and positive counts per author in train (full)
author_total = np.bincount(author_train_ids, minlength=author_card).astype(np.int64)
author_pos = np.bincount(author_train_ids, weights=y_train, minlength=author_card).astype(np.int64)

# For train rows: leave-one-out counts
train_author_ids = X_train_cat[:, FIELDS.index("author_id")]
train_counts_excl = author_total[train_author_ids] - 1
train_pos_excl = author_pos[train_author_ids] - y_train
# Laplace smoothing alpha=1 (equivalent to (pos_excl+1)/(count_excl+2))
train_author_rate = (train_pos_excl + 1.0) / (train_counts_excl + 2.0)
# For any negative counts_excl (shouldn't happen) clip
train_author_rate = np.where(train_counts_excl >= 0, train_author_rate, 0.5)

# For valid rows: use full-train counts (no leakage)
valid_author_ids = X_valid_cat[:, FIELDS.index("author_id")]
valid_author_rate = (author_pos[valid_author_ids] + 1.0) / (author_total[valid_author_ids] + 2.0)
# If an author in valid wasn't seen in train, author_total[...] == 0 and rate becomes 0.5 (the prior)
valid_author_rate = np.where(author_total[valid_author_ids] >= 0, valid_author_rate, 0.5)

# Store as numeric columns shaped (N, 1)
X_train_num = train_author_rate.reshape(-1, 1).astype(np.float32)
X_valid_num = valid_author_rate.reshape(-1, 1).astype(np.float32)

# For test we will compute later from train counts (not touching test.y)
# Determine embedding sizes from FEATURE_CARDINALITIES.
emb_sizes = {}
for f in FIELDS:
    card = FEATURE_CARDINALITIES.get(f, None)
    if card is None:
        raise RuntimeError(f"FEATURE_CARDINALITIES missing entry for {f}")
    emb_sizes[f] = int(card) + 1  # keep padding_idx=0 safe

# FM model augmented to accept numeric features that also participate in interactions
class FMWithNumeric(nn.Module):
    def __init__(self, fields, emb_sizes, k, n_numeric=0):
        super().__init__()
        self.fields = fields
        self.k = k
        self.n_numeric = n_numeric
        # linear embedding per categorical field (scalar)
        self.linears = nn.ModuleDict({
            f: nn.Embedding(emb_sizes[f], 1, padding_idx=0) for f in fields
        })
        # categorical embeddings for interactions
        self.embs = nn.ModuleDict({
            f: nn.Embedding(emb_sizes[f], k, padding_idx=0) for f in fields
        })
        # numeric linear coefficients and numeric embeddings for interactions
        if n_numeric > 0:
            # linear scalar per numeric input
            self.num_linear = nn.Parameter(torch.zeros(n_numeric))
            # numeric interaction vectors (n_numeric x k)
            self.num_emb = nn.Parameter(torch.randn(n_numeric, k) * 0.01)
        else:
            self.num_linear = None
            self.num_emb = None
        # global bias
        self.bias = nn.Parameter(torch.zeros(1))
        # init embeddings
        for emb in self.embs.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        for lin in self.linears.values():
            nn.init.zeros_(lin.weight)

    def forward(self, x_cat, x_num=None):
        # x_cat: LongTensor (B, F)
        # x_num: FloatTensor (B, n_numeric) or None
        B = x_cat.shape[0]
        device = x_cat.device
        lin_sum = torch.zeros(B, dtype=torch.float32, device=device)
        embs = []
        for i, f in enumerate(self.fields):
            idx = x_cat[:, i]
            lin_val = self.linears[f](idx).squeeze(-1)  # (B,)
            lin_sum += lin_val
            emb = self.embs[f](idx)  # (B, k)
            embs.append(emb)
        # numeric linear contribution
        if self.n_numeric > 0 and x_num is not None:
            # linear scalars:
            lin_sum = lin_sum + (x_num * self.num_linear.view(1, -1)).sum(dim=1)
            # numeric produce embeddings by scaling learned vector by scalar value
            # x_num: (B, n_numeric); num_emb: (n_numeric, k) => produce (B, k) per numeric then sum across numerics
            # We'll append each numeric's (B,k) to embs so interactions include them
            # compute (B, n_numeric, k) then sum along numeric dimension into list elements per numeric
            num_embs = (x_num.unsqueeze(2) * self.num_emb.unsqueeze(0))  # (B, n_numeric, k)
            # Break into list of (B,k) for interaction formula
            for j in range(self.n_numeric):
                embs.append(num_embs[:, j, :])
        # interaction term
        sum_emb = torch.stack(embs, dim=0).sum(dim=0)  # (B, k)
        sum_square = (sum_emb * sum_emb).sum(dim=1)    # (B,)
        sq_sum = torch.stack([(e * e).sum(dim=1) for e in embs], dim=0).sum(dim=0)  # (B,)
        interaction = 0.5 * (sum_square - sq_sum)
        logits = self.bias + lin_sum + interaction
        return logits

# Instantiate model with one numeric (author rate)
device = torch.device("cpu")
model = FMWithNumeric(FIELDS, emb_sizes, K, n_numeric=1).to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()

# Training arrays
X_train_cat_np = X_train_cat  # (N, F)
X_train_num_np = X_train_num  # (N, 1)
n_train = X_train_cat_np.shape[0]
indices = np.arange(n_train)

# Training loop (vectorized batches)
model.train()
for epoch in range(EPOCHS):
    np.random.shuffle(indices)
    for start in range(0, n_train, BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]
        xb_cat = torch.from_numpy(X_train_cat_np[batch_idx]).to(torch.long).to(device)
        xb_num = torch.from_numpy(X_train_num_np[batch_idx]).to(torch.float32).to(device)
        yb = torch.from_numpy(y_train[batch_idx]).to(torch.float32).to(device)
        opt.zero_grad()
        logits = model(xb_cat, xb_num)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

# Inference helper
def predict_batch_numpy(model, X_cat_np, X_num_np=None, batch_size=8192):
    model.eval()
    preds = np.empty((X_cat_np.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, X_cat_np.shape[0], batch_size):
            end = start + batch_size
            xb_cat = torch.from_numpy(X_cat_np[start:end]).to(torch.long).to(device)
            xb_num = None
            if X_num_np is not None:
                xb_num = torch.from_numpy(X_num_np[start:end]).to(torch.float32).to(device)
            logits = model(xb_cat, xb_num)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            preds[start:end] = probs
    return preds

# Predict on validation
valid_scores = predict_batch_numpy(model, X_valid_cat, X_valid_num, batch_size=BATCH_SIZE)

# Save validation scores for harness
out = os.environ.get("ITER_OUT")
if out:
    np.save(os.path.join(out, "scores_valid.npy"), np.asarray(valid_scores, dtype=np.float64))

# Evaluate using the provided evaluator
res = evaluate(valid.user_id, valid.y, valid_scores)
primary = float(res["primary"])
gauc = float(res["gauc"])
ndcg5 = float(res["ndcg@5"])

# FINDINGS: brief statistics about the author-rate numeric we added
mean_rate = float(X_train_num.mean())
std_rate = float(X_train_num.std())
print(f'FINDINGS author_rate_train_mean={mean_rate:.6f} author_rate_train_std={std_rate:.6f}')

# Score test split and save scores (must not touch test.y)
te = load("test")
X_test_cat = np.vstack([te.X[f].astype(np.int64) for f in FIELDS]).T
# Compute test author-rate from train counts (full-train prior)
test_author_ids = X_test_cat[:, FIELDS.index("author_id")]
test_author_rate = (author_pos[test_author_ids] + 1.0) / (author_total[test_author_ids] + 2.0)
test_author_rate = test_author_rate.reshape(-1, 1).astype(np.float32)
test_scores = predict_batch_numpy(model, X_test_cat, test_author_rate, batch_size=BATCH_SIZE)
if out:
    np.save(os.path.join(out, "scores_test.npy"), np.asarray(test_scores, dtype=np.float64))

# End timer
t1 = time.time()
wall = float(t1 - t0)

# Final required single-line METRICS output
print(f'METRICS {{"primary": {primary:.6f}, "gauc": {gauc:.6f}, "ndcg@5": {ndcg5:.6f}, "gpu_seconds": {wall:.3f}}}')