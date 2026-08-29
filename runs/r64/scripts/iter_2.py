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

# Settings that match the reference baseline
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
EPOCHS = 3
BATCH_SIZE = 8192  # large batches to keep loop count reasonable on CPU
SEED = 12345

torch.manual_seed(SEED)
np.random.seed(SEED)

# Load training and validation splits
train = load("train")
valid = load("valid")

# Quick checks
for f in FIELDS:
    if f not in train.X:
        raise RuntimeError(f"Expected field {f} in data but it is missing.")

# Prepare integer arrays for chosen fields
X_train = np.vstack([train.X[f].astype(np.int64) for f in FIELDS]).T  # shape (N, F)
y_train = train.y.astype(np.float32)

X_valid = np.vstack([valid.X[f].astype(np.int64) for f in FIELDS]).T
y_valid = valid.y.astype(np.float32)

# Determine embedding sizes from FEATURE_CARDINALITIES.
# We add +1 to be safe (ids are 0..card-1 with 0 reserved for unseen).
emb_sizes = {}
for f in FIELDS:
    card = FEATURE_CARDINALITIES.get(f, None)
    if card is None:
        raise RuntimeError(f"FEATURE_CARDINALITIES missing entry for {f}")
    emb_sizes[f] = int(card) + 1

# PyTorch model implementing a classic Factorization Machine for categorical fields
class FM(nn.Module):
    def __init__(self, fields, emb_sizes, k):
        super().__init__()
        self.fields = fields
        self.k = k
        # linear terms per field (scalar per id)
        self.linears = nn.ModuleDict({
            f: nn.Embedding(emb_sizes[f], 1, padding_idx=0) for f in fields
        })
        # embeddings for interactions per field
        self.embs = nn.ModuleDict({
            f: nn.Embedding(emb_sizes[f], k, padding_idx=0) for f in fields
        })
        # global bias
        self.bias = nn.Parameter(torch.zeros(1))
        # init
        for emb in self.embs.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        for lin in self.linears.values():
            nn.init.zeros_(lin.weight)

    def forward(self, x):
        # x: LongTensor shape (B, F) where columns correspond to self.fields
        B = x.shape[0]
        device = x.device
        # linear term
        lin_sum = torch.zeros(B, dtype=torch.float32, device=device)
        embs = []
        for i, f in enumerate(self.fields):
            idx = x[:, i]
            lin_val = self.linears[f](idx).squeeze(-1)  # (B,)
            lin_sum += lin_val
            emb = self.embs[f](idx)  # (B, k)
            embs.append(emb)
        # interaction term: 0.5*(sum_square - sum_of_squares)
        sum_emb = torch.stack(embs, dim=0).sum(dim=0)  # (B, k)
        sum_square = (sum_emb * sum_emb).sum(dim=1)    # (B,)
        sq_sum = torch.stack([(e * e).sum(dim=1) for e in embs], dim=0).sum(dim=0)  # (B,)
        interaction = 0.5 * (sum_square - sq_sum)
        logits = self.bias + lin_sum + interaction
        return logits

# Instantiate model
device = torch.device("cpu")
model = FM(FIELDS, emb_sizes, K).to(device)

# Prepare optimizer and loss
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()

# Convert data to torch for indexing; we will select batches via numpy indices and then convert
X_train_np = X_train  # (N, F)
n_train = X_train_np.shape[0]
indices = np.arange(n_train)

# Training loop
model.train()
for epoch in range(EPOCHS):
    np.random.shuffle(indices)
    # iterate batches
    for start in range(0, n_train, BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]
        xb = torch.from_numpy(X_train_np[batch_idx]).to(torch.long).to(device)
        yb = torch.from_numpy(y_train[batch_idx]).to(torch.float32).to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
    # optional lightweight progress print (commented to keep output minimal)
    # print(f"Epoch {epoch+1}/{EPOCHS} finished")

# Inference helpers
def predict_batch_numpy(model, X_np, batch_size=8192):
    model.eval()
    preds = np.empty((X_np.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, X_np.shape[0], batch_size):
            xb = torch.from_numpy(X_np[start:start + batch_size]).to(torch.long).to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            preds[start:start + batch_size] = probs
    return preds

# Predict on validation
valid_scores = predict_batch_numpy(model, X_valid, batch_size=BATCH_SIZE)

# Save validation scores for harness
out = os.environ.get("ITER_OUT")
if out:
    np.save(os.path.join(out, "scores_valid.npy"), np.asarray(valid_scores, dtype=np.float64))

# Evaluate using the provided evaluator
res = evaluate(valid.user_id, valid.y, valid_scores)
primary = float(res["primary"])
gauc = float(res["gauc"])
ndcg5 = float(res["ndcg@5"])

# Score test split and save scores (must not touch test.y)
te = load("test")
X_test = np.vstack([te.X[f].astype(np.int64) for f in FIELDS]).T
test_scores = predict_batch_numpy(model, X_test, batch_size=BATCH_SIZE)
if out:
    np.save(os.path.join(out, "scores_test.npy"), np.asarray(test_scores, dtype=np.float64))

# End timer
t1 = time.time()
wall = float(t1 - t0)

# Final required single-line METRICS output
print(f'METRICS {{"primary": {primary:.6f}, "gauc": {gauc:.6f}, "ndcg@5": {ndcg5:.6f}, "gpu_seconds": {wall:.3f}}}')