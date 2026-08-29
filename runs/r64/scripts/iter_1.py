# Complete runnable script to reproduce the official FM baseline.
# It fits a Factorization Machine (k=16, lr=0.001) on the five categorical fields:
#   user_id, video_id, author_id, tab, duration_bucket
# and evaluates on validation, then scores the test split (without touching test labels).
# Saves validation and test scores to ITER_OUT as required and prints the METRICS line.
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

start_time = time.time()

# Fields used by baseline (the names match those in the provided dataset)
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16                 # embedding size
LR = 0.001             # learning rate (baseline)
BATCH_SIZE = 16384     # large batch to speed up on CPU
EPOCHS = 3             # modest number of epochs (fit within time limit)

# Load train and valid
train = load("train")
valid = load("valid")

# Sanity: ensure our chosen fields exist
for f in FIELDS:
    if f not in train.X:
        raise RuntimeError(f"Field {f} not present in data")

# Prepare numpy arrays of indices for each field for train and valid
# Each is shape (n_rows,)
def stack_field_arrays(split, fields):
    arrs = []
    for f in fields:
        a = split.X[f].astype(np.int64)
        arrs.append(a)
    # shape (n_rows, n_fields)
    return np.stack(arrs, axis=1)

X_train = stack_field_arrays(train, FIELDS)
y_train = train.y.astype(np.float32)

X_valid = stack_field_arrays(valid, FIELDS)
y_valid = valid.y.astype(np.float32)

n_train = X_train.shape[0]
n_valid = X_valid.shape[0]

# Build PyTorch tensors (on CPU)
device = torch.device("cpu")
X_train_t = torch.from_numpy(X_train).to(device)
y_train_t = torch.from_numpy(y_train).to(device)

X_valid_t = torch.from_numpy(X_valid).to(device)
y_valid_t = torch.from_numpy(y_valid).to(device)

# Determine embedding sizes from FEATURE_CARDINALITIES
embed_sizes = []
for f in FIELDS:
    card = FEATURE_CARDINALITIES[f]
    # pipeline uses contiguous ids and 0=unseen; FEATURE_CARDINALITIES should match max id capacity
    # We'll trust FEATURE_CARDINALITIES as the number of ids and create embeddings of that size.
    embed_sizes.append(int(card))

class FM(nn.Module):
    def __init__(self, field_sizes, k):
        super().__init__()
        self.n_fields = len(field_sizes)
        # linear term: one scalar weight per id
        self.linears = nn.ModuleList([
            nn.Embedding(num_embeddings=sz, embedding_dim=1) for sz in field_sizes
        ])
        # factor embeddings for interactions
        self.factors = nn.ModuleList([
            nn.Embedding(num_embeddings=sz, embedding_dim=k) for sz in field_sizes
        ])
        # global bias
        self.bias = nn.Parameter(torch.zeros(1))
        # initialize embeddings small
        for emb in self.linears:
            nn.init.normal_(emb.weight, mean=0.0, std=1e-3)
        for emb in self.factors:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

    def forward(self, x):
        # x: (batch, n_fields) long tensor of indices
        batch = x.shape[0]
        # linear part
        linear_terms = []
        factor_list = []
        for i in range(self.n_fields):
            idx = x[:, i]  # (batch,)
            linear_i = self.linears[i](idx).squeeze(1)  # (batch,)
            linear_terms.append(linear_i)
            factor_i = self.factors[i](idx)  # (batch, k)
            factor_list.append(factor_i)
        linear_sum = torch.stack(linear_terms, dim=1).sum(dim=1)  # (batch,)
        # interaction term: 0.5 * (sum(v))^2 - sum(v^2)
        # sum_of_vectors: (batch, k)
        sum_v = torch.stack(factor_list, dim=0).sum(dim=0)  # (batch, k)
        sum_v_square = sum_v * sum_v  # (batch, k)
        sum_of_sq = torch.stack([v * v for v in factor_list], dim=0).sum(dim=0)  # (batch, k)
        interaction = 0.5 * (sum_v_square.sum(dim=1) - sum_of_sq.sum(dim=1))  # (batch,)
        logits = self.bias + linear_sum + interaction
        return logits.squeeze(0) if logits.dim() == 0 else logits

# Instantiate model
model = FM(field_sizes=embed_sizes, k=K).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop (simple minibatch)
num_batches = int(np.ceil(n_train / BATCH_SIZE))
perm = np.arange(n_train)

for epoch in range(EPOCHS):
    np.random.shuffle(perm)
    # iterate
    for i in range(num_batches):
        lo = i * BATCH_SIZE
        hi = min(n_train, (i+1) * BATCH_SIZE)
        idx = perm[lo:hi]
        xb = X_train_t[idx]
        yb = y_train_t[idx]
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        # gradient clipping to stabilize on CPU
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    # optional brief evaluation each epoch (not used for early stopping here)
    # compute train-loss approx on a small sample to see progress
    with torch.no_grad():
        sample_idx = np.random.choice(n_train, min(10000, n_train), replace=False)
        logits_sample = model(X_train_t[sample_idx])
        loss_sample = criterion(logits_sample, y_train_t[sample_idx]).item()
    print(f"Epoch {epoch+1}/{EPOCHS} sample_train_loss={loss_sample:.6f}", flush=True)

# Predict on validation (produce numpy array of floats)
model.eval()
with torch.no_grad():
    # Predict in chunks to moderate memory use
    valid_preds = []
    CH = 200_000
    for sidx in range(0, n_valid, CH):
        eidx = min(n_valid, sidx+CH)
        xb = X_valid_t[sidx:eidx]
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        valid_preds.append(probs)
    valid_scores = np.concatenate(valid_preds, axis=0).astype(np.float64)

# Evaluate using organizer's evaluate
eval_res = evaluate(valid.user_id, valid.y, valid_scores)
primary = float(eval_res["primary"])
gauc = float(eval_res["gauc"])
ndcg5 = float(eval_res["ndcg@5"])

# Save valid scores per contract
out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "scores_valid.npy"), np.asarray(valid_scores, dtype=np.float64))

# Score test split and save predictions (do NOT access test.y)
te = load("test")
# prepare test matrix (same fields)
def build_test_matrix(split, fields):
    arrs = []
    for f in fields:
        a = split.X[f].astype(np.int64)
        arrs.append(a)
    return np.stack(arrs, axis=1)

X_test = build_test_matrix(te, FIELDS)
n_test = X_test.shape[0]
X_test_t = torch.from_numpy(X_test).to(device)

model.eval()
with torch.no_grad():
    test_preds_parts = []
    CH = 200_000
    for sidx in range(0, n_test, CH):
        eidx = min(n_test, sidx+CH)
        xb = X_test_t[sidx:eidx]
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        test_preds_parts.append(probs)
    test_scores = np.concatenate(test_preds_parts, axis=0).astype(np.float64)

if out_dir:
    np.save(os.path.join(out_dir, "scores_test.npy"), np.asarray(test_scores, dtype=np.float64))

elapsed = time.time() - start_time

# Final required METRICS line (exactly this format)
metrics = {"primary": primary, "gauc": gauc, "ndcg@5": ndcg5, "gpu_seconds": float(elapsed)}
print("METRICS " + json.dumps(metrics), flush=True)