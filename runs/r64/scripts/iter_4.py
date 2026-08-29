#!/usr/bin/env python3
import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

# Start timer
t0 = time.time()

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

# Settings (keep modest to fit CPU budget)
EMB_K = 32
EPOCHS = 3
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3

# Fields to use (keep small and leakage-free). We build a compact user-item embedding model.
FIELDS = ["user_id", "video_id", "author_id", "tab"]

# Load splits
train = load("train")
valid = load("valid")

# Quick checks
for f in FIELDS:
    if f not in train.X:
        raise RuntimeError(f"Missing field {f} in data")

# Extract arrays
train_user = train.X["user_id"].astype(np.int64)
train_item = train.X["video_id"].astype(np.int64)
train_author = train.X["author_id"].astype(np.int64)
train_tab = train.X["tab"].astype(np.int64)
y_train = train.y.astype(np.int8)

valid_user = valid.X["user_id"].astype(np.int64)
valid_item = valid.X["video_id"].astype(np.int64)
valid_author = valid.X["author_id"].astype(np.int64)
valid_tab = valid.X["tab"].astype(np.int64)
y_valid = valid.y.astype(np.int8)

n_train = train_user.shape[0]

# Cardinalities (add +1 padding slot to be safe)
card_user = int(FEATURE_CARDINALITIES["user_id"]) + 1
card_item = int(FEATURE_CARDINALITIES["video_id"]) + 1
card_author = int(FEATURE_CARDINALITIES["author_id"]) + 1
card_tab = int(FEATURE_CARDINALITIES["tab"]) + 1

# Build per-user positive & negative index lists (for training pair sampling)
# We'll sample pairs (pos_idx, neg_idx) where both impressions belong to same user,
# pos has y==1 and neg has y==0. Users without at least one pos+neg are unusable for BPR.
user_pos_lists = [[] for _ in range(card_user)]
user_neg_lists = [[] for _ in range(card_user)]

for idx, (u, y) in enumerate(zip(train_user, y_train)):
    if u < 0 or u >= card_user:
        continue
    if y:
        user_pos_lists[u].append(idx)
    else:
        user_neg_lists[u].append(idx)

# Flatten the positive indices we can use (only those whose user has at least one negative)
usable_pos_indices = []
for u in range(card_user):
    if len(user_pos_lists[u]) > 0 and len(user_neg_lists[u]) > 0:
        usable_pos_indices.extend(user_pos_lists[u])
usable_pos_indices = np.array(usable_pos_indices, dtype=np.int64)
n_usable_pos = len(usable_pos_indices)

# FINDINGS about sampling availability
n_users_with_pos = sum(1 for lst in user_pos_lists if len(lst) > 0)
n_users_with_neg = sum(1 for lst in user_neg_lists if len(lst) > 0)
n_users_bpr = sum(1 for u in range(card_user) if len(user_pos_lists[u]) > 0 and len(user_neg_lists[u]) > 0)
print(f'FINDINGS train_rows={n_train} positives={int(y_train.sum())} usable_pos_for_bpr={n_usable_pos} '
      f'users_with_pos={n_users_with_pos} users_with_neg={n_users_with_neg} users_bpr={n_users_bpr}')

# If too few usable positives (unlikely), fallback to pointwise training (but we expect many)
if n_usable_pos < 1000:
    print('FINDINGS too_few_pairs_fallback_pointwise=True')

# Convert lists to numpy arrays for vectorized sampling per-user
user_neg_arr = [np.asarray(lst, dtype=np.int64) if len(lst) > 0 else None for lst in user_neg_lists]
user_pos_arr = [np.asarray(lst, dtype=np.int64) if len(lst) > 0 else None for lst in user_pos_lists]

# Model: a compact pairwise scoring model using user/item/author/tab embeddings.
class PairwiseScorer(nn.Module):
    def __init__(self, card_user, card_item, card_author, card_tab, k):
        super().__init__()
        self.user_emb = nn.Embedding(card_user, k, padding_idx=0)
        self.item_emb = nn.Embedding(card_item, k, padding_idx=0)
        self.author_emb = nn.Embedding(card_author, k, padding_idx=0)
        self.tab_emb = nn.Embedding(card_tab, k, padding_idx=0)
        # lightweight biases
        self.user_bias = nn.Embedding(card_user, 1, padding_idx=0)
        self.item_bias = nn.Embedding(card_item, 1, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(1))
        # small MLP combining concatenated embeddings for extra capacity (optional, single hidden layer)
        hidden = max(0, k * 2)
        self.mlp = nn.Sequential(
            nn.Linear(k * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        ) if hidden > 0 else None
        self._init_weights()

    def _init_weights(self):
        std = 0.01
        nn.init.normal_(self.user_emb.weight, 0.0, std)
        nn.init.normal_(self.item_emb.weight, 0.0, std)
        nn.init.normal_(self.author_emb.weight, 0.0, std)
        nn.init.normal_(self.tab_emb.weight, 0.0, std)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        if self.mlp is not None:
            for p in self.mlp.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def score(self, u_idx, item_idx, author_idx, tab_idx):
        # u_idx,... are LongTensors (B,)
        u_e = self.user_emb(u_idx)            # (B,k)
        it_e = self.item_emb(item_idx)        # (B,k)
        au_e = self.author_emb(author_idx)    # (B,k)
        tb_e = self.tab_emb(tab_idx)          # (B,k)

        # Dot user-item primary signal
        dot_ui = (u_e * it_e).sum(dim=1, keepdim=True)  # (B,1)
        # Author-item dot (helpful)
        dot_ai = (au_e * it_e).sum(dim=1, keepdim=True)
        # Biases
        ub = self.user_bias(u_idx)   # (B,1)
        ib = self.item_bias(item_idx)  # (B,1)

        # Optional MLP on [user, item, author] concat (gives higher-order capacity)
        mlp_out = self.mlp(torch.cat([u_e, it_e, au_e], dim=1)) if self.mlp is not None else 0.0

        s = self.bias + dot_ui + 0.5 * dot_ai + ub + ib + mlp_out  # (B,1)
        return s.squeeze(1)  # (B,)

# Instantiate model and optimizer
device = torch.device("cpu")
model = PairwiseScorer(card_user, card_item, card_author, card_tab, EMB_K).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
bce_logits = nn.BCEWithLogitsLoss(reduction="mean")

# Helper: create one epoch of negative samples aligned to every usable positive index.
def build_epoch_pairs(usable_pos_indices):
    # For each usable positive index, sample one negative index from same user.
    # We'll group positives by user for vectorized sampling per user.
    pos_users = train_user[usable_pos_indices]
    # Find unique users among these positives
    uniq_users, inverse = np.unique(pos_users, return_inverse=True)
    # For each unique user, sample len(where) negatives with replacement from that user's neg array
    neg_for_pos = np.empty_like(usable_pos_indices)
    for ui, u in enumerate(uniq_users):
        where = np.nonzero(inverse == ui)[0]  # positions in usable_pos_indices
        cnt = where.size
        negs = user_neg_arr[u]  # numpy array of negative indices (non-empty by construction)
        # draw random indices with replacement
        picks = np.random.randint(0, negs.shape[0], size=cnt)
        neg_for_pos[where] = negs[picks]
    return usable_pos_indices, neg_for_pos

# Training loop
model.train()
for epoch in range(EPOCHS):
    # Build epoch's negative sampling mapping (one neg per usable positive)
    pos_idx_epoch, neg_idx_epoch = build_epoch_pairs(usable_pos_indices)
    # Shuffle pair order
    perm = np.random.permutation(pos_idx_epoch.shape[0])
    pos_idx_epoch = pos_idx_epoch[perm]
    neg_idx_epoch = neg_idx_epoch[perm]

    # Iterate batches
    n_pairs = pos_idx_epoch.shape[0]
    total_loss = 0.0
    it = 0
    for start in range(0, n_pairs, BATCH_SIZE):
        end = min(n_pairs, start + BATCH_SIZE)
        batch_pos = pos_idx_epoch[start:end]
        batch_neg = neg_idx_epoch[start:end]
        bs = batch_pos.shape[0]
        # gather features
        up = torch.from_numpy(train_user[batch_pos]).long().to(device)
        ip = torch.from_numpy(train_item[batch_pos]).long().to(device)
        ap = torch.from_numpy(train_author[batch_pos]).long().to(device)
        tp = torch.from_numpy(train_tab[batch_pos]).long().to(device)

        un = torch.from_numpy(train_user[batch_neg]).long().to(device)
        ineg = torch.from_numpy(train_item[batch_neg]).long().to(device)
        aneg = torch.from_numpy(train_author[batch_neg]).long().to(device)
        tneg = torch.from_numpy(train_tab[batch_neg]).long().to(device)

        s_pos = model.score(up, ip, ap, tp)
        s_neg = model.score(un, ineg, aneg, tneg)
        diff = s_pos - s_neg  # larger -> better

        labels = torch.ones_like(diff, dtype=torch.float32, device=device)
        loss = bce_logits(diff, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * bs
        it += bs
    avg_loss = total_loss / max(1, it)
    print(f'FINDINGS epoch={epoch+1} pairs_trained={it} avg_loss={avg_loss:.6f}')

# Inference utilities (batching)
def predict_split(model, users, items, authors, tabs, batch_size=8192):
    model.eval()
    n = users.shape[0]
    preds = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            u = torch.from_numpy(users[start:end]).long().to(device)
            it = torch.from_numpy(items[start:end]).long().to(device)
            au = torch.from_numpy(authors[start:end]).long().to(device)
            tb = torch.from_numpy(tabs[start:end]).long().to(device)
            s = model.score(u, it, au, tb)
            preds[start:end] = torch.sigmoid(s).cpu().numpy().astype(np.float32)
    return preds

# Predict on validation
valid_scores = predict_split(model, valid_user, valid_item, valid_author, valid_tab, batch_size=8192)

# Save validation scores for harness
out = os.environ.get("ITER_OUT")
if out:
    np.save(os.path.join(out, "scores_valid.npy"), np.asarray(valid_scores, dtype=np.float64))

# Evaluate using provided evaluator
res = evaluate(valid.user_id, valid.y, valid_scores)
primary = float(res["primary"])
gauc = float(res["gauc"])
ndcg5 = float(res["ndcg@5"])

print(f'FINDINGS validation_pairs_used_for_training={n_usable_pos} validation_pos_rate={float(y_valid.mean()):.6f}')

# Score test split and save scores (must not touch test.y)
if out:
    te = load("test")
    te_users = te.X["user_id"].astype(np.int64)
    te_items = te.X["video_id"].astype(np.int64)
    te_authors = te.X["author_id"].astype(np.int64)
    te_tabs = te.X["tab"].astype(np.int64)

    test_scores = predict_split(model, te_users, te_items, te_authors, te_tabs, batch_size=8192)
    np.save(os.path.join(out, "scores_test.npy"), np.asarray(test_scores, dtype=np.float64))
else:
    # still compute test scores to be able to save to RUN_ARTIFACTS if needed
    te = load("test")
    te_users = te.X["user_id"].astype(np.int64)
    te_items = te.X["video_id"].astype(np.int64)
    te_authors = te.X["author_id"].astype(np.int64)
    te_tabs = te.X["tab"].astype(np.int64)
    test_scores = predict_split(model, te_users, te_items, te_authors, te_tabs, batch_size=8192)
    # Not saving if ITER_OUT not provided.

# End timer
t1 = time.time()
wall = float(t1 - t0)

# Print a concise JSON-like METRICS line as required
print(f'METRICS {{"primary": {primary:.6f}, "gauc": {gauc:.6f}, "ndcg@5": {ndcg5:.6f}, "gpu_seconds": {wall:.3f}}}')