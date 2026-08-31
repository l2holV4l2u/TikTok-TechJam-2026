import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 8675309
DIM = 32
EPOCHS = 8
LR = 0.01
HALF_LIFE_DAYS = 4.0
PRED_BATCH = 65536

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))


def sigmoid_np(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def probability_scale(x):
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
        return np.clip(x, 1e-7, 1.0 - 1e-7)
    return sigmoid_np(x)


def make_recency_weights(train):
    dates = np.asarray(train.date, dtype=np.int64)
    unique_dates = np.sort(np.unique(dates))
    age_lookup = {
        int(date): len(unique_dates) - 1 - i
        for i, date in enumerate(unique_dates)
    }
    ages = np.fromiter(
        (age_lookup[int(d)] for d in dates),
        dtype=np.float32,
        count=dates.size,
    )
    weights = np.exp2(-ages / HALF_LIFE_DAYS).astype(np.float32)
    weights /= np.mean(weights)
    return weights


def build_normalized_graph(users, entities, edge_weights, n_users, n_entities):
    users = np.asarray(users, dtype=np.int64)
    entities = np.asarray(entities, dtype=np.int64)
    edge_weights = np.asarray(edge_weights, dtype=np.float32)

    entity_nodes = entities + n_users
    n_nodes = n_users + n_entities

    degree = np.bincount(
        np.concatenate([users, entity_nodes]),
        weights=np.concatenate([edge_weights, edge_weights]),
        minlength=n_nodes,
    ).astype(np.float64)
    inv_sqrt_degree = np.zeros(n_nodes, dtype=np.float64)
    positive_degree = degree > 0
    inv_sqrt_degree[positive_degree] = 1.0 / np.sqrt(
        degree[positive_degree]
    )

    normalized = (
        edge_weights.astype(np.float64)
        * inv_sqrt_degree[users]
        * inv_sqrt_degree[entity_nodes]
    ).astype(np.float32)

    rows = np.concatenate([users, entity_nodes])
    cols = np.concatenate([entity_nodes, users])
    values = np.concatenate([normalized, normalized])

    indices = torch.from_numpy(
        np.stack([rows, cols], axis=0).astype(np.int64, copy=False)
    )
    values_tensor = torch.from_numpy(values)
    graph = torch.sparse_coo_tensor(
        indices,
        values_tensor,
        size=(n_nodes, n_nodes),
        dtype=torch.float32,
    ).coalesce()
    return graph


def make_logged_negative_sampler(users, entities, labels, n_users):
    users = np.asarray(users, dtype=np.int64)
    entities = np.asarray(entities, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)

    negative_rows = np.flatnonzero(labels == 0)
    order = np.argsort(users[negative_rows], kind="stable")
    negative_rows = negative_rows[order]
    negative_users = users[negative_rows]

    counts = np.bincount(
        negative_users, minlength=n_users
    ).astype(np.int64)
    starts = np.zeros(n_users, dtype=np.int64)
    if n_users > 1:
        starts[1:] = np.cumsum(counts[:-1])

    positive_rows = np.flatnonzero(labels == 1)
    positive_users = users[positive_rows]
    usable = counts[positive_users] > 0
    positive_rows = positive_rows[usable]
    positive_users = positive_users[usable]

    return {
        "positive_rows": positive_rows,
        "positive_users": positive_users,
        "negative_rows": negative_rows,
        "counts": counts,
        "starts": starts,
        "entities": entities,
    }


class LightGCN(nn.Module):
    def __init__(self, n_users, n_entities, graph):
        super().__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.graph = graph

        self.user_embedding = nn.Embedding(n_users, DIM)
        self.entity_embedding = nn.Embedding(n_entities, DIM)
        self.entity_bias = nn.Embedding(n_entities, 1)

        nn.init.normal_(self.user_embedding.weight, std=0.08)
        nn.init.normal_(self.entity_embedding.weight, std=0.08)
        nn.init.zeros_(self.entity_bias.weight)

    def layers(self):
        base = torch.cat(
            [self.user_embedding.weight, self.entity_embedding.weight],
            dim=0,
        )
        layer1 = torch.sparse.mm(self.graph, base)
        layer2 = torch.sparse.mm(self.graph, layer1)
        return base, layer1, layer2

    def embeddings(self, depth=2):
        layers = self.layers()
        combined = torch.stack(layers[:depth + 1], dim=0).mean(dim=0)
        return combined[:self.n_users], combined[self.n_users:]

    def score_from_embeddings(self, user_ids, entity_ids, user_e, entity_e):
        return (
            torch.sum(user_e[user_ids] * entity_e[entity_ids], dim=1)
            + self.entity_bias(entity_ids).squeeze(1)
        )


def train_graph_model(
    users,
    entities,
    labels,
    recency_weights,
    n_users,
    n_entities,
    seed,
):
    positive_mask = labels == 1
    graph = build_normalized_graph(
        users[positive_mask],
        entities[positive_mask],
        recency_weights[positive_mask],
        n_users,
        n_entities,
    )
    sampler = make_logged_negative_sampler(
        users, entities, labels, n_users
    )

    positive_rows = sampler["positive_rows"]
    positive_users_np = sampler["positive_users"]
    negative_rows = sampler["negative_rows"]
    negative_counts = sampler["counts"]
    negative_starts = sampler["starts"]

    pos_users = torch.from_numpy(positive_users_np)
    pos_entities = torch.from_numpy(entities[positive_rows])
    pos_weights = torch.from_numpy(
        recency_weights[positive_rows].astype(np.float32, copy=False)
    )

    model = LightGCN(n_users, n_entities, graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(EPOCHS):
        offsets = (
            rng.random(positive_rows.size)
            * negative_counts[positive_users_np]
        ).astype(np.int64)
        sampled_negative_rows = negative_rows[
            negative_starts[positive_users_np] + offsets
        ]
        neg_entities = torch.from_numpy(entities[sampled_negative_rows])

        optimizer.zero_grad(set_to_none=True)
        user_e, entity_e = model.embeddings(depth=2)

        positive_scores = model.score_from_embeddings(
            pos_users, pos_entities, user_e, entity_e
        )
        negative_scores = model.score_from_embeddings(
            pos_users, neg_entities, user_e, entity_e
        )

        pair_loss = F.softplus(-(positive_scores - negative_scores))
        rank_loss = torch.sum(pair_loss * pos_weights) / torch.sum(pos_weights)

        regularization = 1e-5 * (
            model.user_embedding(pos_users).pow(2).mean()
            + model.entity_embedding(pos_entities).pow(2).mean()
            + model.entity_embedding(neg_entities).pow(2).mean()
        )
        loss = rank_loss + regularization
        loss.backward()
        optimizer.step()

    return model


def predict_graph(model, user_ids, entity_ids, depth):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    entity_ids = np.asarray(entity_ids, dtype=np.int64)
    output = np.empty(user_ids.size, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        user_e, entity_e = model.embeddings(depth=depth)
        for start in range(0, user_ids.size, PRED_BATCH):
            end = min(start + PRED_BATCH, user_ids.size)
            u = torch.from_numpy(user_ids[start:end])
            e = torch.from_numpy(entity_ids[start:end])
            output[start:end] = model.score_from_embeddings(
                u, e, user_e, entity_e
            ).cpu().numpy()
    return output


train = load("train")
labels = np.asarray(train.y, dtype=np.int8)
train_users = np.asarray(train.X["user_id"], dtype=np.int64)
train_videos = np.asarray(train.X["video_id"], dtype=np.int64)
train_authors = np.asarray(train.X["author_id"], dtype=np.int64)

n_users = int(FEATURE_CARDINALITIES["user_id"])
n_videos = int(FEATURE_CARDINALITIES["video_id"])
n_authors = int(FEATURE_CARDINALITIES["author_id"])

recency_weights = make_recency_weights(train)

video_model = train_graph_model(
    train_users,
    train_videos,
    labels,
    recency_weights,
    n_users,
    n_videos,
    SEED + 101,
)
author_model = train_graph_model(
    train_users,
    train_authors,
    labels,
    recency_weights,
    n_users,
    n_authors,
    SEED + 202,
)

valid = load("valid")
valid_users = np.asarray(valid.X["user_id"], dtype=np.int64)
valid_videos = np.asarray(valid.X["video_id"], dtype=np.int64)
valid_authors = np.asarray(valid.X["author_id"], dtype=np.int64)

video_base_valid = sigmoid_np(
    predict_graph(video_model, valid_users, valid_videos, depth=0)
)
video_l1_valid = sigmoid_np(
    predict_graph(video_model, valid_users, valid_videos, depth=1)
)
video_l2_valid = sigmoid_np(
    predict_graph(video_model, valid_users, valid_videos, depth=2)
)
author_l2_valid = sigmoid_np(
    predict_graph(author_model, valid_users, valid_authors, depth=2)
)

raw_valid = {
    "unpropagated_graph_control": video_base_valid,
    "lightgcn_user_video_l1": video_l1_valid,
    "lightgcn_user_video_l2": video_l2_valid,
    "lightgcn_heterogeneous_fusion": (
        0.72 * video_l2_valid + 0.28 * author_l2_valid
    ),
    "lightgcn_multiscale_fusion": (
        0.25 * video_l1_valid
        + 0.55 * video_l2_valid
        + 0.20 * author_l2_valid
    ),
}

candidate_scores = {}
candidate_specs = {}

for name, scores in raw_valid.items():
    metric = evaluate(valid.user_id, valid.y, scores)
    candidate_scores[name] = float(metric["primary"])
    candidate_specs[name] = {
        "raw_name": name,
        "alpha": 1.0,
        "scores": scores,
        "blended": False,
    }

shared_dir = os.environ.get("SHARED_ARTIFACTS")
inc_valid = None
inc_test_path = None
if shared_dir:
    valid_path = os.path.join(
        shared_dir, "incumbent_valid_scores.npy"
    )
    test_path = os.path.join(
        shared_dir, "incumbent_test_scores.npy"
    )
    if os.path.exists(valid_path) and os.path.exists(test_path):
        inc_valid = probability_scale(np.load(valid_path))
        inc_test_path = test_path

if inc_valid is not None:
    for name, scores in raw_valid.items():
        for alpha in (0.10, 0.25, 0.50, 0.75):
            blended = alpha * scores + (1.0 - alpha) * inc_valid
            candidate_name = f"{name}_incblend_{alpha:.2f}"
            metric = evaluate(valid.user_id, valid.y, blended)
            candidate_scores[candidate_name] = float(metric["primary"])
            candidate_specs[candidate_name] = {
                "raw_name": name,
                "alpha": alpha,
                "scores": blended,
                "blended": True,
            }

winner_name = max(candidate_scores, key=candidate_scores.get)
winner = candidate_specs[winner_name]
valid_scores = winner["scores"]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner_name,
            "positive_rows": int(np.sum(labels)),
            "video_graph_nodes": n_users + n_videos,
            "author_graph_nodes": n_users + n_authors,
            "recency_half_life_days": HALF_LIFE_DAYS,
            "recency_weight_min": float(recency_weights.min()),
            "recency_weight_max": float(recency_weights.max()),
        },
        sort_keys=True,
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner["blended"]:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(
                raw_valid[winner["raw_name"]], dtype=np.float64
            ),
        )

test = load("test")
test_users = np.asarray(test.X["user_id"], dtype=np.int64)
test_videos = np.asarray(test.X["video_id"], dtype=np.int64)
test_authors = np.asarray(test.X["author_id"], dtype=np.int64)

needed_name = winner["raw_name"]
if needed_name == "unpropagated_graph_control":
    raw_test_scores = sigmoid_np(
        predict_graph(video_model, test_users, test_videos, depth=0)
    )
elif needed_name == "lightgcn_user_video_l1":
    raw_test_scores = sigmoid_np(
        predict_graph(video_model, test_users, test_videos, depth=1)
    )
elif needed_name == "lightgcn_user_video_l2":
    raw_test_scores = sigmoid_np(
        predict_graph(video_model, test_users, test_videos, depth=2)
    )
elif needed_name == "lightgcn_heterogeneous_fusion":
    video_test = sigmoid_np(
        predict_graph(video_model, test_users, test_videos, depth=2)
    )
    author_test = sigmoid_np(
        predict_graph(author_model, test_users, test_authors, depth=2)
    )
    raw_test_scores = 0.72 * video_test + 0.28 * author_test
else:
    video_l1_test = sigmoid_np(
        predict_graph(video_model, test_users, test_videos, depth=1)
    )
    video_l2_test = sigmoid_np(
        predict_graph(video_model, test_users, test_videos, depth=2)
    )
    author_test = sigmoid_np(
        predict_graph(author_model, test_users, test_authors, depth=2)
    )
    raw_test_scores = (
        0.25 * video_l1_test
        + 0.55 * video_l2_test
        + 0.20 * author_test
    )

if winner["blended"]:
    incumbent_test = probability_scale(np.load(inc_test_path))
    alpha = winner["alpha"]
    test_scores = (
        alpha * raw_test_scores + (1.0 - alpha) * incumbent_test
    )
else:
    test_scores = raw_test_scores

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
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
        }
    )
)