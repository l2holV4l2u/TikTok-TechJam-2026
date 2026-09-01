import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 2048
PRED_BATCH_SIZE = 32768
EPOCHS = 10

GRAPH_DIM = 32
GRAPH_LAYERS = 2
GRAPH_EPOCHS = 20
GRAPH_LR = 0.025
GRAPH_WEIGHT_DECAY = 1e-6
GRAPH_CHECKPOINT_EPOCHS = {4, 8, 12, 16, 20}
BLEND_ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))

USER_CARD = int(FEATURE_CARDINALITIES["user_id"])
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])
AUTHOR_CARD = int(FEATURE_CARDINALITIES["author_id"])
VIDEO_OFFSET = USER_CARD
AUTHOR_OFFSET = USER_CARD + VIDEO_CARD
GRAPH_NODE_COUNT = USER_CARD + VIDEO_CARD + AUTHOR_CARD


def make_fm_features(split):
    x = np.stack(
        [np.asarray(split.X[f], dtype=np.int64) for f in FIELDS],
        axis=1,
    )
    x += offsets[None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


class FactorizationMachine(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim + 1)
        self.bias = nn.Parameter(torch.zeros(()))

        with torch.no_grad():
            self.embedding.weight[:, :embedding_dim].normal_(
                mean=0.0, std=0.01
            )
            self.embedding.weight[:, embedding_dim].zero_()

    def forward(self, x):
        parameters = self.embedding(x)
        factors = parameters[:, :, :EMBED_DIM]
        linear = parameters[:, :, EMBED_DIM].sum(dim=1)

        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)

        return self.bias + linear + interaction


def predict_fm(model, x):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, x.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, x.shape[0])
            result[start:end] = model(x[start:end]).cpu().numpy()
    return result


def build_normalized_graph(train):
    labels = np.asarray(train.y)
    positive = labels == 1

    users = np.asarray(train.X["user_id"], dtype=np.int64)[positive]
    videos = (
        np.asarray(train.X["video_id"], dtype=np.int64)[positive]
        + VIDEO_OFFSET
    )
    authors = (
        np.asarray(train.X["author_id"], dtype=np.int64)[positive]
        + AUTHOR_OFFSET
    )

    src = np.concatenate([users, videos, users, authors])
    dst = np.concatenate([videos, users, authors, users])

    indices = torch.from_numpy(
        np.ascontiguousarray(np.stack([src, dst], axis=0))
    )
    values = torch.ones(indices.shape[1], dtype=torch.float32)

    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        (GRAPH_NODE_COUNT, GRAPH_NODE_COUNT),
    ).coalesce()

    # Treat repeated positive impressions as one preference edge so that
    # frequently logged pairs do not dominate propagation.
    indices = adjacency.indices()
    values = torch.ones(indices.shape[1], dtype=torch.float32)

    degrees = torch.zeros(GRAPH_NODE_COUNT, dtype=torch.float32)
    degrees.index_add_(0, indices[0], values)
    inverse_sqrt_degree = degrees.clamp_min(1.0).pow(-0.5)

    normalized_values = (
        inverse_sqrt_degree[indices[0]]
        * inverse_sqrt_degree[indices[1]]
    )

    return torch.sparse_coo_tensor(
        indices,
        normalized_values,
        (GRAPH_NODE_COUNT, GRAPH_NODE_COUNT),
    ).coalesce()


def propagate_graph(base_embeddings, adjacency):
    layers = [base_embeddings]
    current = base_embeddings
    for _ in range(GRAPH_LAYERS):
        current = torch.sparse.mm(adjacency, current)
        layers.append(current)
    return torch.stack(layers, dim=0).mean(dim=0)


def prepare_bpr_rows(train):
    labels = np.asarray(train.y)
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    authors = np.asarray(train.X["author_id"], dtype=np.int64)

    negative_mask = labels == 0
    negative_users = users[negative_mask]
    negative_videos = videos[negative_mask]
    negative_authors = authors[negative_mask]

    order = np.argsort(negative_users, kind="stable")
    sorted_negative_users = negative_users[order]
    sorted_negative_videos = negative_videos[order]
    sorted_negative_authors = negative_authors[order]

    negative_counts = np.bincount(
        sorted_negative_users,
        minlength=USER_CARD,
    ).astype(np.int64)
    negative_starts = np.empty(USER_CARD, dtype=np.int64)
    negative_starts[0] = 0
    np.cumsum(negative_counts[:-1], out=negative_starts[1:])

    positive_mask = (labels == 1) & (negative_counts[users] > 0)

    positive_users = users[positive_mask]
    positive_videos = videos[positive_mask]
    positive_authors = authors[positive_mask]

    return {
        "positive_users": torch.from_numpy(
            np.ascontiguousarray(positive_users)
        ),
        "positive_videos": torch.from_numpy(
            np.ascontiguousarray(positive_videos)
        ),
        "positive_authors": torch.from_numpy(
            np.ascontiguousarray(positive_authors)
        ),
        "negative_counts": torch.from_numpy(negative_counts),
        "negative_starts": torch.from_numpy(negative_starts),
        "sorted_negative_videos": torch.from_numpy(
            np.ascontiguousarray(sorted_negative_videos)
        ),
        "sorted_negative_authors": torch.from_numpy(
            np.ascontiguousarray(sorted_negative_authors)
        ),
    }


def graph_scores(final_embeddings, split):
    users = torch.from_numpy(
        np.ascontiguousarray(
            np.asarray(split.X["user_id"], dtype=np.int64)
        )
    )
    videos = torch.from_numpy(
        np.ascontiguousarray(
            np.asarray(split.X["video_id"], dtype=np.int64) + VIDEO_OFFSET
        )
    )
    authors = torch.from_numpy(
        np.ascontiguousarray(
            np.asarray(split.X["author_id"], dtype=np.int64) + AUTHOR_OFFSET
        )
    )

    result = np.empty(users.numel(), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, users.numel(), PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, users.numel())
            user_embedding = final_embeddings[users[start:end]]
            video_embedding = final_embeddings[videos[start:end]]
            author_embedding = final_embeddings[authors[start:end]]

            score = (
                (user_embedding * video_embedding).sum(dim=1)
                + (user_embedding * author_embedding).sum(dim=1)
            )
            result[start:end] = score.cpu().numpy()

    return result


train = load("train")
valid = load("valid")

x_train = make_fm_features(train)
x_valid = make_fm_features(valid)
train_y_np = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y)
valid_users = np.asarray(valid.user_id)
y_train = torch.from_numpy(train_y_np)

model = FactorizationMachine(total_cardinality, EMBED_DIM)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    foreach=True,
)

generator = torch.Generator()
generator.manual_seed(SEED)

best_fm_primary = -math.inf
best_fm_metrics = None
best_state = None
best_valid_fm = None
n_train = x_train.shape[0]

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(n_train, generator=generator)

    loss_sum = 0.0
    seen = 0

    for start in range(0, n_train, BATCH_SIZE):
        indices = permutation[start:start + BATCH_SIZE]
        xb = x_train[indices]
        yb = y_train[indices]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()

        batch_n = indices.numel()
        loss_sum += float(loss.detach()) * batch_n
        seen += batch_n

    valid_fm = predict_fm(model, x_valid)
    metrics = evaluate(valid_users, valid_y, valid_fm)

    print(
        f"fm_epoch={epoch + 1} "
        f"loss={loss_sum / seen:.6f} "
        f"primary={float(metrics['primary']):.6f} "
        f"gauc={float(metrics['gauc']):.6f} "
        f"ndcg@5={float(metrics['ndcg@5']):.6f}",
        flush=True,
    )

    if float(metrics["primary"]) > best_fm_primary:
        best_fm_primary = float(metrics["primary"])
        best_fm_metrics = {k: float(v) for k, v in metrics.items()}
        best_valid_fm = valid_fm.copy()
        best_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

model.load_state_dict(best_state)

adjacency = build_normalized_graph(train)
bpr = prepare_bpr_rows(train)

positive_users = bpr["positive_users"]
positive_videos = bpr["positive_videos"]
positive_authors = bpr["positive_authors"]
negative_counts = bpr["negative_counts"]
negative_starts = bpr["negative_starts"]
sorted_negative_videos = bpr["sorted_negative_videos"]
sorted_negative_authors = bpr["sorted_negative_authors"]

graph_base = nn.Parameter(
    torch.empty(GRAPH_NODE_COUNT, GRAPH_DIM, dtype=torch.float32)
)
with torch.no_grad():
    graph_base.normal_(mean=0.0, std=0.05)

graph_optimizer = torch.optim.Adam(
    [graph_base],
    lr=GRAPH_LR,
    weight_decay=GRAPH_WEIGHT_DECAY,
)

graph_generator = torch.Generator()
graph_generator.manual_seed(SEED + 17)

best_metrics = dict(best_fm_metrics)
best_scores = best_valid_fm.copy()
best_alpha = 0.0
best_graph_scale = 1.0
best_graph_embeddings = None
num_positive_pairs = positive_users.numel()

for epoch in range(1, GRAPH_EPOCHS + 1):
    graph_optimizer.zero_grad(set_to_none=True)

    final_embeddings = propagate_graph(graph_base, adjacency)

    counts = negative_counts[positive_users]
    random_fraction = torch.rand(
        num_positive_pairs,
        generator=graph_generator,
    )
    negative_positions = (
        negative_starts[positive_users]
        + torch.floor(random_fraction * counts).to(torch.int64)
    )

    negative_videos = sorted_negative_videos[negative_positions]
    negative_authors = sorted_negative_authors[negative_positions]

    user_embedding = final_embeddings[positive_users]
    positive_video_embedding = final_embeddings[
        positive_videos + VIDEO_OFFSET
    ]
    positive_author_embedding = final_embeddings[
        positive_authors + AUTHOR_OFFSET
    ]
    negative_video_embedding = final_embeddings[
        negative_videos + VIDEO_OFFSET
    ]
    negative_author_embedding = final_embeddings[
        negative_authors + AUTHOR_OFFSET
    ]

    positive_scores = (
        (user_embedding * positive_video_embedding).sum(dim=1)
        + (user_embedding * positive_author_embedding).sum(dim=1)
    )
    negative_scores = (
        (user_embedding * negative_video_embedding).sum(dim=1)
        + (user_embedding * negative_author_embedding).sum(dim=1)
    )

    graph_loss = F.softplus(negative_scores - positive_scores).mean()
    graph_loss.backward()
    graph_optimizer.step()

    print(
        f"graph_epoch={epoch} bpr_loss={float(graph_loss.detach()):.6f}",
        flush=True,
    )

    if epoch in GRAPH_CHECKPOINT_EPOCHS:
        with torch.inference_mode():
            checkpoint_embeddings = propagate_graph(
                graph_base,
                adjacency,
            ).detach()
            valid_graph = graph_scores(checkpoint_embeddings, valid)

        graph_scale = max(float(np.std(valid_graph)), 1e-6)
        normalized_graph = valid_graph / graph_scale

        for alpha in BLEND_ALPHAS:
            candidate_scores = (
                best_valid_fm + alpha * normalized_graph
            )
            candidate_metrics = evaluate(
                valid_users,
                valid_y,
                candidate_scores,
            )

            print(
                f"graph_checkpoint={epoch} "
                f"alpha={alpha:.2f} "
                f"primary={float(candidate_metrics['primary']):.6f} "
                f"gauc={float(candidate_metrics['gauc']):.6f} "
                f"ndcg@5={float(candidate_metrics['ndcg@5']):.6f}",
                flush=True,
            )

            if (
                float(candidate_metrics["primary"])
                > float(best_metrics["primary"])
            ):
                best_metrics = {
                    key: float(value)
                    for key, value in candidate_metrics.items()
                }
                best_scores = candidate_scores.copy()
                best_alpha = float(alpha)
                best_graph_scale = graph_scale
                best_graph_embeddings = checkpoint_embeddings.clone()

print(
    f"selected_graph_alpha={best_alpha:.2f} "
    f"graph_scale={best_graph_scale:.6f}",
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = make_fm_features(test)
    test_fm = predict_fm(model, x_test)

    if best_alpha == 0.0 or best_graph_embeddings is None:
        test_scores = test_fm
    else:
        test_graph = graph_scores(best_graph_embeddings, test)
        test_scores = (
            test_fm
            + best_alpha * (test_graph / best_graph_scale)
        )

    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

final_metrics = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(final_metrics))