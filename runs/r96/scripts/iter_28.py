import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 94217
THREADS = min(16, os.cpu_count() or 1)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

DEVICE = torch.device("cpu")
BATCH_SIZE = 8192
EPOCHS = 3
LATENT_DIM = 20
CAT_DIM = 12
HALF_LIFE = 4.0

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "hour",
    "upload_type",
    "music_type",
    "user_active_degree",
    "fans_user_num_range",
    "register_days_range",
    "is_video_author",
    "is_live_streamer",
    "onehot_feat3",
    "onehot_feat8",
]


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    ordered_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = ordered_users[1:] != ordered_users[:-1]
    start_indices = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = ordered_users[:-1] != ordered_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((
        np.asarray([-1], dtype=np.int64),
        end_positions,
    )))
    row_sizes = np.repeat(sizes, sizes)
    positions = np.arange(n, dtype=np.int64) - start_indices

    ranked = (
        positions.astype(np.float64) + 0.5
    ) / row_sizes.astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def categorical_matrix(split):
    return np.ascontiguousarray(
        np.stack([
            np.asarray(split.X[name], dtype=np.int64)
            for name in FIELDS
        ], axis=1),
        dtype=np.int64,
    )


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    weights = np.power(
        0.5,
        age.astype(np.float32) / HALF_LIFE,
    )
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def make_collective_factors(train, entity_field, rank):
    user = np.asarray(train.X["user_id"], dtype=np.int32)
    entity = np.asarray(train.X[entity_field], dtype=np.int32)
    labels = np.asarray(train.y, dtype=np.float32)

    n_user = int(FEATURE_CARDINALITIES["user_id"])
    n_entity = int(FEATURE_CARDINALITIES[entity_field])

    global_rate = float(labels.mean())

    # Positive residuals express affinity and negative residuals express
    # observed dislike. Duplicate user/entity observations are aggregated
    # by CSR construction.
    values = labels - global_rate
    matrix = sparse.coo_matrix(
        (values, (user, entity)),
        shape=(n_user, n_entity),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    # Temper very dense pairs so repeated logging cannot fully dominate
    # the collaborative geometry.
    counts = sparse.coo_matrix(
        (
            np.ones(len(user), dtype=np.float32),
            (user, entity),
        ),
        shape=(n_user, n_entity),
        dtype=np.float32,
    ).tocsr()
    counts.sum_duplicates()
    matrix.data /= np.sqrt(np.maximum(counts.data, 1.0))
    del counts

    u, singular, vt = svds(
        matrix,
        k=rank,
        which="LM",
        return_singular_vectors=True,
        random_state=SEED + (17 if entity_field == "video_id" else 31),
    )

    descending = np.argsort(singular)[::-1]
    singular = singular[descending]
    u = u[:, descending]
    vt = vt[descending, :]

    root = np.sqrt(np.maximum(singular, 1e-8))
    user_factors = u * root[None, :]
    entity_factors = vt.T * root[None, :]

    # Equalize dimensions for neural optimization while retaining matching
    # signs between the two sides.
    user_scale = np.maximum(
        np.std(user_factors, axis=0, keepdims=True), 1e-5
    )
    entity_scale = np.maximum(
        np.std(entity_factors, axis=0, keepdims=True), 1e-5
    )
    user_factors = user_factors / user_scale
    entity_factors = entity_factors / entity_scale

    user_factors[0] = 0.0
    entity_factors[0] = 0.0

    explained_proxy = float(
        np.sum(singular ** 2)
        / max(float(np.sum(matrix.data.astype(np.float64) ** 2)), 1e-12)
    )
    print("FINDINGS " + json.dumps({
        "spectral_view": entity_field,
        "rank": rank,
        "matrix_nnz": int(matrix.nnz),
        "top_singular": float(singular[0]),
        "rank_energy_fraction": explained_proxy,
    }, sort_keys=True))

    return (
        np.asarray(user_factors, dtype=np.float32),
        np.asarray(entity_factors, dtype=np.float32),
    )


class FrozenCollectiveViews(nn.Module):
    def __init__(self, uv_user, video, ua_user, author):
        super().__init__()
        self.uv_user = nn.Embedding.from_pretrained(
            torch.from_numpy(uv_user), freeze=True
        )
        self.video = nn.Embedding.from_pretrained(
            torch.from_numpy(video), freeze=True
        )
        self.ua_user = nn.Embedding.from_pretrained(
            torch.from_numpy(ua_user), freeze=True
        )
        self.author = nn.Embedding.from_pretrained(
            torch.from_numpy(author), freeze=True
        )

    def forward(self, x_cat):
        user_id = x_cat[:, 0]
        video_id = x_cat[:, 1]
        author_id = x_cat[:, 2]

        uv = self.uv_user(user_id) * self.video(video_id)
        ua = self.ua_user(user_id) * self.author(author_id)
        return torch.cat([uv, ua], dim=1)


class CategoricalBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                int(FEATURE_CARDINALITIES[field]),
                CAT_DIM,
            )
            for field in FIELDS
        ])
        self.linear = nn.ModuleList([
            nn.Embedding(
                int(FEATURE_CARDINALITIES[field]),
                1,
            )
            for field in FIELDS
        ])
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.025)
        for embedding in self.linear:
            nn.init.zeros_(embedding.weight)

    def embed(self, x_cat):
        return torch.stack([
            embedding(x_cat[:, index])
            for index, embedding in enumerate(self.embeddings)
        ], dim=1)

    def wide(self, x_cat):
        terms = [
            embedding(x_cat[:, index])
            for index, embedding in enumerate(self.linear)
        ]
        return torch.stack(terms, dim=1).sum(dim=1).squeeze(1)

    @staticmethod
    def fm_score(embedded):
        summed = embedded.sum(dim=1)
        return 0.5 * (
            summed.square().sum(dim=1)
            - embedded.square().sum(dim=(1, 2))
        )


class SpectralContextTower(CategoricalBase):
    """
    Fixed collective factors are impression features. A context tower
    transforms their elementwise affinity using content/exposure fields.
    """
    def __init__(self, factors):
        super().__init__()
        self.collective = FrozenCollectiveViews(*factors)
        input_dim = 2 * LATENT_DIM + len(FIELDS) * CAT_DIM
        self.network = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x_cat):
        embedded = self.embed(x_cat)
        spectral = self.collective(x_cat)
        features = torch.cat([
            spectral,
            embedded.flatten(start_dim=1),
        ], dim=1)
        return (
            self.wide(x_cat)
            + self.network(features).squeeze(1)
            + self.bias
        )


class GatedSpectralFM(CategoricalBase):
    """
    An FM and the two collaborative views form separate experts. Context
    embeddings choose the contribution of each expert per impression.
    """
    def __init__(self, factors):
        super().__init__()
        self.collective = FrozenCollectiveViews(*factors)
        self.spectral_experts = nn.Linear(
            2 * LATENT_DIM, 2, bias=True
        )
        context_dim = (len(FIELDS) - 3) * CAT_DIM
        self.gate = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        self.residual = nn.Sequential(
            nn.Linear(2 * LATENT_DIM + context_dim, 96),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(96, 1),
        )
        self.fm_scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x_cat):
        embedded = self.embed(x_cat)
        spectral = self.collective(x_cat)
        context = embedded[:, 3:, :].flatten(start_dim=1)

        spectral_scores = self.spectral_experts(spectral)
        fm_score = self.fm_scale * self.fm_score(embedded)
        experts = torch.cat([
            fm_score.unsqueeze(1),
            spectral_scores,
        ], dim=1)

        gate = torch.softmax(self.gate(context), dim=1)
        mixture = torch.sum(gate * experts, dim=1)

        residual = self.residual(
            torch.cat([spectral, context], dim=1)
        ).squeeze(1)

        return self.wide(x_cat) + mixture + residual + self.bias


class TensorCPModel(CategoricalBase):
    """
    CP-style third-order factors directly score user-video-context and
    user-author-context triples, unlike pairwise FM or spectral lookup.
    """
    def __init__(self):
        super().__init__()
        self.tab_factor = nn.Embedding(
            int(FEATURE_CARDINALITIES["tab"]), CAT_DIM
        )
        self.tag_factor = nn.Embedding(
            int(FEATURE_CARDINALITIES["tag"]), CAT_DIM
        )
        self.hour_factor = nn.Embedding(
            int(FEATURE_CARDINALITIES["hour"]), CAT_DIM
        )
        self.duration_factor = nn.Embedding(
            int(FEATURE_CARDINALITIES["duration_bucket"]), CAT_DIM
        )
        for embedding in (
            self.tab_factor,
            self.tag_factor,
            self.hour_factor,
            self.duration_factor,
        ):
            nn.init.normal_(embedding.weight, std=0.08)

        self.tensor_weights = nn.Parameter(torch.ones(4))
        self.pair_scale = nn.Parameter(torch.tensor(0.5))
        self.context_mlp = nn.Sequential(
            nn.Linear((len(FIELDS) - 3) * CAT_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x_cat):
        embedded = self.embed(x_cat)
        user = embedded[:, 0, :]
        video = embedded[:, 1, :]
        author = embedded[:, 2, :]

        tab = self.tab_factor(x_cat[:, FIELDS.index("tab")])
        tag = self.tag_factor(x_cat[:, FIELDS.index("tag")])
        hour = self.hour_factor(x_cat[:, FIELDS.index("hour")])
        duration = self.duration_factor(
            x_cat[:, FIELDS.index("duration_bucket")]
        )

        triples = torch.stack([
            (user * video * tab).sum(dim=1),
            (user * video * hour).sum(dim=1),
            (user * author * tag).sum(dim=1),
            (user * video * duration).sum(dim=1),
        ], dim=1)

        tensor_score = torch.sum(
            triples * self.tensor_weights.unsqueeze(0), dim=1
        )
        context = embedded[:, 3:, :].flatten(start_dim=1)

        return (
            self.wide(x_cat)
            + self.pair_scale * self.fm_score(embedded)
            + tensor_score
            + self.context_mlp(context).squeeze(1)
            + self.bias
        )


def train_model(model, x_cat, labels, weights, seed, name):
    torch.manual_seed(seed)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=0.0024,
        weight_decay=2e-5,
    )

    labels_tensor = torch.from_numpy(labels)
    weights_tensor = torch.from_numpy(weights)
    n = len(labels)

    for epoch in range(EPOCHS):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(seed + 1009 * epoch)
        permutation = torch.randperm(n, generator=generator)

        total_loss = 0.0
        total_rows = 0

        for start in range(0, n, BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE].numpy()
            batch_x = torch.from_numpy(x_cat[indices]).to(DEVICE)
            batch_y = labels_tensor[indices].to(DEVICE)
            batch_w = weights_tensor[indices].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            losses = F.binary_cross_entropy_with_logits(
                logits, batch_y, reduction="none"
            )
            loss = torch.sum(losses * batch_w) / torch.sum(batch_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()

            total_loss += float(loss.detach()) * len(indices)
            total_rows += len(indices)

        print("FINDINGS " + json.dumps({
            "model": name,
            "epoch": epoch + 1,
            "weighted_logloss": total_loss / max(total_rows, 1),
        }, sort_keys=True))

    return model


@torch.no_grad()
def predict_model(model, x_cat):
    model.eval()
    result = np.empty(len(x_cat), dtype=np.float32)

    for start in range(0, len(x_cat), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(x_cat))
        batch_x = torch.from_numpy(x_cat[start:end]).to(DEVICE)
        result[start:end] = (
            model(batch_x).cpu().numpy().astype(np.float32)
        )

    return result


train = load("train")
valid = load("valid")
test = load("test")

x_train = categorical_matrix(train)
x_valid = categorical_matrix(valid)
x_test = categorical_matrix(test)

labels_train = np.asarray(train.y, dtype=np.float32)
weights_train = recency_weights(train.date)

uv_user, video_factors = make_collective_factors(
    train, "video_id", LATENT_DIM
)
ua_user, author_factors = make_collective_factors(
    train, "author_id", LATENT_DIM
)
factors = (
    uv_user,
    video_factors,
    ua_user,
    author_factors,
)

model_constructors = [
    (
        "spectral_context_tower",
        lambda: SpectralContextTower(factors),
        SEED + 11,
    ),
    (
        "gated_spectral_fm",
        lambda: GatedSpectralFM(factors),
        SEED + 29,
    ),
    (
        "tensor_cp",
        TensorCPModel,
        SEED + 47,
    ),
]

own_valid = {}
own_test = {}

for name, constructor, seed in model_constructors:
    model = constructor()
    model = train_model(
        model,
        x_train,
        labels_train,
        weights_train,
        seed,
        name,
    )
    own_valid[name] = predict_model(model, x_valid)
    own_test[name] = predict_model(model, x_test)
    del model
    gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

candidate_valid = {"trusted_incumbent": inc_valid}
candidate_metrics = {
    "trusted_incumbent": evaluate(
        valid.user_id, valid.y, inc_valid
    )
}
candidate_specs = {
    "trusted_incumbent": ("incumbent", None)
}

valid_ranks = {}
test_ranks = {}

for name in own_valid:
    raw_valid = np.asarray(own_valid[name], dtype=np.float64)
    valid_ranks[name] = rank_percentile(
        valid.user_id, raw_valid
    )
    test_ranks[name] = rank_percentile(
        test.user_id, own_test[name]
    )

    candidate_valid[name] = raw_valid
    candidate_metrics[name] = evaluate(
        valid.user_id, valid.y, raw_valid
    )
    candidate_specs[name] = (name, None)

    for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
        candidate_name = f"{name}_incumbent_{alpha:.2f}"
        score = (
            alpha * valid_ranks[name]
            + (1.0 - alpha) * inc_valid_rank
        )
        candidate_valid[candidate_name] = score
        candidate_metrics[candidate_name] = evaluate(
            valid.user_id, valid.y, score
        )
        candidate_specs[candidate_name] = (name, alpha)

# The three mechanisms have distinct error sources, so also test their
# rank aggregate before combining it with the trusted incumbent.
ensemble_valid = np.mean(
    np.stack([
        valid_ranks[name] for name in own_valid
    ], axis=1),
    axis=1,
)
ensemble_test = np.mean(
    np.stack([
        test_ranks[name] for name in own_test
    ], axis=1),
    axis=1,
)

candidate_valid["spectral_tensor_ensemble"] = ensemble_valid
candidate_metrics["spectral_tensor_ensemble"] = evaluate(
    valid.user_id, valid.y, ensemble_valid
)
candidate_specs["spectral_tensor_ensemble"] = ("ensemble", None)

for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
    candidate_name = f"spectral_tensor_ensemble_incumbent_{alpha:.2f}"
    score = (
        alpha * ensemble_valid
        + (1.0 - alpha) * inc_valid_rank
    )
    candidate_valid[candidate_name] = score
    candidate_metrics[candidate_name] = evaluate(
        valid.user_id, valid.y, score
    )
    candidate_specs[candidate_name] = ("ensemble", alpha)

best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = np.asarray(candidate_valid[best_name], dtype=np.float64)
best_family, best_alpha = candidate_specs[best_name]

if best_family == "incumbent":
    best_test = inc_test
    raw_valid_for_audit = np.asarray(
        own_valid["gated_spectral_fm"], dtype=np.float64
    )
elif best_family == "ensemble":
    raw_valid_for_audit = ensemble_valid
    if best_alpha is None:
        best_test = ensemble_test
    else:
        best_test = (
            best_alpha * ensemble_test
            + (1.0 - best_alpha) * inc_test_rank
        )
else:
    raw_valid_for_audit = np.asarray(
        own_valid[best_family], dtype=np.float64
    )
    if best_alpha is None:
        best_test = np.asarray(
            own_test[best_family], dtype=np.float64
        )
    else:
        best_test = (
            best_alpha * test_ranks[best_family]
            + (1.0 - best_alpha) * inc_test_rank
        )

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "best_family": best_family,
    "best_blend_alpha": best_alpha,
    "spectral_rank": LATENT_DIM,
    "categorical_embedding_dim": CAT_DIM,
    "epochs": EPOCHS,
    "half_life_days": HALF_LIFE,
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_family == "incumbent" or best_alpha is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid_for_audit, dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))