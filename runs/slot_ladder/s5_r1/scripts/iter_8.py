import os
import time
import json
import random
import numpy as np
import torch
from torch import nn
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

DEVICE = torch.device("cpu")
PRED_BATCH = 65536


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    repeated_starts = np.repeat(starts, sizes)
    denominators = np.repeat(np.maximum(sizes - 1, 1), sizes)
    sorted_rank = (
        np.arange(n, dtype=np.float64) - repeated_starts
    ) / denominators

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_rank
    return result


class SideAwareBPR(nn.Module):
    """
    Pairwise latent ranker. A user vector scores the sum of video, author,
    and tag vectors, with separate side-entity biases.
    """

    def __init__(self, rank=40):
        super().__init__()
        self.user = nn.Embedding(
            int(FEATURE_CARDINALITIES["user_id"]), rank
        )
        self.video = nn.Embedding(
            int(FEATURE_CARDINALITIES["video_id"]), rank
        )
        self.author = nn.Embedding(
            int(FEATURE_CARDINALITIES["author_id"]), rank
        )
        self.tag = nn.Embedding(
            int(FEATURE_CARDINALITIES["tag"]), rank
        )
        self.video_bias = nn.Embedding(
            int(FEATURE_CARDINALITIES["video_id"]), 1
        )
        self.author_bias = nn.Embedding(
            int(FEATURE_CARDINALITIES["author_id"]), 1
        )
        self.tag_bias = nn.Embedding(
            int(FEATURE_CARDINALITIES["tag"]), 1
        )

        for emb in (self.user, self.video, self.author, self.tag):
            nn.init.normal_(emb.weight, mean=0.0, std=0.025)
        for emb in (self.video_bias, self.author_bias, self.tag_bias):
            nn.init.zeros_(emb.weight)

    def score(self, user, video, author, tag):
        u = self.user(user)
        item = (
            self.video(video)
            + 0.55 * self.author(author)
            + 0.35 * self.tag(tag)
        )
        interaction = (u * item).sum(dim=-1)
        bias = (
            self.video_bias(video).squeeze(-1)
            + 0.55 * self.author_bias(author).squeeze(-1)
            + 0.35 * self.tag_bias(tag).squeeze(-1)
        )
        return interaction + bias

    def forward(
        self,
        user,
        pos_video,
        pos_author,
        pos_tag,
        neg_video,
        neg_author,
        neg_tag,
    ):
        positive = self.score(user, pos_video, pos_author, pos_tag)
        negative = self.score(user, neg_video, neg_author, neg_tag)
        return positive, negative


def prepare_bpr_sampling(train):
    y = np.asarray(train.y)
    users = np.asarray(train.X["user_id"], dtype=np.int64)

    positive_rows = np.flatnonzero(y == 1)
    negative_rows = np.flatnonzero(y == 0)

    neg_users = users[negative_rows]
    neg_order = np.argsort(neg_users, kind="stable")
    negative_rows = negative_rows[neg_order]
    neg_users = neg_users[neg_order]

    unique_neg_users, starts, counts = np.unique(
        neg_users, return_index=True, return_counts=True
    )
    positive_users = users[positive_rows]
    locations = np.searchsorted(unique_neg_users, positive_users)
    locations_clipped = np.minimum(
        locations, max(0, len(unique_neg_users) - 1)
    )
    valid = (
        (locations < len(unique_neg_users))
        & (unique_neg_users[locations_clipped] == positive_users)
    )

    positive_rows = positive_rows[valid]
    positive_users = positive_users[valid]
    locations = locations[valid]

    return {
        "positive_rows": positive_rows,
        "positive_users": positive_users,
        "negative_rows": negative_rows,
        "negative_starts": starts[locations],
        "negative_counts": counts[locations],
    }


def train_bpr(train, epochs=6, batch_size=8192):
    sampling = prepare_bpr_sampling(train)
    pos_rows = sampling["positive_rows"]
    pos_users = sampling["positive_users"]
    neg_rows_grouped = sampling["negative_rows"]
    neg_starts = sampling["negative_starts"]
    neg_counts = sampling["negative_counts"]

    video = np.asarray(train.X["video_id"], dtype=np.int64)
    author = np.asarray(train.X["author_id"], dtype=np.int64)
    tag = np.asarray(train.X["tag"], dtype=np.int64)

    model = SideAwareBPR(rank=40).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0022, weight_decay=2e-6
    )
    rng = np.random.default_rng(SEED + 17)
    n = len(pos_rows)

    model.train()
    for epoch in range(epochs):
        sampled_offsets = (
            rng.random(n) * neg_counts
        ).astype(np.int64)
        neg_rows = neg_rows_grouped[neg_starts + sampled_offsets]
        order = rng.permutation(n)

        for begin in range(0, n, batch_size):
            sel = order[begin:begin + batch_size]
            pr = pos_rows[sel]
            nr = neg_rows[sel]

            user_t = torch.from_numpy(pos_users[sel]).long()
            pv = torch.from_numpy(video[pr]).long()
            pa = torch.from_numpy(author[pr]).long()
            pt = torch.from_numpy(tag[pr]).long()
            nv = torch.from_numpy(video[nr]).long()
            na = torch.from_numpy(author[nr]).long()
            nt = torch.from_numpy(tag[nr]).long()

            positive, negative = model(
                user_t, pv, pa, pt, nv, na, nt
            )
            pair_loss = torch.nn.functional.softplus(
                negative - positive
            ).mean()

            regularizer = 1e-5 * (
                model.user(user_t).pow(2).mean()
                + model.video(pv).pow(2).mean()
                + model.video(nv).pow(2).mean()
            )
            loss = pair_loss + regularizer

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict_bpr(model, split):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    authors = np.asarray(split.X["author_id"], dtype=np.int64)
    tags = np.asarray(split.X["tag"], dtype=np.int64)

    result = np.empty(len(users), dtype=np.float64)
    model.eval()
    for begin in range(0, len(users), PRED_BATCH):
        end = min(begin + PRED_BATCH, len(users))
        result[begin:end] = model.score(
            torch.from_numpy(users[begin:end]).long(),
            torch.from_numpy(videos[begin:end]).long(),
            torch.from_numpy(authors[begin:end]).long(),
            torch.from_numpy(tags[begin:end]).long(),
        ).numpy().astype(np.float64)
    return result


class PositiveSVD:
    """
    Low-rank reconstruction of the binary user-video positive matrix.
    Repeated impressions are collapsed, preventing prolific videos from
    dominating solely through duplicate logs.
    """

    def __init__(self, rank=48):
        self.rank = rank
        self.user_factors = None
        self.item_factors = None
        self.item_prior = None

    def fit(self, train):
        users = np.asarray(train.X["user_id"], dtype=np.int64)
        videos = np.asarray(train.X["video_id"], dtype=np.int64)
        positives = np.asarray(train.y) == 1

        n_users = int(FEATURE_CARDINALITIES["user_id"])
        n_videos = int(FEATURE_CARDINALITIES["video_id"])

        matrix = sparse.coo_matrix(
            (
                np.ones(int(positives.sum()), dtype=np.float32),
                (users[positives], videos[positives]),
            ),
            shape=(n_users, n_videos),
            dtype=np.float32,
        ).tocsr()
        matrix.sum_duplicates()
        matrix.data[:] = 1.0

        item_frequency = np.asarray(matrix.sum(axis=0)).ravel()
        self.item_prior = np.log1p(item_frequency).astype(np.float32)

        u, singular_values, vt = svds(
            matrix,
            k=self.rank,
            which="LM",
            random_state=SEED,
        )
        order = np.argsort(singular_values)[::-1]
        singular_values = singular_values[order]
        u = u[:, order]
        vt = vt[order]

        root_s = np.sqrt(np.maximum(singular_values, 1e-8))
        self.user_factors = (
            u * root_s[None, :]
        ).astype(np.float32)
        self.item_factors = (
            vt.T * root_s[None, :]
        ).astype(np.float32)
        return self

    def predict(self, split):
        users = np.asarray(split.X["user_id"], dtype=np.int64)
        videos = np.asarray(split.X["video_id"], dtype=np.int64)
        latent = np.sum(
            self.user_factors[users] * self.item_factors[videos],
            axis=1,
        )
        return (
            latent + 0.015 * self.item_prior[videos]
        ).astype(np.float64)


def chronological_predecessor(split):
    n = len(split.user_id)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort(
        (
            rows,
            np.asarray(split.time_ms, dtype=np.int64),
            np.asarray(split.user_id, dtype=np.int64),
        )
    )

    sorted_users = np.asarray(split.user_id, dtype=np.int64)[order]
    has_previous = np.r_[
        False, sorted_users[1:] == sorted_users[:-1]
    ]

    predecessor = np.full(n, -1, dtype=np.int64)
    current_rows = order[has_previous]
    previous_rows = order[np.flatnonzero(has_previous) - 1]
    predecessor[current_rows] = previous_rows
    return predecessor


def fit_rate_table(keys, labels, smoothing, global_rate):
    keys = np.asarray(keys, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    sums = np.bincount(
        inverse, weights=labels, minlength=len(unique_keys)
    )
    rates = (
        sums + smoothing * global_rate
    ) / (counts + smoothing)
    return unique_keys, rates.astype(np.float32)


def lookup_rate(keys, table_keys, table_rates, default):
    keys = np.asarray(keys, dtype=np.int64)
    locations = np.searchsorted(table_keys, keys)
    clipped = np.minimum(locations, max(0, len(table_keys) - 1))
    found = (
        (locations < len(table_keys))
        & (table_keys[clipped] == keys)
    )
    result = np.full(len(keys), default, dtype=np.float64)
    result[found] = table_rates[clipped[found]]
    return result


class TransitionTargetModel:
    """
    Empirical-Bayes sequence model using only the identities of earlier
    impressions, never their evaluation outcomes.
    """

    def __init__(self):
        self.global_rate = None
        self.video_table = None
        self.tag_pair_table = None
        self.author_tag_table = None
        self.video_tag_table = None

    def fit(self, train):
        y = np.asarray(train.y, dtype=np.float64)
        self.global_rate = float(y.mean())

        video = np.asarray(train.X["video_id"], dtype=np.int64)
        author = np.asarray(train.X["author_id"], dtype=np.int64)
        tag = np.asarray(train.X["tag"], dtype=np.int64)
        predecessor = chronological_predecessor(train)

        self.video_table = fit_rate_table(
            video, y, smoothing=35.0, global_rate=self.global_rate
        )

        rows = np.flatnonzero(predecessor >= 0)
        previous = predecessor[rows]

        tag_card = int(FEATURE_CARDINALITIES["tag"])
        author_card = int(FEATURE_CARDINALITIES["author_id"])

        tag_pair_key = tag[previous] * tag_card + tag[rows]
        author_tag_key = author[previous] * tag_card + tag[rows]
        video_tag_key = video[previous] * tag_card + tag[rows]

        self.tag_pair_table = fit_rate_table(
            tag_pair_key,
            y[rows],
            smoothing=30.0,
            global_rate=self.global_rate,
        )
        self.author_tag_table = fit_rate_table(
            author_tag_key,
            y[rows],
            smoothing=18.0,
            global_rate=self.global_rate,
        )
        self.video_tag_table = fit_rate_table(
            video_tag_key,
            y[rows],
            smoothing=16.0,
            global_rate=self.global_rate,
        )
        return self

    def predict(self, split):
        video = np.asarray(split.X["video_id"], dtype=np.int64)
        author = np.asarray(split.X["author_id"], dtype=np.int64)
        tag = np.asarray(split.X["tag"], dtype=np.int64)
        predecessor = chronological_predecessor(split)

        video_rate = lookup_rate(
            video,
            self.video_table[0],
            self.video_table[1],
            self.global_rate,
        )
        result = video_rate.copy()

        rows = np.flatnonzero(predecessor >= 0)
        if len(rows) == 0:
            return result

        previous = predecessor[rows]
        tag_card = int(FEATURE_CARDINALITIES["tag"])

        tag_pair_key = tag[previous] * tag_card + tag[rows]
        author_tag_key = author[previous] * tag_card + tag[rows]
        video_tag_key = video[previous] * tag_card + tag[rows]

        tag_rate = lookup_rate(
            tag_pair_key,
            self.tag_pair_table[0],
            self.tag_pair_table[1],
            self.global_rate,
        )
        author_tag_rate = lookup_rate(
            author_tag_key,
            self.author_tag_table[0],
            self.author_tag_table[1],
            self.global_rate,
        )
        video_tag_rate = lookup_rate(
            video_tag_key,
            self.video_tag_table[0],
            self.video_tag_table[1],
            self.global_rate,
        )

        result[rows] += (
            0.32 * (tag_rate - self.global_rate)
            + 0.24 * (author_tag_rate - self.global_rate)
            + 0.20 * (video_tag_rate - self.global_rate)
        )
        return result


train = load("train")
valid = load("valid")

bpr_model = train_bpr(train)
svd_model = PositiveSVD(rank=48).fit(train)
transition_model = TransitionTargetModel().fit(train)

valid_own = {
    "bpr": predict_bpr(bpr_model, valid),
    "positive_svd": svd_model.predict(valid),
    "transition_target": transition_model.predict(valid),
}

valid_own_ranks = {
    name: within_user_rank(valid.user_id, scores)
    for name, scores in valid_own.items()
}
valid_own_ranks["latent_sequence_ensemble"] = np.mean(
    np.column_stack(
        [
            valid_own_ranks["bpr"],
            valid_own_ranks["positive_svd"],
            valid_own_ranks["transition_target"],
        ]
    ),
    axis=1,
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not os.path.exists(incumbent_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores missing")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
incumbent_valid_rank = within_user_rank(
    valid.user_id, incumbent_valid
)

candidate_scores = {}
candidate_metrics = {}
candidate_recipes = {}
candidate_raw = {}

for name, rank_scores in valid_own_ranks.items():
    standalone_metric = evaluate(
        valid.user_id, valid.y, rank_scores
    )
    candidate_scores[name] = rank_scores
    candidate_metrics[name] = float(
        standalone_metric["primary"]
    )
    candidate_recipes[name] = ("standalone", name, 1.0)
    candidate_raw[name] = rank_scores

    for own_weight in (0.10, 0.20, 0.30, 0.40, 0.50):
        candidate_name = f"{name}_rankblend_w{own_weight:.2f}"
        blended = (
            own_weight * rank_scores
            + (1.0 - own_weight) * incumbent_valid_rank
        )
        metric = evaluate(valid.user_id, valid.y, blended)
        candidate_scores[candidate_name] = blended
        candidate_metrics[candidate_name] = float(
            metric["primary"]
        )
        candidate_recipes[candidate_name] = (
            "rankblend",
            name,
            own_weight,
        )
        candidate_raw[candidate_name] = rank_scores

winner_name = max(candidate_metrics, key=candidate_metrics.get)
valid_scores = candidate_scores[winner_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_metrics.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
)

print(
    "FINDINGS "
    + json.dumps(
        {
            "bpr_standalone": candidate_metrics["bpr"],
            "positive_svd_standalone": candidate_metrics["positive_svd"],
            "transition_target_standalone": candidate_metrics[
                "transition_target"
            ],
            "latent_sequence_ensemble_standalone": candidate_metrics[
                "latent_sequence_ensemble"
            ],
            "winner": winner_name,
        },
        separators=(",", ":"),
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if candidate_recipes[winner_name][0] != "standalone":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(
                candidate_raw[winner_name], dtype=np.float64
            ),
        )

test = load("test")
test_own = {
    "bpr": predict_bpr(bpr_model, test),
    "positive_svd": svd_model.predict(test),
    "transition_target": transition_model.predict(test),
}
test_own_ranks = {
    name: within_user_rank(test.user_id, scores)
    for name, scores in test_own.items()
}
test_own_ranks["latent_sequence_ensemble"] = np.mean(
    np.column_stack(
        [
            test_own_ranks["bpr"],
            test_own_ranks["positive_svd"],
            test_own_ranks["transition_target"],
        ]
    ),
    axis=1,
)

recipe_type, recipe_name, own_weight = candidate_recipes[
    winner_name
]
own_test_scores = test_own_ranks[recipe_name]

if recipe_type == "standalone":
    test_scores = own_test_scores
else:
    if not os.path.exists(incumbent_test_path):
        raise FileNotFoundError(
            "Trusted incumbent test scores missing"
        )
    incumbent_test = np.asarray(
        np.load(incumbent_test_path), dtype=np.float64
    )
    incumbent_test_rank = within_user_rank(
        test.user_id, incumbent_test
    )
    test_scores = (
        own_weight * own_test_scores
        + (1.0 - own_weight) * incumbent_test_rank
    )

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
        },
        separators=(",", ":"),
    )
)