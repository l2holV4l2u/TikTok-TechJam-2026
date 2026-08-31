import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 1847
DEVICE = "cpu"
K_HISTORY = 8
DIN_DIM = 16
DIN_EPOCHS = 2
BATCH_SIZE = 8192
LR = 0.009

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def within_user_ranks(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids
    ))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    group = np.cumsum(starts_mask) - 1
    local = np.arange(n, dtype=np.int64) - starts[group]
    sizes = np.diff(np.r_[starts, n])

    percentile = (
        local.astype(np.float64) + 0.5
    ) / sizes[group].astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = percentile
    return result


def combine_sources(a, b):
    class Combined:
        pass

    c = Combined()
    c.X = {
        name: np.concatenate([a.X[name], b.X[name]])
        for name in a.X
    }
    c.user_id = c.X["user_id"]
    c.video_id = c.X["video_id"]
    c.time_ms = np.concatenate([a.time_ms, b.time_ms])
    c.date = np.concatenate([a.date, b.date])
    return c


def chronological_order(source):
    n = len(source.user_id)
    return np.lexsort((
        np.arange(n, dtype=np.int64),
        np.asarray(source.time_ms, dtype=np.int64),
        np.asarray(source.user_id, dtype=np.int64)
    ))


def build_source_histories(source, y, fields, k=K_HISTORY):
    """
    For every source row, return the last k positive entities strictly
    preceding that row in (user_id, time_ms, row position) order.
    """
    n = len(y)
    order = chronological_order(source)
    users_sorted = np.asarray(source.user_id, dtype=np.int64)[order]
    y_sorted = np.asarray(y, dtype=np.int8)[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = users_sorted[1:] != users_sorted[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    cumulative = np.cumsum(y_sorted, dtype=np.int64)
    bases = cumulative[starts] - y_sorted[starts]
    base_per_row = np.repeat(bases, sizes)
    prior_positive_count = cumulative - y_sorted - base_per_row

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    positive_counts = np.bincount(
        users_sorted,
        weights=y_sorted.astype(np.int64),
        minlength=n_users
    ).astype(np.int64)
    positive_offsets = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(positive_counts, out=positive_offsets[1:])

    lags = np.arange(k, 0, -1, dtype=np.int64)
    indices = (
        positive_offsets[users_sorted, None] +
        prior_positive_count[:, None] -
        lags[None, :]
    )
    valid = (
        indices >= positive_offsets[users_sorted, None]
    ) & (
        indices < positive_offsets[users_sorted + 1, None]
    )

    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n, dtype=np.int64)

    histories = {}
    positive_mask = y_sorted == 1
    for field in fields:
        sorted_values = np.asarray(source.X[field], dtype=np.int32)[order]
        positive_values = sorted_values[positive_mask]
        hist_sorted = np.zeros((n, k), dtype=np.int32)
        good = valid
        hist_sorted[good] = positive_values[indices[good]]
        histories[field] = hist_sorted[inverse]

    state = {
        "positive_offsets": positive_offsets,
        "positive_counts": positive_counts,
        "positive_values": {}
    }
    for field in fields:
        sorted_values = np.asarray(source.X[field], dtype=np.int32)[order]
        state["positive_values"][field] = sorted_values[positive_mask]

    return histories, state


def build_target_histories(source_state, target, fields, k=K_HISTORY):
    users = np.asarray(target.user_id, dtype=np.int64)
    offsets = source_state["positive_offsets"]
    counts = source_state["positive_counts"]

    lags = np.arange(k, 0, -1, dtype=np.int64)
    indices = offsets[users, None] + counts[users, None] - lags[None, :]
    valid = (
        indices >= offsets[users, None]
    ) & (
        indices < offsets[users + 1, None]
    )

    histories = {}
    for field in fields:
        values = source_state["positive_values"][field]
        h = np.zeros((len(users), k), dtype=np.int32)
        h[valid] = values[indices[valid]]
        histories[field] = h
    return histories


class SequenceDIN(nn.Module):
    def __init__(self, mean_rate):
        super().__init__()
        nu = int(FEATURE_CARDINALITIES["user_id"])
        nv = int(FEATURE_CARDINALITIES["video_id"])
        na = int(FEATURE_CARDINALITIES["author_id"])
        nt = int(FEATURE_CARDINALITIES["tag"])
        nd = int(FEATURE_CARDINALITIES["duration_bucket"])
        d = DIN_DIM

        self.user = nn.Embedding(nu, d)
        self.video = nn.Embedding(nv, d, padding_idx=0)
        self.author = nn.Embedding(na, d, padding_idx=0)
        self.tag = nn.Embedding(nt, d, padding_idx=0)

        self.video_bias = nn.Embedding(nv, 1)
        self.author_bias = nn.Embedding(na, 1)
        self.tag_bias = nn.Embedding(nt, 1)
        self.duration_bias = nn.Embedding(nd, 1)
        self.user_bias = nn.Embedding(nu, 1)

        self.sequence_mlp = nn.Sequential(
            nn.Linear(6, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )
        self.global_bias = nn.Parameter(torch.tensor(
            math.log(mean_rate / (1.0 - mean_rate)),
            dtype=torch.float32
        ))

        with torch.no_grad():
            for emb in (self.user, self.video, self.author, self.tag):
                emb.weight.normal_(0.0, 0.04)
            self.video.weight[0].zero_()
            self.author.weight[0].zero_()
            self.tag.weight[0].zero_()
            for emb in (
                self.video_bias, self.author_bias, self.tag_bias,
                self.duration_bias, self.user_bias
            ):
                emb.weight.zero_()

    @staticmethod
    def attention_features(candidate, history, ids):
        # Candidate-conditioned attention over recent positive entities.
        scale = candidate.shape[-1] ** -0.5
        similarity = (history * candidate[:, None, :]).sum(dim=2) * scale
        mask = ids.ne(0)
        masked = similarity.masked_fill(~mask, -1e4)
        weight = torch.softmax(masked, dim=1) * mask.float()
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)

        attended = (weight[:, :, None] * history).sum(dim=1)
        attended_similarity = (attended * candidate).sum(dim=1) * scale
        maximum_similarity = similarity.masked_fill(~mask, -10.0).max(dim=1).values
        any_history = mask.any(dim=1)
        maximum_similarity = torch.where(
            any_history, maximum_similarity,
            torch.zeros_like(maximum_similarity)
        )
        return attended_similarity, maximum_similarity

    def forward(self, u, v, a, tag, duration, hv, ha, ht):
        ue = self.user(u)
        ve = self.video(v)
        ae = self.author(a)
        te = self.tag(tag)

        static = (
            (ue * ve).sum(dim=1) +
            0.55 * (ue * ae).sum(dim=1)
        ) / math.sqrt(DIN_DIM)

        v_att, v_max = self.attention_features(ve, self.video(hv), hv)
        a_att, a_max = self.attention_features(ae, self.author(ha), ha)
        t_att, t_max = self.attention_features(te, self.tag(ht), ht)
        sequence = self.sequence_mlp(torch.stack(
            [v_att, v_max, a_att, a_max, t_att, t_max], dim=1
        )).squeeze(1)

        bias = (
            self.user_bias(u).squeeze(1) +
            self.video_bias(v).squeeze(1) +
            self.author_bias(a).squeeze(1) +
            self.tag_bias(tag).squeeze(1) +
            self.duration_bias(duration).squeeze(1)
        )
        return self.global_bias + static + sequence + bias


def make_din_arrays(split, histories):
    return (
        np.asarray(split.X["user_id"], dtype=np.int32),
        np.asarray(split.X["video_id"], dtype=np.int32),
        np.asarray(split.X["author_id"], dtype=np.int32),
        np.asarray(split.X["tag"], dtype=np.int32),
        np.asarray(split.X["duration_bucket"], dtype=np.int32),
        histories["video_id"],
        histories["author_id"],
        histories["tag"],
    )


def fit_din(arrays, y):
    torch.manual_seed(SEED)
    model = SequenceDIN(float(np.mean(y))).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=2e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    tensors = tuple(torch.from_numpy(x) for x in arrays)
    target = torch.from_numpy(np.asarray(y, dtype=np.float32))
    n = len(y)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    for _ in range(DIN_EPOCHS):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:min(start + BATCH_SIZE, n)]
            batch = tuple(x[idx].long() for x in tensors)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch)
            loss = criterion(logits, target[idx])
            loss.backward()
            optimizer.step()
    return model


def predict_din(model, arrays):
    model.eval()
    tensors = tuple(torch.from_numpy(x) for x in arrays)
    result = np.empty(len(arrays[0]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(result), 32768):
            end = min(start + 32768, len(result))
            batch = tuple(x[start:end].long() for x in tensors)
            result[start:end] = model(*batch).cpu().numpy()
    return result


def entity_rate(ids, y, cardinality, strength=35.0):
    prior = float(np.mean(y))
    count = np.bincount(ids, minlength=cardinality).astype(np.float64)
    positive = np.bincount(
        ids, weights=np.asarray(y, dtype=np.float64),
        minlength=cardinality
    )
    return ((positive + strength * prior) / (count + strength)).astype(
        np.float32
    )


class SparseTransition:
    def __init__(self, keys, residuals):
        self.keys = keys
        self.residuals = residuals

    @staticmethod
    def fit(previous, current, y, parent_rate, cardinality, strength):
        keys = (
            previous.astype(np.int64, copy=False) * np.int64(cardinality) +
            current.astype(np.int64, copy=False)
        )
        order = np.argsort(keys, kind="mergesort")
        sorted_keys = keys[order]
        residual = (
            np.asarray(y, dtype=np.float64) -
            parent_rate[current].astype(np.float64)
        )[order]

        first = np.empty(len(keys), dtype=bool)
        first[0] = True
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
        starts = np.flatnonzero(first)
        unique = sorted_keys[starts]
        sums = np.add.reduceat(residual, starts)
        counts = np.diff(np.r_[starts, len(keys)])
        values = (sums / (counts + strength)).astype(np.float32)
        return SparseTransition(unique, values)

    def predict(self, previous, current, cardinality):
        keys = (
            previous.astype(np.int64, copy=False) * np.int64(cardinality) +
            current.astype(np.int64, copy=False)
        )
        loc = np.searchsorted(self.keys, keys)
        result = np.zeros(len(keys), dtype=np.float32)
        possible = loc < len(self.keys)
        ii = np.flatnonzero(possible)
        if len(ii):
            matched = self.keys[loc[ii]] == keys[ii]
            jj = ii[matched]
            result[jj] = self.residuals[loc[jj]]
        return result


class MarkovSequenceModel:
    def __init__(self, source, y, source_histories):
        self.fields = ["video_id", "author_id", "tag"]
        strengths = {
            "video_id": 12.0,
            "author_id": 14.0,
            "tag": 20.0
        }
        self.rates = {}
        self.transitions = {}

        for field in self.fields:
            cardinality = int(FEATURE_CARDINALITIES[field])
            current = np.asarray(source.X[field], dtype=np.int64)
            previous = source_histories[field][:, -1].astype(np.int64)
            rate = entity_rate(current, y, cardinality, strength=35.0)
            self.rates[field] = rate
            self.transitions[field] = SparseTransition.fit(
                previous, current, y, rate, cardinality, strengths[field]
            )

    def predict(self, target, target_histories):
        components = {}
        residuals = {}
        for field in self.fields:
            current = np.asarray(target.X[field], dtype=np.int64)
            previous = target_histories[field][:, -1].astype(np.int64)
            components[field] = safe_logit(self.rates[field][current])
            residuals[field] = self.transitions[field].predict(
                previous, current, int(FEATURE_CARDINALITIES[field])
            )

        entity = (
            0.38 * components["video_id"] +
            0.37 * components["author_id"] +
            0.25 * components["tag"]
        )
        transition = (
            0.38 * residuals["video_id"] +
            0.37 * residuals["author_id"] +
            0.25 * residuals["tag"]
        )
        return (entity + 4.0 * transition).astype(np.float32)


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

history_fields = ["video_id", "author_id", "tag"]
train_histories, train_state = build_source_histories(
    train, y_train, history_fields
)
valid_histories = build_target_histories(
    train_state, valid, history_fields
)

# Family 1: neural candidate-conditioned attention over recent positives.
train_din_arrays = make_din_arrays(train, train_histories)
valid_din_arrays = make_din_arrays(valid, valid_histories)
din_model = fit_din(train_din_arrays, y_train)
din_valid = predict_din(din_model, valid_din_arrays)

# Family 2: sparse first-order transition target statistics.
markov_model = MarkovSequenceModel(train, y_train, train_histories)
markov_valid = markov_model.predict(valid, valid_histories)

families = {
    "din_recent_positive_attention": din_valid,
    "markov_positive_transition": markov_valid
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_valid_rank = within_user_ranks(valid.user_id, inc_valid)

candidate_scores = {
    "trusted_incumbent": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    )
}
best_primary = -np.inf
best_name = None
best_family = None
best_alpha = None
best_scores = None
best_raw = None
best_metrics = None

alphas = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]

for family_name, raw in families.items():
    raw_metrics = evaluate(valid.user_id, y_valid, raw)
    candidate_scores[family_name + "_raw"] = float(raw_metrics["primary"])
    raw_rank = within_user_ranks(valid.user_id, raw)

    for alpha in alphas:
        blended = (
            (1.0 - alpha) * inc_valid_rank +
            alpha * raw_rank
        )
        metrics = evaluate(valid.user_id, y_valid, blended)
        name = family_name + "_blend_" + str(alpha)
        candidate_scores[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_name = name
            best_family = family_name
            best_alpha = float(alpha)
            best_scores = blended.copy()
            best_raw = np.asarray(raw).copy()
            best_metrics = metrics

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": best_name,
    "winner_family": best_family,
    "blend_alpha": best_alpha,
    "din_raw_primary": candidate_scores[
        "din_recent_positive_attention_raw"
    ],
    "markov_raw_primary": candidate_scores[
        "markov_positive_transition_raw"
    ],
    "users_with_positive_history_valid": float(np.mean(
        valid_histories["video_id"][:, -1] != 0
    ))
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64)
    )

# Refit the validation-selected recipe on train+validation, then construct
# test histories exclusively from outcomes in that combined past window.
combined = combine_sources(train, valid)
y_combined = np.concatenate([
    y_train,
    y_valid.astype(np.float32, copy=False)
])
test = load("test")

combined_histories, combined_state = build_source_histories(
    combined, y_combined, history_fields
)
test_histories = build_target_histories(
    combined_state, test, history_fields
)

if best_family == "din_recent_positive_attention":
    del din_model
    combined_din_arrays = make_din_arrays(combined, combined_histories)
    test_din_arrays = make_din_arrays(test, test_histories)
    refit_model = fit_din(combined_din_arrays, y_combined)
    raw_test = predict_din(refit_model, test_din_arrays)
else:
    refit_model = MarkovSequenceModel(
        combined, y_combined, combined_histories
    )
    raw_test = refit_model.predict(test, test_histories)

inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)
test_scores = (
    (1.0 - best_alpha) * within_user_ranks(test.user_id, inc_test) +
    best_alpha * within_user_ranks(test.user_id, raw_test)
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed)
}))