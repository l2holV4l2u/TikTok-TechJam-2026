import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 73129
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
DEVICE = torch.device("cpu")

HISTORY_LEN = 6
BATCH_SIZE = 8192
EPOCHS = 2
EMBED_DIM = 16
HALF_LIFE = 4.0

SIDE_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "hour",
]
SIDE_CARDS = [int(FEATURE_CARDINALITIES[f]) for f in SIDE_FIELDS]
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])


def concatenate_attr(splits, attr, key=None):
    if key is None:
        return np.concatenate([np.asarray(getattr(s, attr)) for s in splits])
    return np.concatenate([np.asarray(getattr(s, attr)[key]) for s in splits])


def ordered_history_for_splits(splits, history_len=HISTORY_LEN):
    uid = concatenate_attr(splits, "user_id").astype(np.int64, copy=False)
    tm = concatenate_attr(splits, "time_ms").astype(np.int64, copy=False)
    vid = concatenate_attr(splits, "X", "video_id").astype(np.int64, copy=False) + 1

    n = len(uid)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, tm, uid))
    sorted_uid = uid[order]
    sorted_vid = vid[order]

    hist_sorted = np.zeros((n, history_len), dtype=np.int32)
    positions = np.arange(n, dtype=np.int64)

    # Columns are oldest-to-newest so the GRU sees chronological context.
    for col, lag in enumerate(range(history_len, 0, -1)):
        source = positions - lag
        good = source >= 0
        good &= np.where(good, sorted_uid[np.maximum(source, 0)] == sorted_uid, False)
        hist_sorted[good, col] = sorted_vid[source[good]].astype(np.int32)

    hist = np.empty_like(hist_sorted)
    hist[order] = hist_sorted
    return hist


def make_side_features(splits):
    cols = []
    for f in SIDE_FIELDS:
        cols.append(
            concatenate_attr(splits, "X", f).astype(np.int64, copy=False) + 1
        )
    return np.stack(cols, axis=1).astype(np.int64, copy=False)


def ordinal_dates(dates):
    dates = np.asarray(dates)
    unique = np.unique(dates)
    mapping = {}
    for d in unique:
        text = str(int(d))
        dt = np.datetime64(
            "%s-%s-%s" % (text[:4], text[4:6], text[6:8]), "D"
        )
        mapping[int(d)] = int(dt.astype(np.int64))
    return np.asarray([mapping[int(d)] for d in dates], dtype=np.float64)


def recency_weights(dates, half_life=HALF_LIFE):
    days = ordinal_dates(dates)
    age = float(days.max()) - days
    w = np.exp2(-age / float(half_life))
    w /= max(float(w.mean()), 1e-12)
    return w.astype(np.float32)


def within_user_percentile(user_ids, scores):
    uid = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, uid))
    su = uid[order]

    starts_flag = np.r_[True, su[1:] != su[:-1]]
    starts = np.maximum.accumulate(
        np.where(starts_flag, np.arange(n, dtype=np.int64), 0)
    )
    end_flag = np.r_[su[:-1] != su[1:], True]
    ends = np.minimum.accumulate(
        np.where(end_flag, np.arange(n, dtype=np.int64), n - 1)[::-1]
    )[::-1]

    pos = np.arange(n, dtype=np.int64) - starts
    lengths = ends - starts + 1
    ranked = np.where(
        lengths > 1,
        pos.astype(np.float64) / np.maximum(lengths - 1, 1),
        0.5,
    )
    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


class SequenceModel(nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family

        self.video_embedding = nn.Embedding(
            VIDEO_CARD + 1, EMBED_DIM, padding_idx=0
        )
        self.side_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, EMBED_DIM, padding_idx=0)
            for card in SIDE_CARDS[1:]
        ])
        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, 1, padding_idx=0)
            for card in SIDE_CARDS
        ])

        if family == "din":
            self.attention = nn.Sequential(
                nn.Linear(4 * EMBED_DIM, 48),
                nn.ReLU(),
                nn.Linear(48, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
        elif family == "gru":
            self.gru = nn.GRU(
                input_size=EMBED_DIM,
                hidden_size=EMBED_DIM,
                batch_first=True,
            )
        else:
            raise ValueError(family)

        self.output = nn.Sequential(
            nn.Linear(4 * EMBED_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.video_embedding.weight, std=0.025)
        with torch.no_grad():
            self.video_embedding.weight[0].zero_()
        for emb in self.side_embeddings:
            nn.init.normal_(emb.weight, std=0.025)
            with torch.no_grad():
                emb.weight[0].zero_()
        for emb in self.linear_embeddings:
            nn.init.zeros_(emb.weight)

    def candidate_embedding(self, side):
        candidate = self.video_embedding(side[:, 0])
        for j, emb in enumerate(self.side_embeddings, start=1):
            candidate = candidate + emb(side[:, j])
        return candidate

    def forward(self, side, history):
        candidate = self.candidate_embedding(side)
        history_emb = self.video_embedding(history)
        mask = history != 0

        if self.family == "din":
            repeated_candidate = candidate.unsqueeze(1).expand_as(history_emb)
            attention_input = torch.cat(
                [
                    repeated_candidate,
                    history_emb,
                    repeated_candidate * history_emb,
                    repeated_candidate - history_emb,
                ],
                dim=2,
            )
            attention_logits = self.attention(attention_input).squeeze(-1)
            attention_logits = attention_logits.masked_fill(~mask, -1e4)
            attention_weights = torch.softmax(attention_logits, dim=1)
            attention_weights = attention_weights * mask.float()
            attention_weights = attention_weights / (
                attention_weights.sum(dim=1, keepdim=True) + 1e-8
            )
            state = (attention_weights.unsqueeze(-1) * history_emb).sum(dim=1)
        else:
            output, _ = self.gru(history_emb)
            lengths = mask.sum(dim=1)
            last = torch.clamp(lengths - 1, min=0)
            state = output[
                torch.arange(len(output), device=output.device), last
            ]
            state = state * (lengths > 0).float().unsqueeze(1)

        interaction = torch.cat(
            [candidate, state, candidate * state, candidate - state], dim=1
        )
        wide = torch.zeros(len(side), device=side.device)
        for j, emb in enumerate(self.linear_embeddings):
            wide = wide + emb(side[:, j]).squeeze(-1)
        return self.bias + wide + self.output(interaction).squeeze(-1)


def fit_sequence_model(family, side, history, labels, dates, seed):
    torch.manual_seed(seed)
    model = SequenceModel(family).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    weights = torch.from_numpy(recency_weights(dates))
    side_t = torch.from_numpy(np.asarray(side, dtype=np.int64))
    hist_t = torch.from_numpy(np.asarray(history, dtype=np.int64))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 19)
    n = len(y)

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = side_t[idx].to(DEVICE)
            hb = hist_t[idx].to(DEVICE)
            yb = y[idx].to(DEVICE)
            wb = weights[idx].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (losses * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.inference_mode()
def predict_sequence_model(model, side, history):
    model.eval()
    side_t = torch.from_numpy(np.asarray(side, dtype=np.int64))
    hist_t = torch.from_numpy(np.asarray(history, dtype=np.int64))
    out = np.empty(len(side), dtype=np.float64)
    for start in range(0, len(side), 16384):
        end = min(start + 16384, len(side))
        logits = model(
            side_t[start:end].to(DEVICE),
            hist_t[start:end].to(DEVICE),
        )
        out[start:end] = logits.cpu().numpy().astype(np.float64)
    return out


TRANSITION_SPECS = [
    # (previous field index, current field index)
    (2, 2),  # previous tag -> current tag
    (1, 2),  # previous author -> current tag
    (0, 2),  # previous video -> current tag
    (3, 3),  # previous duration bucket -> current duration bucket
]
SHIFTED_CARDS = [c + 1 for c in SIDE_CARDS]


def fit_transition_model(side, history, labels, dates):
    y = np.asarray(labels, dtype=np.float64)
    weights = recency_weights(dates).astype(np.float64)
    previous_video = np.asarray(history[:, -1], dtype=np.int64)
    has_previous = previous_video != 0

    # History stores video IDs. Build aligned previous-row side features using
    # chronological row links rather than trying to infer side fields from IDs.
    # The caller supplies _previous_side as an attached global below.
    previous_side = fit_transition_model.previous_side
    global_rate = float(np.sum(weights * y) / np.sum(weights))
    tables = []

    for prev_idx, cur_idx in TRANSITION_SPECS:
        prev_values = previous_side[:, prev_idx]
        cur_values = side[:, cur_idx]
        cur_card = SHIFTED_CARDS[cur_idx]
        prev_card = SHIFTED_CARDS[prev_idx]
        key = prev_values * cur_card + cur_values
        size = prev_card * cur_card

        count = np.bincount(
            key[has_previous],
            weights=weights[has_previous],
            minlength=size,
        ).astype(np.float64)
        positive = np.bincount(
            key[has_previous],
            weights=(weights * y)[has_previous],
            minlength=size,
        ).astype(np.float64)
        tables.append((prev_idx, cur_idx, count, positive))

    return {"global": global_rate, "tables": tables}


def predict_transition_model(model, side, history, previous_side):
    n = len(side)
    has_previous = np.asarray(history[:, -1]) != 0
    prior = float(model["global"])
    numerator = np.full(n, prior * 8.0, dtype=np.float64)
    denominator = np.full(n, 8.0, dtype=np.float64)

    for prev_idx, cur_idx, count, positive in model["tables"]:
        cur_card = SHIFTED_CARDS[cur_idx]
        key = previous_side[:, prev_idx] * cur_card + side[:, cur_idx]
        c = count[key]
        p = positive[key]
        local_weight = np.minimum(c, 40.0)
        local_rate = (p + 12.0 * prior) / (c + 12.0)
        numerator += local_weight * local_rate
        denominator += local_weight

    score = numerator / denominator
    score[~has_previous] = prior
    return score


def previous_side_for_splits(splits, side):
    uid = concatenate_attr(splits, "user_id").astype(np.int64, copy=False)
    tm = concatenate_attr(splits, "time_ms").astype(np.int64, copy=False)
    n = len(uid)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, tm, uid))
    su = uid[order]
    ss = side[order]

    previous_sorted = np.zeros_like(ss)
    same = np.r_[False, su[1:] == su[:-1]]
    previous_sorted[same] = ss[:-1][same[1:]]

    previous = np.empty_like(previous_sorted)
    previous[order] = previous_sorted
    return previous


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path):
    raise RuntimeError("Trusted incumbent validation scores are unavailable")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

train_side = make_side_features([train])
train_history = ordered_history_for_splits([train])
train_previous_side = previous_side_for_splits([train], train_side)

train_valid_side_all = make_side_features([train, valid])
train_valid_history_all = ordered_history_for_splits([train, valid])
train_valid_previous_all = previous_side_for_splits(
    [train, valid], train_valid_side_all
)
n_train = len(y_train)
valid_side = train_valid_side_all[n_train:]
valid_history = train_valid_history_all[n_train:]
valid_previous_side = train_valid_previous_all[n_train:]

raw_predictions = {}
trained_models = {}

din_model = fit_sequence_model(
    "din",
    train_side,
    train_history,
    y_train,
    train.date,
    SEED + 100,
)
raw_predictions["din_sequence"] = predict_sequence_model(
    din_model, valid_side, valid_history
)
trained_models["din_sequence"] = din_model

gru_model = fit_sequence_model(
    "gru",
    train_side,
    train_history,
    y_train,
    train.date,
    SEED + 200,
)
raw_predictions["gru_sequence"] = predict_sequence_model(
    gru_model, valid_side, valid_history
)
trained_models["gru_sequence"] = gru_model

fit_transition_model.previous_side = train_previous_side
transition_model = fit_transition_model(
    train_side, train_history, y_train, train.date
)
raw_predictions["transition_statistics"] = predict_transition_model(
    transition_model,
    valid_side,
    valid_history,
    valid_previous_side,
)
trained_models["transition_statistics"] = transition_model

candidate_scores = {}
candidate_predictions = {}
candidate_recipes = {}

inc_metric = evaluate(valid.user_id, valid.y, inc_valid)
candidate_scores["incumbent"] = float(inc_metric["primary"])
candidate_predictions["incumbent"] = inc_valid
candidate_recipes["incumbent"] = {
    "family": "incumbent",
    "mode": "raw",
    "alpha": 0.0,
}

inc_rank = within_user_percentile(valid.user_id, inc_valid)

for family, pred in raw_predictions.items():
    raw_metric = evaluate(valid.user_id, valid.y, pred)
    candidate_scores[family] = float(raw_metric["primary"])
    candidate_predictions[family] = pred
    candidate_recipes[family] = {
        "family": family,
        "mode": "raw",
        "alpha": 1.0,
    }

    pred_rank = within_user_percentile(valid.user_id, pred)
    rank_name = family + "_rank"
    rank_metric = evaluate(valid.user_id, valid.y, pred_rank)
    candidate_scores[rank_name] = float(rank_metric["primary"])
    candidate_predictions[rank_name] = pred_rank
    candidate_recipes[rank_name] = {
        "family": family,
        "mode": "rank",
        "alpha": 1.0,
    }

    for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
        name = "%s_rankblend_%.2f" % (family, alpha)
        blended = (1.0 - alpha) * inc_rank + alpha * pred_rank
        metric = evaluate(valid.user_id, valid.y, blended)
        candidate_scores[name] = float(metric["primary"])
        candidate_predictions[name] = blended
        candidate_recipes[name] = {
            "family": family,
            "mode": "rank_blend",
            "alpha": alpha,
        }

winner = max(candidate_scores, key=candidate_scores.get)
recipe = candidate_recipes[winner]
valid_scores = np.asarray(candidate_predictions[winner], dtype=np.float64)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS sequence_standalone din=%.6f gru=%.6f transition=%.6f winner=%s"
    % (
        candidate_scores["din_sequence"],
        candidate_scores["gru_sequence"],
        candidate_scores["transition_statistics"],
        winner,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "scores_valid.npy"), valid_scores)

test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

family = recipe["family"]
mode = recipe["mode"]
alpha = float(recipe["alpha"])

if family == "incumbent":
    test_scores = inc_test.copy()
else:
    y_valid = np.asarray(valid.y, dtype=np.int8)
    y_combined = np.concatenate([y_train, y_valid])
    combined_dates = np.concatenate([
        np.asarray(train.date),
        np.asarray(valid.date),
    ])

    combined_side = train_valid_side_all
    combined_history = train_valid_history_all
    combined_previous_side = train_valid_previous_all

    all_test_side = make_side_features([train, valid, test])
    all_test_history = ordered_history_for_splits([train, valid, test])
    all_test_previous = previous_side_for_splits(
        [train, valid, test], all_test_side
    )
    n_combined = len(y_combined)
    test_side = all_test_side[n_combined:]
    test_history = all_test_history[n_combined:]
    test_previous_side = all_test_previous[n_combined:]

    if family == "din_sequence":
        final_model = fit_sequence_model(
            "din",
            combined_side,
            combined_history,
            y_combined,
            combined_dates,
            SEED + 100,
        )
        raw_test = predict_sequence_model(
            final_model, test_side, test_history
        )
    elif family == "gru_sequence":
        final_model = fit_sequence_model(
            "gru",
            combined_side,
            combined_history,
            y_combined,
            combined_dates,
            SEED + 200,
        )
        raw_test = predict_sequence_model(
            final_model, test_side, test_history
        )
    elif family == "transition_statistics":
        fit_transition_model.previous_side = combined_previous_side
        final_model = fit_transition_model(
            combined_side,
            combined_history,
            y_combined,
            combined_dates,
        )
        raw_test = predict_transition_model(
            final_model,
            test_side,
            test_history,
            test_previous_side,
        )
    else:
        raise RuntimeError("Unknown selected family: " + family)

    if mode == "raw":
        test_scores = raw_test
    else:
        test_rank = within_user_percentile(test.user_id, raw_test)
        if mode == "rank":
            test_scores = test_rank
        elif mode == "rank_blend":
            inc_test_rank = within_user_percentile(test.user_id, inc_test)
            test_scores = (
                (1.0 - alpha) * inc_test_rank + alpha * test_rank
            )
        else:
            raise RuntimeError("Unknown selected mode: " + mode)

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