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
SEED = 7319
BATCH_SIZE = 8192
EPOCHS = 4
RANK = 8

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
]

torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)

cardinalities = np.asarray(
    [int(FEATURE_CARDINALITIES[f]) for f in FIELDS], dtype=np.int64
)
offsets = np.cumsum(
    np.concatenate([np.zeros(1, dtype=np.int64), cardinalities[:-1]])
)
N_FIELDS = len(FIELDS)
TOTAL_CARDINALITY = int(cardinalities.sum())


def make_matrix(parts):
    cols = []
    for f, off in zip(FIELDS, offsets):
        if len(parts) == 1:
            a = np.asarray(parts[0].X[f], dtype=np.int64)
        else:
            a = np.concatenate(
                [np.asarray(p.X[f], dtype=np.int64) for p in parts]
            )
        cols.append(a + int(off))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FieldAwareFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1, sparse=True)
        # Each category has a different vector when interacting with each
        # partner field.
        self.factors = nn.Embedding(
            TOTAL_CARDINALITY * N_FIELDS, RANK, sparse=True
        )
        self.bias = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            self.linear.weight.zero_()
            self.factors.weight.normal_(0.0, 0.01)

    def forward(self, x):
        out = self.bias + self.linear(x).squeeze(-1).sum(dim=1)
        interaction = torch.zeros(
            x.shape[0], dtype=torch.float32, device=x.device
        )
        for i in range(N_FIELDS):
            xi = x[:, i]
            for j in range(i + 1, N_FIELDS):
                xj = x[:, j]
                vi_for_j = self.factors(xi * N_FIELDS + j)
                vj_for_i = self.factors(xj * N_FIELDS + i)
                interaction = interaction + (
                    vi_for_j * vj_for_i
                ).sum(dim=1)
        return out + interaction

    def sparse_parameters(self):
        return [self.linear.weight, self.factors.weight]

    def dense_parameters(self):
        return [self.bias]


class DeepFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1, sparse=True)
        self.embedding = nn.Embedding(
            TOTAL_CARDINALITY, RANK, sparse=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(N_FIELDS * RANK, 64),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            self.linear.weight.zero_()
            self.embedding.weight.normal_(0.0, 0.01)

    def forward(self, x):
        emb = self.embedding(x)
        linear = self.linear(x).squeeze(-1).sum(dim=1)

        summed = emb.sum(dim=1)
        fm = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        ).sum(dim=1)

        deep = self.mlp(emb.reshape(emb.shape[0], -1)).squeeze(1)
        return self.bias + linear + fm + deep

    def sparse_parameters(self):
        return [self.linear.weight, self.embedding.weight]

    def dense_parameters(self):
        return list(self.mlp.parameters()) + [self.bias]


def build_model(family):
    if family == "ffm":
        return FieldAwareFM()
    if family == "deepfm":
        return DeepFM()
    raise ValueError(family)


def predict_torch(model, x_np):
    model.eval()
    ans = np.empty(len(x_np), dtype=np.float64)
    x = torch.from_numpy(x_np)
    with torch.no_grad():
        for st in range(0, len(x_np), BATCH_SIZE * 2):
            en = min(st + BATCH_SIZE * 2, len(x_np))
            ans[st:en] = (
                model(x[st:en]).cpu().numpy().astype(np.float64)
            )
    return ans


def fit_torch(
    family,
    x_np,
    y_np,
    epochs,
    valid_tuple=None,
    seed=SEED,
):
    seed_everything(seed)
    model = build_model(family)

    sparse_optimizer = torch.optim.SparseAdam(
        model.sparse_parameters(), lr=0.002
    )
    dense_optimizer = torch.optim.Adam(
        model.dense_parameters(), lr=0.001
    )

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(
        np.ascontiguousarray(y_np, dtype=np.float32)
    )
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_scores = None
    best_metrics = None
    best_epoch = epochs

    for epoch in range(1, epochs + 1):
        model.train()
        gen = torch.Generator()
        gen.manual_seed(seed + 1009 * epoch)
        order = torch.randperm(len(x_np), generator=gen)

        total_loss = 0.0
        total_n = 0
        for st in range(0, len(x_np), BATCH_SIZE):
            idx = order[st:st + BATCH_SIZE]
            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)

            logits = model(x[idx])
            loss = criterion(logits, y[idx])
            loss.backward()

            sparse_optimizer.step()
            dense_optimizer.step()

            total_loss += float(loss.detach()) * len(idx)
            total_n += len(idx)

        if valid_tuple is not None:
            xv, vu, vy = valid_tuple
            vs = predict_torch(model, xv)
            met = evaluate(vu, vy, vs)
            print(
                "%s epoch=%d loss=%.6f primary=%.6f "
                "gauc=%.6f ndcg@5=%.6f"
                % (
                    family,
                    epoch,
                    total_loss / max(total_n, 1),
                    met["primary"],
                    met["gauc"],
                    met["ndcg@5"],
                ),
                flush=True,
            )
            if (
                best_metrics is None
                or met["primary"] > best_metrics["primary"]
            ):
                best_epoch = epoch
                best_scores = vs.copy()
                best_metrics = met
                best_state = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                }

    if valid_tuple is not None:
        model.load_state_dict(best_state)
        return model, best_epoch, best_scores, best_metrics
    return model


def logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def fit_oneway_table(ids, y, cardinality, prior, strength):
    counts = np.bincount(ids, minlength=cardinality).astype(np.float64)
    positives = np.bincount(
        ids, weights=y, minlength=cardinality
    ).astype(np.float64)
    rate = (positives + strength * prior) / (counts + strength)
    return logit(rate)


def fit_sparse_pair(a, b, b_cardinality, y, prior, strength):
    keys = (
        np.asarray(a, dtype=np.int64) * int(b_cardinality)
        + np.asarray(b, dtype=np.int64)
    )
    uniq, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    positives = np.bincount(
        inverse, weights=y, minlength=len(uniq)
    ).astype(np.float64)
    rates = (positives + strength * prior) / (
        counts.astype(np.float64) + strength
    )
    return uniq, logit(rates), b_cardinality, logit(prior)


def lookup_sparse_pair(table, a, b):
    uniq, values, b_cardinality, default = table
    keys = (
        np.asarray(a, dtype=np.int64) * int(b_cardinality)
        + np.asarray(b, dtype=np.int64)
    )
    pos = np.searchsorted(uniq, keys)
    safe = np.minimum(pos, len(uniq) - 1)
    found = (pos < len(uniq)) & (uniq[safe] == keys)
    out = np.full(len(keys), default, dtype=np.float64)
    out[found] = values[safe[found]]
    return out


def fit_empirical_bayes(parts, y):
    prior = float(np.mean(y))
    tables = {}
    strengths = {
        "user_id": 35.0,
        "video_id": 30.0,
        "author_id": 30.0,
        "tab": 80.0,
        "duration_bucket": 80.0,
        "tag": 55.0,
        "upload_type": 70.0,
        "music_type": 70.0,
        "hour": 90.0,
    }

    for field in FIELDS:
        ids = (
            np.asarray(parts[0].X[field], dtype=np.int64)
            if len(parts) == 1
            else np.concatenate(
                [np.asarray(p.X[field], dtype=np.int64) for p in parts]
            )
        )
        tables[field] = fit_oneway_table(
            ids,
            y,
            int(FEATURE_CARDINALITIES[field]),
            prior,
            strengths[field],
        )

    user = (
        np.asarray(parts[0].X["user_id"], dtype=np.int64)
        if len(parts) == 1
        else np.concatenate(
            [np.asarray(p.X["user_id"], dtype=np.int64) for p in parts]
        )
    )
    author = (
        np.asarray(parts[0].X["author_id"], dtype=np.int64)
        if len(parts) == 1
        else np.concatenate(
            [np.asarray(p.X["author_id"], dtype=np.int64) for p in parts]
        )
    )
    tag = (
        np.asarray(parts[0].X["tag"], dtype=np.int64)
        if len(parts) == 1
        else np.concatenate(
            [np.asarray(p.X["tag"], dtype=np.int64) for p in parts]
        )
    )

    pair_ua = fit_sparse_pair(
        user,
        author,
        int(FEATURE_CARDINALITIES["author_id"]),
        y,
        prior,
        18.0,
    )
    pair_ut = fit_sparse_pair(
        user,
        tag,
        int(FEATURE_CARDINALITIES["tag"]),
        y,
        prior,
        22.0,
    )
    return {
        "prior": prior,
        "tables": tables,
        "pair_ua": pair_ua,
        "pair_ut": pair_ut,
    }


def predict_empirical_bayes(model, part):
    t = model["tables"]

    # user_id itself is constant within a user's evaluation group, so it is
    # retained only weakly; content and context terms form most of the rank.
    score = (
        0.05 * t["user_id"][np.asarray(part.X["user_id"])]
        + 1.00 * t["video_id"][np.asarray(part.X["video_id"])]
        + 0.75 * t["author_id"][np.asarray(part.X["author_id"])]
        + 0.55 * t["tab"][np.asarray(part.X["tab"])]
        + 0.45
        * t["duration_bucket"][
            np.asarray(part.X["duration_bucket"])
        ]
        + 0.55 * t["tag"][np.asarray(part.X["tag"])]
        + 0.30 * t["upload_type"][np.asarray(part.X["upload_type"])]
        + 0.20 * t["music_type"][np.asarray(part.X["music_type"])]
        + 0.25 * t["hour"][np.asarray(part.X["hour"])]
    )

    ua = lookup_sparse_pair(
        model["pair_ua"],
        part.X["user_id"],
        part.X["author_id"],
    )
    ut = lookup_sparse_pair(
        model["pair_ut"],
        part.X["user_id"],
        part.X["tag"],
    )
    return score + 0.80 * ua + 0.45 * ut


def standardize(a):
    a = np.asarray(a, dtype=np.float64)
    sd = float(np.std(a))
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return (a - float(np.mean(a))) / sd


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
x_train = make_matrix([train])
x_valid = make_matrix([valid])

own_valid = {}
selected_epochs = {}

for family in ["ffm", "deepfm"]:
    model, epoch, scores, met = fit_torch(
        family,
        x_train,
        y_train,
        EPOCHS,
        valid_tuple=(x_valid, valid.user_id, y_valid),
        seed=SEED + (0 if family == "ffm" else 101),
    )
    own_valid[family] = scores
    selected_epochs[family] = epoch
    del model

eb_model = fit_empirical_bayes([train], y_train.astype(np.float64))
own_valid["empirical_bayes"] = predict_empirical_bayes(
    eb_model, valid
)
del eb_model

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

candidate_report = {}
best = None

inc_z = standardize(inc_valid)

for family, raw_scores in own_valid.items():
    raw_metrics = evaluate(valid.user_id, y_valid, raw_scores)
    candidate_report[family + "_raw"] = float(
        raw_metrics["primary"]
    )

    cand_z = standardize(raw_scores)
    for alpha in [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]:
        blended = alpha * cand_z + (1.0 - alpha) * inc_z
        met = evaluate(valid.user_id, y_valid, blended)
        name = "%s_blend_%.2f" % (family, alpha)
        candidate_report[name] = float(met["primary"])
        if best is None or met["primary"] > best["metrics"]["primary"]:
            best = {
                "family": family,
                "alpha": alpha,
                "scores": blended.copy(),
                "raw_scores": raw_scores.copy(),
                "metrics": met,
                "name": name,
            }

print(
    "CANDIDATES " + json.dumps(candidate_report, sort_keys=True),
    flush=True,
)
print(
    "FINDINGS winner=%s alpha=%.2f epochs=%s"
    % (
        best["family"],
        best["alpha"],
        str(selected_epochs.get(best["family"], "nonparametric")),
    ),
    flush=True,
)

# Refit the selected recipe on all labels available before test.
test = load("test")
y_train_valid = np.concatenate(
    [y_train, y_valid.astype(np.float32)]
)

if best["family"] in ("ffm", "deepfm"):
    x_train_valid = np.concatenate([x_train, x_valid], axis=0)
    final_model = fit_torch(
        best["family"],
        x_train_valid,
        y_train_valid,
        selected_epochs[best["family"]],
        valid_tuple=None,
        seed=SEED + (0 if best["family"] == "ffm" else 101),
    )
    x_test = make_matrix([test])
    own_test = predict_torch(final_model, x_test)
else:
    final_eb = fit_empirical_bayes(
        [train, valid], y_train_valid.astype(np.float64)
    )
    own_test = predict_empirical_bayes(final_eb, test)

inc_test = np.load(inc_test_path).astype(np.float64)
test_scores = (
    best["alpha"] * standardize(own_test)
    + (1.0 - best["alpha"]) * standardize(inc_test)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best["scores"], dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best["raw_scores"], dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
metrics = best["metrics"]
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, '
    '"ndcg@5": %.10f, "gpu_seconds": %.3f}'
    % (
        metrics["primary"],
        metrics["gauc"],
        metrics["ndcg@5"],
        elapsed,
    )
)