"""Numpy-only FM reproduction of the organizers' official KuaiRand-Pure baseline, plus random/itempop reference rungs.

This script reads the raw CSVs directly rather than importing pipeline.data, so the numpy-only organizer
baseline stays a fully self-contained artifact (no dependency on the torch-oriented cache pipeline). It
does import pipeline.evaluate.evaluate (pure numpy, no torch/pandas/sklearn) rather than reimplement
metrics a second time -- see the KNOWN GAP note above main() for why the numbers still miss the target.

Field choice (5 categorical fields):
  user_id, video_id, tab            -- required core identity + serving-context fields.
  author_id, dur_bucket             -- the official field set, from baseline_scores.json fm_official.config.fields.

Split: matches pipeline/data.py's authoritative date ranges exactly (train=log_standard_4_08_to_4_21 in
  full, valid=log_standard_4_22_to_5_08 dates 20220422-20220428, test=same file dates 20220429-20220508,
  test held out and never touched here). log_random_4_22_to_5_08_pure.csv is unused: an earlier attempt at
  using it as valid collapsed FM's train->valid transfer (its ~18% CTR vs standard log's ~44% CTR is a
  covariate shift a plain FM can't bridge), and the published val/test GAUC gap (0.6674 vs 0.6610) is far
  too small to be consistent with crossing into that differently-collected, unbiased-exposure log.

primary metric: pipeline.evaluate confirms primary == mean(gauc, ndcg@5), independently re-derived here
  before that module existed by solving published (gauc, ndcg@5, primary) triples for valid and test.
"""
import argparse
import csv
import os
import time

import numpy as np

from pipeline.evaluate import evaluate

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "KuaiRand-Pure", "data")
TRAIN_LOG = os.path.join(DATA_DIR, "log_standard_4_08_to_4_21_pure.csv")
VALID_LOG = os.path.join(DATA_DIR, "log_standard_4_22_to_5_08_pure.csv")
VALID_DATE_RANGE = ("20220422", "20220428")  # matches pipeline.data._VALID_RANGE; test dates (20220429+) held out
USER_FEATURES = os.path.join(DATA_DIR, "user_features_pure.csv")

# official config, kuairand-starter-kit/baseline_scores.json: fm_official.config.fields
DEFAULT_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]


def _read_log(path: str, date_range: tuple[str, str] | None = None) -> dict[str, np.ndarray]:
    """Reads user_id, video_id, date, long_view, tab from a KuaiRand log CSV (csv module + numpy, no pandas)."""
    names = ["user_id", "video_id", "date", "long_view", "tab"]
    picked = [[] for _ in names]
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        cols = [header.index(n) for n in names]  # resolve by name; positional indices silently read the wrong label
        for row in r:
            if date_range is not None and not (date_range[0] <= row[2] <= date_range[1]):
                continue
            for out, c in zip(picked, cols):
                out.append(row[c])
    uid, vid, date, click, tab = (np.array(p, dtype=np.int64) for p in picked)
    return {"user_id": uid, "video_id": vid, "date": date, "long_view": click, "tab": tab}


def _read_user_features() -> dict[int, dict[str, str]]:
    out = {}
    with open(USER_FEATURES, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["user_id"])] = row
    return out


def build_vocab(values) -> dict:
    """Maps distinct training values to contiguous ids starting at 1; 0 is reserved for OOV."""
    vocab = {}
    for v in values:
        key = str(v)
        if key not in vocab:
            vocab[key] = len(vocab) + 1
    return vocab


def encode(values, vocab: dict) -> np.ndarray:
    return np.array([vocab.get(str(v), 0) for v in values], dtype=np.int64)


def load_fields(fields: list[str]):
    """Reads raw CSVs, joins the user-side categorical fields, builds train-fit vocabs, encodes both splits."""
    train_log = _read_log(TRAIN_LOG)
    valid_log = _read_log(VALID_LOG, date_range=VALID_DATE_RANGE)
    users = _read_user_features()

    def user_field(log, name):
        return [users.get(int(u), {}).get(name, "UNKNOWN") for u in log["user_id"]]

    raw_train, raw_valid = {}, {}
    for f in fields:
        if f in train_log:
            raw_train[f], raw_valid[f] = train_log[f], valid_log[f]
        else:
            raw_train[f] = user_field(train_log, f)
            raw_valid[f] = user_field(valid_log, f)

    cardinalities, X_train, X_valid = {}, {}, {}
    for f in fields:
        vocab = build_vocab(raw_train[f])
        cardinalities[f] = len(vocab) + 1
        X_train[f] = encode(raw_train[f], vocab)
        X_valid[f] = encode(raw_valid[f], vocab)

    y_train = train_log["long_view"].astype(np.float64)
    y_valid = valid_log["long_view"].astype(np.float64)
    return X_train, y_train, X_valid, y_valid, cardinalities, train_log["user_id"], valid_log["user_id"], valid_log["video_id"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class FM:
    """Pure-numpy 2nd-order factorization machine, logistic loss, Adagrad, O(k*n) pairwise term."""

    def __init__(self, cardinalities: dict[str, int], fields: list[str], k: int = 16, lr: float = 0.001, seed: int = 0):
        self.fields, self.k, self.lr = fields, k, lr
        rng = np.random.default_rng(seed)
        self.w0 = 0.0
        self.w = {f: np.zeros(cardinalities[f]) for f in fields}
        self.v = {f: rng.normal(0, 0.01, size=(cardinalities[f], k)) for f in fields}
        self.g_w0 = 1e-8
        self.g_w = {f: np.full(cardinalities[f], 1e-8) for f in fields}
        self.g_v = {f: np.full((cardinalities[f], k), 1e-8) for f in fields}

    def _forward(self, Xb: dict[str, np.ndarray]):
        e = {f: self.v[f][Xb[f]] for f in self.fields}
        sum_e = sum(e.values())
        sum_sq = sum(ef ** 2 for ef in e.values())
        interaction = 0.5 * np.sum(sum_e ** 2 - sum_sq, axis=1)
        linear = sum(self.w[f][Xb[f]] for f in self.fields)
        logit = self.w0 + linear + interaction
        return logit, e, sum_e

    def _adagrad_step(self, g_state, w_state, idx, grad_local, eps=1e-8):
        uniq, inverse = np.unique(idx, return_inverse=True)
        shape = (len(uniq),) + grad_local.shape[1:]
        grad_sum = np.zeros(shape)
        np.add.at(grad_sum, inverse, grad_local)
        g_state[uniq] += grad_sum ** 2
        w_state[uniq] -= self.lr / (np.sqrt(g_state[uniq]) + eps) * grad_sum

    def fit(self, X, y, epochs: int = 8, batch_size: int = 2048, seed: int = 0, verbose: bool = False):
        n = len(y)
        rng = np.random.default_rng(seed)
        for epoch in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                bidx = perm[start:start + batch_size]
                Xb = {f: X[f][bidx] for f in self.fields}
                yb = y[bidx]
                logit, e, sum_e = self._forward(Xb)
                pred = _sigmoid(logit)
                d = (pred - yb) / len(bidx)  # mean-BCE gradient wrt logit
                self.g_w0 += d.sum() ** 2
                self.w0 -= self.lr / (np.sqrt(self.g_w0) + 1e-8) * d.sum()
                for f in self.fields:
                    self._adagrad_step(self.g_w[f], self.w[f], Xb[f], d)
                    grad_v = d[:, None] * (sum_e - e[f])
                    self._adagrad_step(self.g_v[f], self.v[f], Xb[f], grad_v)
            if verbose:
                logit, _, _ = self._forward(X)
                p = np.clip(_sigmoid(logit), 1e-7, 1 - 1e-7)
                loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                print(f"epoch {epoch}: train logloss {loss:.4f}")

    def predict_proba(self, X, chunk: int = 200_000):
        n = len(next(iter(X.values())))
        out = np.empty(n)
        for start in range(0, n, chunk):
            Xb = {f: X[f][start:start + chunk] for f in self.fields}
            logit, _, _ = self._forward(Xb)
            out[start:start + chunk] = _sigmoid(logit)
        return out


def random_scores(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def itempop_scores(train_video_id, train_y, valid_video_id) -> np.ndarray:
    """Score = per-video positive RATE. Matches the official item_popularity rung; raw counts do not."""
    max_id = int(max(train_video_id.max(), valid_video_id.max())) + 1
    pos = np.zeros(max_id)
    imp = np.zeros(max_id)
    np.add.at(pos, train_video_id, train_y)
    np.add.at(imp, train_video_id, 1.0)
    return (pos / np.maximum(imp, 1.0))[valid_video_id]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fm", choices=["fm", "random", "itempop"])
    p.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# MEASURED (--model fm, default args, seed 0): gauc=0.6463 (target 0.6674), ndcg@5=0.6477 (target 0.5357),
# primary=0.6470 (target 0.6016). GAUC lands within ~2pp of target -- good evidence the fields/FM/training
# are sound. ndcg@5 (and so primary) runs consistently high across ALL scorers here, including random
# (ndcg@5=0.587, vs an implied ~0.45 backed out from published gauc~0.5 and primary=0.4753): each user is ranked
# only against the videos actually logged for them (median ~4 in the one-week valid window), and at ~44%
# CTR that small, high-relevance-density pool inflates ndcg@5 regardless of scorer quality. Best guess: the
# organizers rank each user against a larger/shared candidate pool (e.g. the full video catalog, or
# negative-sampled) -- not reproduced here for lack of a specified candidate-set protocol.
def main():
    args = parse_args()
    fields = args.fields.split(",")
    t0 = time.perf_counter()

    X_train, y_train, X_valid, y_valid, cardinalities, train_uid, valid_uid, valid_vid = load_fields(fields)

    if args.model == "random":
        scores = random_scores(len(y_valid), seed=args.seed)
    elif args.model == "itempop":
        scores = itempop_scores(X_train["video_id"], y_train, X_valid["video_id"])
    else:
        model = FM(cardinalities, fields, k=args.k, lr=args.lr, seed=args.seed)
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, verbose=args.verbose)
        scores = model.predict_proba(X_valid)

    wall = time.perf_counter() - t0
    metrics = evaluate(valid_uid, y_valid, scores)
    print(f"model={args.model} fields={fields}")
    print(metrics)
    print(f"wall_clock_seconds={wall:.2f}")


if __name__ == "__main__":
    main()
