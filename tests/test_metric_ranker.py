"""Tests for the standalone metric-aligned ranking experiment."""
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from pipeline.data import AUX_DTYPES, FEATURE_CARDINALITIES, NUMERIC_FEATURES, write_cache
from research.metric_ranker import (
    NumericPreprocessor,
    WholeUserSlateSampler,
    lambda_ndcg_loss,
    main as metric_ranker_main,
    pairwise_auc_loss,
    user_groups,
)


class _SplitLike:
    def __len__(self):
        return len(self.user_id)


def _dense_split(n_users: int, rows_per_user: int, seed: int):
    rng = np.random.default_rng(seed)
    n = n_users * rows_per_user
    user_id = np.repeat(np.arange(n_users, dtype=np.int64), rows_per_user)
    video_id = rng.integers(1, 100, size=n, dtype=np.int64)
    X = {
        feat: rng.integers(1, min(card, 16), size=n, dtype=np.int64)
        for feat, card in FEATURE_CARDINALITIES.items()
    }
    X["user_id"] = np.repeat(np.arange(1, n_users + 1, dtype=np.int64), rows_per_user)
    X["video_id"] = video_id
    y = np.tile(np.array([1, 0, 0, 1, 0, 0], dtype=np.int8), n_users)[:n]
    aux = {}
    for name, dtype in AUX_DTYPES.items():
        if dtype == "int8":
            aux[name] = ((rng.random(n) < 0.2) & (y == 1)).astype(np.int8)
        else:
            aux[name] = np.where(y == 1, 1000.0, 0.0).astype(np.float32)
    date = np.full(n, 20220410 + seed % 10, dtype=np.int32)
    time_ms = np.full(n, 1649675512388 + seed, dtype=np.int64)
    num = {
        name: rng.normal(loc=10.0 + seed, scale=2.0, size=n).astype(np.float32)
        for name in NUMERIC_FEATURES
    }
    return user_id, video_id, X, y, aux, date, time_ms, num


def _build_dense_cache(path: Path) -> None:
    for i, split in enumerate(("train", "valid", "test")):
        data = _dense_split(n_users=8 if split == "train" else 3, rows_per_user=6, seed=10 + i)
        write_cache(path, split, *data[:5], date=data[5], time_ms=data[6], num=data[7])


def test_lambda_losses_prefer_correct_order_and_backpropagate():
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    good = torch.tensor([2.0, -1.0, 1.0, -2.0], requires_grad=True)
    bad = torch.tensor([-2.0, 2.0, -1.0, 1.0], requires_grad=True)

    assert lambda_ndcg_loss(good, labels, topk=3) < lambda_ndcg_loss(bad, labels, topk=3)
    assert pairwise_auc_loss(good, labels) < pairwise_auc_loss(bad, labels)

    loss = lambda_ndcg_loss(bad, labels, topk=3) + pairwise_auc_loss(bad, labels)
    loss.backward()
    assert torch.isfinite(bad.grad).all()
    assert torch.count_nonzero(bad.grad).item() > 0


def test_user_groups_are_whole_user_not_day_chunks():
    users = np.array([2, 1, 2, 1, 1, 2], dtype=np.int64)
    labels = np.array([1, 0, 0, 1, 0, 1], dtype=np.int8)

    groups = user_groups(users, labels, require_mixed=True)
    as_sets = [set(group.tolist()) for group in groups]

    assert {1, 3, 4} in as_sets
    assert {0, 2, 5} in as_sets


def test_sampler_truncates_large_users_but_keeps_positive_and_negative():
    users = np.repeat(np.array([1, 2], dtype=np.int64), 8)
    labels = np.tile(np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8), 2)
    sampler = WholeUserSlateSampler(users, labels, max_slate=4, users_per_epoch=2, seed=3)

    for rows in sampler.epoch():
        assert len(rows) == 4
        assert np.unique(users[rows]).size == 1
        assert np.any(labels[rows] == 1)
        assert np.any(labels[rows] == 0)


def test_numeric_preprocessor_state_is_train_only():
    train = _SplitLike()
    valid = _SplitLike()
    train.user_id = np.arange(4)
    valid.user_id = np.arange(2)
    train.num = {"duration_ms": np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)}
    valid.num = {"duration_ms": np.array([10000.0, 20000.0], dtype=np.float32)}

    prep = NumericPreprocessor.fit(train, names=("duration_ms",))
    before = (prep.center.copy(), prep.scale.copy())
    transformed = prep.transform(valid)

    assert np.array_equal(prep.center, before[0])
    assert np.array_equal(prep.scale, before[1])
    assert transformed.shape == (2, 1)
    assert transformed.max() == 8.0


def test_smoke_writes_valid_and_test_scores_with_hidden_test_labels():
    tmp = Path(tempfile.mkdtemp(prefix="metric_ranker_cache_"))
    out = tmp / "out"
    previous_cache = os.environ.get("KUAIRAND_CACHE_DIR")
    previous_hide = os.environ.get("AGENT_HIDE_TEST_LABELS")
    try:
        _build_dense_cache(tmp)
        os.environ["KUAIRAND_CACHE_DIR"] = str(tmp)
        os.environ["AGENT_HIDE_TEST_LABELS"] = "1"
        result = metric_ranker_main([
            "--out-dir", str(out),
            "--epochs", "1",
            "--max-users-per-epoch", "4",
            "--max-slate", "4",
            "--user-batch-size", "2",
            "--pred-batch-size", "8",
            "--fields", "user_id,video_id,author_id,tab",
            "--embed-dim", "4",
            "--hidden-dims", "8",
            "--aux-targets", "is_click",
            "--threads", "1",
            "--seed", "7",
        ])

        valid_scores = np.load(out / "scores_valid.npy")
        test_scores = np.load(out / "scores_test.npy")
        assert valid_scores.shape == (18,)
        assert test_scores.shape == (18,)
        assert np.isfinite(valid_scores).all()
        assert np.isfinite(test_scores).all()
        assert result["fixed_epochs"] == 1
    finally:
        if previous_cache is None:
            os.environ.pop("KUAIRAND_CACHE_DIR", None)
        else:
            os.environ["KUAIRAND_CACHE_DIR"] = previous_cache
        if previous_hide is None:
            os.environ.pop("AGENT_HIDE_TEST_LABELS", None)
        else:
            os.environ["AGENT_HIDE_TEST_LABELS"] = previous_hide
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
