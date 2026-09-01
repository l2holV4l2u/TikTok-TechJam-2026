"""Feature assembly for the limit probe. Off the submission path -- limits/ is a scratch folder."""
import numpy as np

from pipeline.data import load
from pipeline.history import historical_features

from limits.affinity import affinity
from limits.session import features as session_features

AFF_ATTRS = ("tag", "video_type", "upload_type", "music_type", "duration_bucket")

ITEM_CAT = ("video_type", "upload_type", "music_type", "tag", "duration_bucket")
ITEM_IDS = ("video_id", "author_id")
CTX_CAT = ("tab", "hour")
USER_CAT = ("user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
            "follow_user_num_range", "fans_user_num_range", "friend_user_num_range",
            "register_days_range", "register_days_bucket") + tuple(f"onehot_feat{i}" for i in range(18))
USER_IDS = ("user_id",)
ITEM_NUM = ("duration_ms",)
USER_NUM = ("user_follow_user_num", "user_fans_user_num", "user_friend_user_num", "user_register_days")

GROUPS = {
    "item_cat": ITEM_CAT, "item_ids": ITEM_IDS, "ctx": CTX_CAT,
    "user_cat": USER_CAT, "user_ids": USER_IDS,
}
NUM_GROUPS = {"item_num": ITEM_NUM, "user_num": USER_NUM}


def build(split_name, groups, hist=True, aff=False, fm=None, sess=False, ctxnorm=False):
    """Return (matrix float32, names, categorical indices, split)."""
    sp = load(split_name)
    cols, names, cat = [], [], []
    for g in groups:
        for f in GROUPS.get(g, ()):
            cat.append(len(cols))
            cols.append(np.asarray(sp.X[f], dtype=np.float32))
            names.append(f)
        for f in NUM_GROUPS.get(g, ()):
            v = np.asarray(sp.num[f], dtype=np.float32) if sp.num and f in sp.num else None
            if v is None:
                continue
            cols.append(v)
            names.append(f)
    if sess:
        for name, v in session_features(split_name).items():
            cols.append(np.asarray(v, dtype=np.float32))
            names.append(name)
    if ctxnorm:
        # ranking happens inside the user's given candidate list, so what matters is how a row
        # compares to that list, not its absolute value
        uid = np.asarray(sp.user_id, dtype=np.int64)
        order = np.argsort(uid, kind="stable")
        u = uid[order]
        start = np.searchsorted(u, np.unique(u))
        size = np.diff(np.append(start, len(u)))
        gid = np.empty(len(u), dtype=np.int64); gid[order] = np.repeat(np.arange(len(size)), size)
        base = {"duration_ms": np.asarray(sp.num["duration_ms"], dtype=np.float64)}
        for key in ("video_id", "author_id"):
            base[f"{key}_lv"] = np.asarray(historical_features(split_name, key=key)[f"{key}_long_view_rate"], np.float64)
        for name, v in base.items():
            mean = np.bincount(gid, weights=v, minlength=len(size)) / np.bincount(gid, minlength=len(size))
            cols.append((v - mean[gid]).astype(np.float32))
            names.append(f"ctx_{name}_dev")
    if aff:
        for attr in AFF_ATTRS:
            for name, v in affinity(split_name, attr).items():
                cols.append(np.asarray(v, dtype=np.float32))
                names.append(name)
    if fm:
        import numpy.lib.npyio  # noqa: F401
        key = "valid" if split_name == "valid" else ("test" if split_name == "test" else "train")
        z = np.load(f"limits/out/{fm}.npz")
        if key not in z:
            raise KeyError(f"{fm}.npz has no {key} predictions")
        cols.append(np.asarray(z[key], dtype=np.float32))
        names.append(f"score_{fm}")
    if hist:
        for key in ("video_id", "author_id"):
            for name, v in historical_features(split_name, key=key).items():
                cols.append(np.asarray(v, dtype=np.float32))
                names.append(name)
    return np.column_stack(cols), names, cat, sp
