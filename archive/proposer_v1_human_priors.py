import re
from typing import Callable

from .kb import retrieve
from .loop import Proposal

CompleteFn = Callable[[str], tuple[str, int, int]]

_HYP_RE = re.compile(r"HYPOTHESIS:\s*(.+)")
_CODE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)

MAX_HISTORY = 5
MAX_FEEDBACK_CHARS = 500
MAX_KB = 3
MAX_ATTEMPTS = 2
MAX_CODE_CHARS = 4000  # sent only on a retry, so steady-state prompt size is unchanged


def _fmt_metrics(metrics: dict) -> str:
    if not metrics:
        return "-"
    parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
    return ",".join(parts)


def _best_metrics(history) -> str:
    best: dict = {}
    for e in history:
        if e.status not in ("ok", "reverted"):
            continue
        for k, v in e.metrics.items():
            if k not in best or v > best[k]:
                best[k] = v
    return _fmt_metrics(best)


def _summarize_history(history, n: int) -> str:
    lines = [
        f"#{e.iter_id} tier{e.tier} {e.status} {_fmt_metrics(e.metrics)} :: {e.hypothesis[:60]}"
        for e in history[-n:]
    ]
    return "\n".join(lines) if lines else "none"


def _kb_query(history, feedback: str | None) -> str:
    # ensemble terms must be seeded: without them retrieval only ever surfaces single-model
    # papers, which silently biases every proposal away from the one direction that works here
    words = ["click", "ranking", "ndcg", "gauc", "kuairand", "short video",
             "ensemble", "blend", "rank aggregation", "decorrelated"]
    for e in history[-3:]:
        words.append(e.hypothesis)
    if feedback:
        words.append(feedback[:200])
    return " ".join(words)


TASK_BRIEF = """You are the proposer inside an autonomous ML research agent competing on KuaiRand-Pure.

DATASET: Kuaishou short-video feed. 1.14M train rows (dates 20220408-0421), 124,909 validation
(0422-0428), 170,588 hidden test (0429-0508). 27K users, 7.6K videos.
LABEL IS `long_view` (0/1), NOT click. The prose calls it click; the official code does not.

METRIC: primary = mean(GAUC, nDCG@5), per user.
  GAUC counts only users with 0 < positives < impressions, weighted by positive count.
  nDCG@5 uses gain 2^rel - 1; users with zero positives score 0 and ARE included in the mean.
TASK: within-user ranking over each user's logged impressions. NOT full-catalog retrieval.
OFFICIAL BASELINE to beat: k=16 FM, fields [user_id, video_id, author_id, tab, dur_bucket].
  validation primary 0.6016 (GAUC 0.6674 / nDCG@5 0.5357)
Reference rungs on validation: random 0.4834, item-popularity 0.5807.
ORACLE CEILING on validation: primary 0.8484 (GAUC 1.0, nDCG@5 0.6968). nDCG cannot reach 1.0
because 27.1% of users are all-negative and score 0 regardless of the model. Judge progress as
a fraction of the 0.6016 -> 0.8484 gap, not against 1.0.

AVAILABLE API (import these; do not reimplement them):
  from pipeline.data import load, FEATURE_CARDINALITIES
  s = load("train")   # or "valid"
  s.X          dict[str, int64 array]  37 categorical features, contiguous ids, 0 = unseen
  s.y          int8 array, long_view -- the scored label
  s.user_id    int64 array   s.video_id  int64 array
  s.aux        dict of post-click signals (is_click, is_like, play_time_ms, ...)
  from pipeline.evaluate import evaluate
  evaluate(user_ids, labels, scores) -> {"primary","gauc","ndcg@5","ndcg@10","recall@50"}

HARD RULES:
  - NEVER use anything from s.aux as an input feature. They are outcomes of the row being
    scored; using them is label leakage and invalidates the result.
  - Train on "train" only. Report metrics on "valid". Never touch test.
  - torch is CPU-ONLY here (2.12.0+cpu). NEVER call .cuda(), .to('cuda'), or use AMP.
  - lightgbm is 4.7, which REMOVED early stopping as a keyword argument EVERYWHERE. Both
    `lgb.train(..., early_stopping_rounds=N)` and `LGBMClassifier.fit(..., verbose=,
    early_stopping_rounds=)` raise TypeError. The only supported form is a callback:
        m = lgb.train(params, dset, num_boost_round=600, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
        preds = m.predict(X, num_iteration=m.best_iteration)
    Put verbosity in params as `"verbose": -1`, not as a fit/train argument.
  - HARD TIME BUDGET: 300 seconds per experiment, killed at the limit and scored as a
    failure. The skeleton below completes in 20s. Stay within ~10x of that. Vectorize in
    torch/numpy; a Python loop over rows, users, or pairs WILL time out on 1.14M rows.
    If a method needs per-user grouping, precompute group boundaries once with argsort,
    never inside the training loop.
  - The script must print, as its FINAL stdout line:
    METRICS {"primary": <float>, "gauc": <float>, "ndcg@5": <float>, "gpu_seconds": <float>}
  - The script MUST ALSO score the test split and save it, so the best iteration can become the
    final submission without a human rebuilding it. Add this at the end, after evaluating valid:

        import os
        out = os.environ.get("ITER_OUT")
        if out:
            te = load("test")
            te_scores = <your model's scores for te, exactly as you scored valid>
            np.save(os.path.join(out, "scores_test.npy"), np.asarray(te_scores, dtype=np.float64))

    Scoring test is allowed; FITTING or SELECTING anything on test is not. Never read te.y.

FEATURE SHAPES: 37 fields, 47,784 total slots. Largest: user_id 30000, video_id 8000,
author_id 7000. Use offset encoding into ONE embedding table -- do not build 37 tables.

WORKING SKELETON. Start from this; it runs. Change the modelling part, not the plumbing.

```python
import json, time, numpy as np, torch, torch.nn as nn
from pipeline.data import load, FEATURE_CARDINALITIES as FC
from pipeline.evaluate import evaluate

t0 = time.perf_counter()
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]   # add/remove here
off = np.cumsum([0] + [FC[f] for f in FIELDS[:-1]]).astype(np.int64)
TOT = sum(FC[f] for f in FIELDS)

def mat(s):                                   # -> int64 (n, len(FIELDS)) offset-encoded
    return np.stack([np.minimum(s.X[f], FC[f]-1) + off[i] for i, f in enumerate(FIELDS)], 1)

tr, va = load("train"), load("valid")
Xtr, ytr = torch.from_numpy(mat(tr)), torch.from_numpy(tr.y.astype(np.float32))
Xva = torch.from_numpy(mat(va))

# per-user groups, precomputed ONCE. Use these for any pairwise/listwise objective;
# grouping inside the training loop is what makes those implementations time out.
gorder = np.argsort(tr.user_id, kind="stable")
gstart = np.flatnonzero(np.r_[True, np.diff(tr.user_id[gorder]) != 0])
gsize = np.diff(np.r_[gstart, len(gorder)])          # rows per user, aligned to gstart
Xg, yg = Xtr[gorder], ytr[gorder]                     # regroup once, index with gstart/gsize

class M(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.emb = nn.Embedding(TOT, k); self.bias = nn.Embedding(TOT, 1)
        nn.init.normal_(self.emb.weight, std=0.01); nn.init.zeros_(self.bias.weight)
    def forward(self, x):                     # x: (B, F) int64 -> returns (B,) logits
        e = self.emb(x)                       # (B, F, k)
        fm = 0.5 * ((e.sum(1) ** 2) - (e ** 2).sum(1)).sum(1)   # (B,)
        return fm + self.bias(x).sum((1, 2))  # (B,)

# RAW IDS vs OFFSET IDS -- this trap has crashed many attempts. `mat()` ADDS per-field
# offsets, so Xtr's video column holds values up to 47,784, not 0..7,999. To build any
# per-video or per-author table, index with the RAW ids and look up with the RAW ids:
#   vid_tr = tr.X["video_id"]                      # raw, 0..FC["video_id"]-1
#   pos = np.bincount(vid_tr, weights=tr.y.astype(float), minlength=FC["video_id"])
#   cnt = np.bincount(vid_tr, minlength=FC["video_id"])
#   rate = pos / np.maximum(cnt, 1)                # train-only statistic
#   feat_va = rate[np.minimum(va.X["video_id"], FC["video_id"]-1)]   # clamp unseen ids
# Never index such a table with a column taken from mat()/Xtr.
#
# ADDING A NUMERIC FEATURE (target encoding, counts, rates). The embedding path takes int64
# ONLY -- passing a float into nn.Embedding is the single most common crash here. Keep the two
# streams separate and combine at the logit:
#   num_tr = torch.from_numpy(np.stack([...], 1).astype(np.float32))   # (n, D) float32
#   self.num = nn.Linear(D, 1)                                          # in __init__
#   def forward(self, x, xn):        # x int64 (B,F), xn float32 (B,D)
#       ...                          # embedding path unchanged, uses x only
#       return fm + self.bias(x).sum((1,2)) + self.num(xn).squeeze(1)
#   and index BOTH in the batch loop: m(Xtr[b], Ntr[b])
# Standardise numeric columns with TRAIN mean/std. Any encoding must be computed from train
# rows only -- using validation rows to build it is leakage and invalidates the score.
#
# SHAPES -- most failed proposals die here. F = len(FIELDS), k = embed dim, B = batch.
#   self.emb(x)      -> (B, F, k)
#   e.flatten(1)     -> (B, F*k)      <- what a deep/MLP tower must consume
#   first Linear     -> nn.Linear(len(FIELDS) * k, hidden)   NOT nn.Linear(k, ...)
#   a Linear tower ends (B, 1) and needs .squeeze(1) -> (B,) before being added to fm
#   but self.bias(x).sum((1, 2)) is ALREADY (B,) -- do NOT .squeeze(1) it, that raises
#   IndexError: Dimension out of range. Only squeeze things that really have a dim 1.
# e.g. adding a deep tower:
#   self.mlp = nn.Sequential(nn.Linear(len(FIELDS)*k, 128), nn.ReLU(), nn.Linear(128, 1))
#   return fm + self.bias(x).sum((1, 2)) + self.mlp(e.flatten(1)).squeeze(1)

m = M(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
lossf = nn.BCEWithLogitsLoss()
for epoch in range(8):
    perm = torch.randperm(len(ytr))
    for i in range(0, len(perm), 8192):
        b = perm[i:i+8192]
        opt.zero_grad(); l = lossf(m(Xtr[b]), ytr[b]); l.backward(); opt.step()

m.eval()
with torch.no_grad():
    sc = torch.cat([m(Xva[i:i+65536]) for i in range(0, len(Xva), 65536)]).numpy()
r = evaluate(va.user_id, va.y, sc)

import os
out = os.environ.get("ITER_OUT")          # save test scores so this run can become the submission
if out:
    te = load("test")
    Xte = torch.from_numpy(mat(te))
    with torch.no_grad():
        ts = torch.cat([m(Xte[i:i+65536]) for i in range(0, len(Xte), 65536)]).numpy()
    np.save(os.path.join(out, "scores_test.npy"), ts.astype(np.float64))

print("METRICS", json.dumps({"primary": r["primary"], "gauc": r["gauc"],
                            "ndcg@5": r["ndcg@5"], "gpu_seconds": time.perf_counter() - t0}))
```

EXHIBIT A -- ENSEMBLING IS A FIRST-CLASS PROPOSAL, NOT A FALLBACK. Across ~12 iterations, no
single model beat baseline by more than noise, but BLENDING models that are each individually
WORSE than baseline did:
  FM (5-seed, early stopped)           valid 0.6021   (loses to baseline alone)
  LightGBM binary, 37 features         valid 0.6010   (loses to baseline alone)
  LightGBM lambdarank                  valid 0.5998   (loses to baseline alone)
  rank-blend of the three, 0.6/0.3/0.1  valid 0.6045, test 0.5984   <- +0.0038 over baseline
Mechanism: the three make DECORRELATED errors -- FM is stronger on GAUC, the GBDTs on nDCG@5,
lambdarank optimises a different objective entirely. Blend on RANKS
(np.argsort(np.argsort(scores))), never raw scores -- the components' scores live on
incompatible, uncalibrated scales, so averaging them directly is meaningless.
A valid proposal may TRAIN A SECOND, DIFFERENTLY-BIASED COMPONENT (different feature set,
objective, or model family) and blend it with an existing one, instead of just swapping one
architecture for another. Pattern, consistent with the skeleton above (`sc_fm` = the FM's
validation scores computed the same way as `sc` above; lightgbm 4.7, CPU, is installed):

  import lightgbm as lgb
  raw = lambda s, fs: np.stack([np.minimum(s.X[f], FC[f]-1) for f in fs], 1)  # RAW ids, no offset
  cat_fields = ["user_id", "author_id", "tab", "duration_bucket", "onehot_feat3"]  # vary this
  gbm = lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.05, verbose=-1)
  gbm.fit(raw(tr, cat_fields), tr.y, categorical_feature=list(range(len(cat_fields))))
  sc_gbm = gbm.predict_proba(raw(va, cat_fields))[:, 1]
  rk = lambda x: np.argsort(np.argsort(x))
  blend = 0.6 * rk(sc_fm) + 0.4 * rk(sc_gbm)                # weights are a free hyperparameter
  r = evaluate(va.user_id, va.y, blend)
  print("METRICS", json.dumps({"primary": r["primary"], "gauc": r["gauc"],
                              "ndcg@5": r["ndcg@5"], "gpu_seconds": time.perf_counter() - t0}))

A LAMBDARANK component (the most decorrelated signal measured, rho ~0.65 vs everything else)
needs QUERY GROUPS or it raises "The NDCG metric requires query information". Rows must be
SORTED BY USER and the group array gives each user's row count, in that same order:

  def grouped(s, fs):                        # -> (X sorted by user, y sorted, group sizes, order)
      o = np.argsort(s.user_id, kind="stable")
      u = s.user_id[o]
      starts = np.flatnonzero(np.r_[True, np.diff(u) != 0])
      sizes = np.diff(np.r_[starts, len(o)])
      return raw(s, fs)[o], s.y[o], sizes, o

  Xg, yg, gtr, _ = grouped(tr, cat_fields)
  Xv, yv, gva, ov = grouped(va, cat_fields)
  d  = lgb.Dataset(Xg, yg, group=gtr, categorical_feature=list(range(len(cat_fields))))
  dv = lgb.Dataset(Xv, yv, group=gva, reference=d)
  par = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[5], learning_rate=0.05,
             num_leaves=127, verbose=-1)
  mr = lgb.train(par, d, 600, valid_sets=[dv],
                 callbacks=[lgb.early_stopping(40, verbose=False)])
  sc_lam = np.empty(len(va.y))               # UNSORT back to original row order before blending
  sc_lam[ov] = mr.predict(Xv, num_iteration=mr.best_iteration)

HOW MANY COMPONENTS ARE SAFE (measured with user-grouped 5-fold CV on validation):
  3 components  in-sample 0.6048 / honest out-of-fold 0.6042  (optimism +0.0006)
  10 components in-sample 0.6052 / honest out-of-fold 0.6045  (optimism +0.0007)
  Tuning blend weights on validation is SAFE up to ~8-10 components at this data size; the
  optimism stays an order of magnitude below the effect sizes that matter. More components from
  DIFFERENT families is the reliable way up. Do not stop at three.

WHAT MAKES A BLEND WORK HERE (measured on validation rank-correlations):
  Components cluster into FAMILIES. Within a family, rank-correlation is 0.98-0.99, so adding a
  second member of the same family adds almost nothing -- this is why averaging 5 FM seeds gains
  only +0.0007. Across families it drops to ~0.65, and that is where the gain comes from.
  The families found so far: FM-style factorisation, gradient-boosted trees with a pointwise
  objective, a RANKING objective (lambdarank correlates only 0.645-0.72 with everything else --
  the single most decorrelated signal), and naive item popularity.
  A component earns its place by being DECORRELATED, not by being individually strong: the
  weakest LightGBM variant is still worth including because it disagrees with the others.
  But decorrelation alone is not enough -- a 50/50 blend of the two least-correlated components
  LOSES. You need decorrelation AND comparable strength, which means weighted blending.
  An equal-weight blend of 4-6 components drawn from DIFFERENT families is a strong, cheap
  starting point. Blend on RANKS, never raw scores.
  Weight-tuning on validation is safe here: measured optimism (in-sample minus honest
  user-grouped out-of-fold) is only +0.0002.

NOT AVAILABLE -- do not propose these, they cannot be built on this data:
  - Per-user behaviour SEQUENCES. `s.X` is one scalar id per feature per row; there is no
    history array. So DIN, DIEN, BST, SASRec and GRU4Rec are NOT implementable here.
    Attention over a user's other rows is possible via the group arrays above, but that is a
    different method -- say so if you mean it.
  - Anything needing the test split, or any external dataset.

YOUR GOAL RIGHT NOW. The submission is built from YOUR best iteration, so a result only
counts if YOUR script produces it. The blend described in EXHIBIT A is the best known approach
(+0.0038) but it was measured OUTSIDE this loop, so it does not exist as far as the submission
is concerned. **Reproducing and extending that blend in your own script is the single highest
value thing you can do, and it is expressly NOT "re-running something already measured".**
Build it, then improve it: add a differently-biased component, or tune the weights.

ALREADY MEASURED -- these are prior results, not forbidden moves. Do not re-run a SINGLE model
from this list expecting it to win on its own; DO use them as blend components:
  - Skeleton as written (5 fields, 8 epochs): primary 0.6003.
  - Same model with all 37 fields: primary 0.6019 (+0.0016, 5x slower). Adding every feature
    is NOT the win.
  - Official FM baseline: 0.6016. Beating it by less than 0.002 is inside noise (seed std
    0.0008); aim for a change with a mechanism, not a reshuffle.
  - Within-user PAIRWISE ranking loss (BPR-style, -logsigmoid(s_pos - s_neg)) is WORSE than
    pointwise logloss here: 0.5935 (8 pairs/user), 0.5955 (32/user), 0.5804 (32/user, 16
    epochs). Do not propose plain pairwise/BPR again. Likely cause: 33% of impressions are
    positive, so this is not sparse implicit feedback, and GAUC rewards calibrated pointwise
    scores. A listwise loss that keeps calibration (e.g. per-user softmax CE blended with
    logloss) is still untested and remains a legitimate proposal.

  - Deeper/wider architectures on this feature set do NOT help: DeepFM 0.5904-0.6010,
    NFM 0.5598, DCN 0.6024, attention-over-user-rows 0.5941. The FM interaction term is
    already close to what these 5 fields support.
  - Early stopping on validation (max 40 epochs, patience 4) instead of a fixed 8 epochs:
    no gain (0.6014 mean over 5 seeds, std 0.0003).
  - 5-seed ensembling (score-average or rank-average), SAME architecture: 0.6020, i.e. +0.0007.
    Real but tiny -- do not confuse with EXHIBIT A's cross-model blend, which is bigger.
  - EXHIBIT A's three components and their 0.6/0.3/0.1 blend (five numbers, given above) are
    measured -- do not re-run them. DO propose a new, differently-biased component (feature
    subset, objective, or family) to add to or swap into the blend.

  - Listwise softmax cross-entropy: 0.5990. DCN-V2: 0.5931. Ranking objectives keep losing
    to plain logloss, consistent with GAUC rewarding calibrated pointwise scores.
  - DURATION IS A WEAK SIGNAL, measured: long_view rate by duration decile runs 0.273-0.376
    with no monotonic trend (shortest videos have the LOWEST rate). `long_view` is already
    defined relative to duration, so the normalisation is baked into the label. Do not build
    elaborate duration features expecting a large gain.

WHERE THE HEADROOM MIGHT BE. Everything model-side lands within +-0.005 of baseline, so
capacity is not the answer -- 1.14M rows over 26K users (~43 rows/user) will not support a
bigger model. Untested directions, in rough order of plausibility:
  - A new blend component (see EXHIBIT A) -- the one directional win measured so far. Any of
    the directions below is a stronger proposal as a BLEND COMPONENT than as a single-model
    replacement for the FM.
  - Train-only target/count encodings for VIDEO and AUTHOR (per-item positive rate, exposure
    count). Must be computed on train rows only; leaking validation into them is a silent bug.
  - Per-user score calibration or centring. GAUC and nDCG are both within-user, so any effect
    constant within a user cancels -- capacity spent on it is wasted.
  - Regularisation of the existing FM (weight decay, embedding dropout, smaller k) rather
    than more capacity.
  - A calibration-preserving objective: logloss BLENDED with a listwise term, not replacing it.
Be honest in the hypothesis about the mechanism you expect. A proposal that just swaps in
another named architecture is very unlikely to help and wastes the budget.

WHAT TO PROPOSE, IN ORDER:
  1. If your history does NOT yet contain a working BLEND, build one. That is the recommended
     opening move, not an advanced step: train two or three differently-biased components in a
     single script (e.g. the skeleton FM plus a LightGBM on raw ids), rank-transform each with
     np.argsort(np.argsort(x)), combine with weights, evaluate the combination. EXHIBIT A has
     the pattern. The skeleton alone scores ~0.600; a blend measured +0.0038 over baseline.
  2. Once a blend exists in your history, improve it incrementally: add one differently-biased
     component, drop a redundant one, or retune the weights.
  3. Only propose a single-model change if you have a specific mechanism that the blend cannot
     absorb.

State which stage you target (features / architecture / loss / training / evaluation /
ensemble). Prefer changes with a clear mechanism for
why they should raise a per-user ranking metric over a logloss-trained FM. Emit the COMPLETE
modified script, not a diff."""


class LLMProposer:
    """LLM-backed Proposer. `complete` is injected so this stays testable offline."""

    def __init__(self, complete: CompleteFn, kb_papers: list[dict] | None = None,
                 max_history: int = MAX_HISTORY):
        self.complete = complete
        self.kb_papers = kb_papers
        self.max_history = max_history

    def propose(self, history, blacklist: set[str], feedback: str | None, tier: int) -> Proposal | None:
        note = None
        tokens_in = tokens_out = 0
        for _ in range(MAX_ATTEMPTS):
            prompt = self._build_prompt(history, blacklist, feedback, tier, note)
            text, ti, to = self.complete(prompt)
            tokens_in += ti
            tokens_out += to
            parsed = self._parse(text)
            if parsed is None:
                note = ("Previous reply was not parseable. Reply with exactly one "
                        "'HYPOTHESIS: <line>' followed by one ```python code fence.")
                continue
            hyp, code = parsed
            if hyp in blacklist:
                note = f'"{hyp}" is blacklisted, propose a different hypothesis.'
                continue
            return Proposal(hyp, code, tokens_in, tokens_out)
        return None

    def _build_prompt(self, history, blacklist, feedback, tier, note) -> str:
        best = _best_metrics(history)
        hist_lines = _summarize_history(history, self.max_history)
        bl = ", ".join(sorted(blacklist)) or "none"
        # query-driven retrieval alone kept surfacing single-model papers, because the query is
        # dominated by recent hypotheses; pin the measured-best direction so it is always seen
        kb_entries = retrieve(_kb_query(history, feedback), k=MAX_KB - 1, papers=self.kb_papers)
        pinned = retrieve("rank aggregation ensemble blend decorrelated", k=1, papers=self.kb_papers)
        for x in pinned:
            if all(x["id"] != e["id"] for e in kb_entries):
                kb_entries = pinned + kb_entries
        kb_lines = "\n".join(
            f"- {p['id']}: {p['title']} ({p['year']}) - {p['expected_effect']}" for p in kb_entries
        ) or "none"
        fb_block = ""
        if feedback:
            # without the crashed source the model rewrites from scratch and introduces a
            # different bug instead of fixing the failing line
            last_code = next((e.diff for e in reversed(history)
                              if e.status in ("failed", "blacklisted") and e.diff), "")
            fb_block = (
                "\nYOUR PREVIOUS ATTEMPT CRASHED. Fix it and keep the same hypothesis.\n"
                f"Traceback:\n{feedback[:MAX_FEEDBACK_CHARS]}\n"
            )
            if last_code:
                fb_block += f"\nThe script that crashed:\n```python\n{last_code[:MAX_CODE_CHARS]}\n```\n"
        note_block = f"\nNOTE: {note}\n" if note else ""
        return (
            TASK_BRIEF + "\n\n"
            f"Tier: {tier}\n"
            f"Current best metrics: {best}\n"
            "Recent outcomes (most recent last):\n"
            f"{hist_lines}\n"
            f"Blacklisted hypotheses (do not repeat): {bl}\n"
            f"{fb_block}"
            "Relevant published methods:\n"
            f"{kb_lines}\n"
            f"{note_block}"
            "Respond EXACTLY as:\n"
            "HYPOTHESIS: <one line, name the method you draw on>\n"
            "```python\n"
            "<complete runnable script; final stdout line must be METRICS {...}>\n"
            "```\n"
        )

    def _parse(self, text: str) -> tuple[str, str] | None:
        hm = _HYP_RE.search(text)
        cm = _CODE_RE.search(text)
        if not hm or not cm:
            return None
        hyp = hm.group(1).strip()
        code = cm.group(1).strip()
        if not hyp or not code:
            return None
        return hyp, code
