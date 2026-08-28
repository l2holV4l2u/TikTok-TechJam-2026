# Can we use `log_random_4_22_to_5_08_pure.csv`?

Scoping investigation. Nothing implemented, no existing run or submission touched.

## Verdict

**Partly legal, but the method it was proposed for is the wrong method.** The blocking issue is
not permission — it is that inverse-propensity weighting corrects toward a distribution we are
not scored on.

## 1. Is it allowed?

The problem statement's out-of-scope list is narrow:

> No external training data or pretrained weights trained on these benchmarks' test labels
> No hidden-test access during development (train + validation only)

The random log is **not external data** — it ships inside KuaiRand-Pure, and §2.4 explicitly
advertises it: *"KuaiRand's randomized-exposure data also enables off-policy / counterfactual
evaluation (OPE)."* So the first clause does not bite. The second does, and the measurements
say exactly how much:

| | |
|---|---|
| rows | 1,186,059 (all `is_rand=1`) |
| in validation window 22-28 Apr | 288,338 |
| **in hidden-test window 29 Apr - 8 May** | **897,721 (75.7%)** |
| test users it covers | 23,875 / 23,875 (**100%**) |
| test videos it covers | 5,982 / 5,982 (**100%**) |
| scored test pairs it contains | 58 / 165,361 (**0.04%**) |

The pair overlap is negligible, so the file does not hand over answers to scored rows. That is
the wrong thing to check, though. 897,721 labelled outcomes from inside the scored window,
covering **every** test user and **every** test video, is outcome data from the evaluation
period. Any per-item or per-user statistic fitted on it is a measurement of how the scored
population behaved during the scored days. That is what "train + validation only" exists to
forbid, whether or not a single scored pair is copied.

**Safe subset: the 288,338 rows dated 22-28 Apr.** They fall in the validation window, which the
organizers say teams may develop on. **Unsafe: the 897,721 rows dated 29 Apr onward** — 76% of
the file, and the part that makes it look valuable.

## 2. The method is wrong regardless

IPS and doubly-robust corrections exist to move a biased logging distribution toward the
unbiased one, and are right when you are *evaluated* on the unbiased distribution. Here we are
not:

| split | long_view rate |
|---|---|
| train (8-21 Apr, standard log) | 0.3366 |
| validation (22-28 Apr, standard log) | 0.3133 |
| hidden test (29 Apr - 8 May, standard log) | 0.3135 |
| **random-exposure log** | **0.0850** |

Training, validation and test all come from the same biased production log and sit within 0.023
of each other. The random log is **3.7x away from all three**. Reweighting training toward
random exposure moves the model away from the distribution it is scored on, not toward it.

The task compounds this. The metric ranks **within each user's logged impressions** — the
candidate list is given, we never choose what to expose. Exposure-bias correction answers "what
would have happened had we shown different items", which this task never asks.

Expected effect of naive IPS here: **negative**.

## 3. Bounded-safe variant, if anyone still wants it

Use only the 288,338 validation-window rows, and only as an *auxiliary signal*, never as a
reweighting of the main objective:

- Under uniform exposure an item's long_view rate estimates intrinsic appeal free of the
  policy's selection effect. As an extra feature it is a different quantity from the biased
  popularity feature already available, and it may survive the date boundary better.
- Or as an auxiliary multi-task head: predict long_view on random-exposure rows alongside the
  main objective, sharing embeddings. Regularisation, not correction.

Both need a hard guard: **assert no row with `date >= 20220429` is ever read.** That assertion
is the whole safety argument and belongs in the loader, not in a comment.

## 4. Expected movement: small, probably inside noise

`docs/generalisation-ceiling.md` measured the binding constraint as the shift across the date
boundary — zero-positive users 5.1% -> 30.3%, median rows per user 59 -> 7. That is a change in
**how many impressions each user has**, not in which items were exposed. Exposure debiasing does
not address it.

Given a safe subset of 288K rows against 1.14M training rows, and a mechanism aimed at a problem
we do not have, our estimate is **under 0.002 — inside seed noise**. We would not describe this
as the most promising unexplored direction; the earlier note in this repo that called it that
was written before these numbers existed, and is superseded.

## 5. What it would cost if greenlit

- `pipeline/data.py`: a fourth source with a date-range guard and an assertion on the upper
  bound; a new cache split (`random_valid`); `Split` unchanged.
- `agent/proposer.py`: the brief would have to describe the log, its exposure semantics, and the
  date restriction — otherwise the agent cannot use it correctly.
- `agent/facts.py`: expose the safe row count so the brief stays measured rather than asserted.
- Roughly a day, against an expected sub-noise gain. Not recommended before the deadline.
