# Portfolio search: summary, revision, and implementation plan

A design for running *n* solution lineages at once instead of one, with an archive of retired
lineages, a consultant that synthesises across them, and an ensemble built from what the archive
collects.

---

## 1. The flow, as you proposed it

> The loop starts by designing architectures, which are sent to an agent and prioritised in a tree.
> That goes into a loop of writing code, improving the model and scoring it. While a score keeps
> improving the agent moves to the next iteration, sends the result back to state a hypothesis, and
> plans the next improvement. If an agent fails to improve twice it moves to another child in the
> tree and tries a different architecture. Each agent that converges under N=3 has its code and score
> stored and ranked, and is replaced by a new agent or by a revived top scorer — alternating, one
> revival for every three new architectures. A consultant agent that knows the literature talks to
> every agent during its loop and looks for improvements applicable to the stored top scorer. The
> run ends when no agent improves any more, or the time limit is hit.

## 2. The flow, revised

> The loop starts by having the model **inspect the data**, then **reproduce the official baseline** —
> that script becomes the tree's root and the first occupant of every slot. From then on the run
> advances in **turns**, not per-agent iterations. Each turn, **three slots** each pick their own
> least-explored live node, and each is told **what its siblings are attempting this turn** and asked
> for a different direction. Three prompts go out, three complete scripts come back, and the harness
> runs them **concurrently in isolated subprocesses**. For each, it **recomputes the score itself**
> from saved arrays rather than trusting what the script printed, and runs a **critic** that rejects
> implausible or leaky results. The turn's score is the **best of the three**, and that single number
> — not any slot's private curve — drives the **one** convergence counter the rules recognise. A slot
> that fails to beat its own best twice is **archived**, not stopped: its code, its scores and a note
> on why it stalled go into the archive, and the slot is refilled — a fresh draft by default, a
> **revived archive member every third refill**, chosen for score *and* for being uncorrelated with
> whatever is live. One **consultant call** per turn reads the archive and all three slots and emits
> a shared belief set plus one short note per slot. The harness then **blends the whole archive plus
> the live candidates**, choosing weights on one half of validation and confirming on the other so a
> wider search cannot buy a validation score that will not transfer. The run stops when the system
> best has not gained more than 0.002 across three turns, or on the 6-hour ceiling. What is submitted
> is the **saved test-score array** of the validation-best checkpoint at that stopping point — not a
> model.

## 3. What changed, and why

| Your design | Revised | Why |
|---|---|---|
| Each agent carries its own N=3 counter; the run continues after one converges | **One** system-level counter over the per-turn best | The spec says *"**a run** is converged when validation score has not improved by more than ε over the last N=3 consecutive iterations"* — singular. Continuing past that point produces a checkpoint that cannot legitimately supply the submission. The repo already truncated a run at its own convergence point and lost 0.0003 on test rather than exploit this. |
| n agents, each running the loop | n **slots**, one loop, n scripts per turn | Same compute, same breadth, but it never leaves the rule. It is also the existing `CANDIDATES` contract ("one iteration is one script, not one model") lifted one level up. |
| Diversity assumed | Diversity **enforced by sibling disclosure and measured by rank correlation** | Three slots given the same brief and the same EDA will converge on the same family. The repo measured what that costs: components correlated at 0.94+, MMoE at 0.9888 against DeepFM, and every blend gained nothing. |
| Assign each agent an architecture | Tell each slot what its **siblings** are doing and require a different direction | Naming architectures in the prompt is a human prior on method space. `pipeline/models.py` is deliberately unreferenced and 0 of 358 generated scripts import it. A negative constraint adds no prior — and it is what `_BROADEN_PARENT` already does across time. |
| Archive = ranked storage for revival | Archive = **ensemble pool** first, revival source second | A set of converged, decorrelated models is exactly the input the harness ensembler lacks. This is the mechanism that attacks the 0.94 ceiling. |
| Revive the top scorer | Revive `argmax(score − λ · max_correlation_with_live)` | Reviving the top scorer when it correlates 0.97 with a live slot spends a slot to learn nothing. |
| Consultant knows the literature and advises each agent | Consultant **synthesises across slots and the archive**; one call per turn | A literature oracle is a human-prior proxy and risks the Autonomy claim (20% of the score). Slots already receive the full 28-paper catalogue. What they lack is knowledge of each other. One call instead of n also keeps Feasibility in the same tier. |
| Select the best on validation | Select on **validation fold A**, confirm on **fold B** (user-grouped) | Selecting the max over more candidates inflates validation without transferring. Measured: 8 → 18 candidates adds **+0.0003** of pure selection noise; 50 candidates adds +0.0018. Tolerable at n=3, not at n=8. |
| "Run ends when no agent improves" | Ends on the **system** convergence rule or the 6h ceiling | Same intent, expressed in the units the rules use. |

### The number that justifies n = 3 and forbids n = 8

Selecting the max of *k* candidates on a validation set with σ = 0.0008:

| candidates *k* | validation inflation from selection alone |
|---|---|
| 8 (today) | +0.00114 |
| **18 (n=3, ~6 turns)** | **+0.00145** |
| 24 | +0.00156 |
| 50 | +0.00180 |

The whole effect being chased is ≈ +0.005 on test. n=3 costs +0.0003 of illusion; going wider starts eating the result.

### The tension to keep in view

Gome ([arXiv:2603.01692](https://arxiv.org/abs/2603.01692)) finds directed updates beat exhaustive
tree search *at frontier model strength*. FML-bench ([arXiv:2605.17373](https://arxiv.org/abs/2605.17373))
finds breadth wins when improvements are sparse — which is this benchmark — but also reports that
*"a simple greedy hill-climber nearly matches the best-performing tree-search agent."*

**So build the portfolio for the decorrelated ensemble pool, not for the search breadth.** If the
slots turn out to correlate above 0.95, the honest conclusion is that you have three expensive
copies of one agent. Phase 2 exists to find that out by turn 2 rather than at the end.

---

## 4. Implementation phases

Six phases. Each leaves a working, rule-compliant system, and each can be abandoned without undoing
the previous one. **Phase 2 is a go/no-go gate.**

| Phase | Delivers | Risk if skipped |
|---|---|---|
| 0 | Safety rails: artifact isolation, ledger fields, fold split | Parallelism corrupts shared state |
| 1 | Parallel turns, one convergence curve | — |
| 2 | Sibling disclosure + correlation gate | Portfolio is 3× cost for 1× information |
| 3 | Archive, slot retirement, refill | No memory across lineages |
| 4 | Consultant | Slots duplicate each other's conclusions |
| 5 | Portfolio ensemble + selection guard | Winner's curse, and the archive earns nothing |
| 6 | Reporting | Deliverables do not show the portfolio |

---

## Phase 0 — Safety rails

**Goal.** Make the current single-slot loop safe to parallelise. No behavioural change.

### Specification

```python
# agent/loop.py — per-slot artifact isolation
artifacts_root = (workdir.parent / "artifacts").resolve()
slot_artifacts = artifacts_root / f"slot_{slot_id}"     # writable, private
shared_artifacts = artifacts_root / "shared"            # published incumbent, read-only to slots
extra_env = {"ITER_OUT": ..., "RUN_ARTIFACTS": str(slot_artifacts),
             "SHARED_ARTIFACTS": str(shared_artifacts)}
```

```python
# agent/ledger.py
@dataclass
class Entry:
    ...                      # existing fields unchanged
    slot_id: int | None = None
    turn: int | None = None
```

```python
# agent/selection.py  (new)
def split_validation(user_id, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (fold_a, fold_b), user-grouped so no user spans both."""

def accept(candidate, incumbent, fold_a, fold_b, epsilon: float = 0.002) -> bool:
    """Win on A by > epsilon AND not lose on B by more than epsilon."""
```

### Steps

1. Split `RUN_ARTIFACTS` into `slot_{k}/` and `shared/`. `_publish_incumbent` writes to `shared/`; the brief's "REUSABLE TRUSTED INCUMBENT" block points at `$SHARED_ARTIFACTS`.
2. Add `slot_id` and `turn` to `Entry`, both defaulting to `None` so existing ledgers still parse.
3. Create `agent/selection.py`; `split_validation` reuses `blend.weights.user_folds(user_ids, 2, seed)`.

### Tests — `tests/test_selection.py`

- `test_folds_never_split_a_user`
- `test_folds_are_deterministic_given_seed`
- `test_accept_rejects_a_fold_a_win_that_loses_on_fold_b`
- `test_accept_passes_a_genuine_win_on_both`
- `test_old_ledger_lines_still_parse_without_slot_id`

### Exit criteria

Full suite green. A single-slot run reproduces its previous score to within seed noise.

---

## Phase 1 — Parallel turns

**Goal.** *n* scripts per turn, one convergence curve. Portfolio logic not yet present: every slot
edits from the same tree with the existing `select("sweep")`.

### Specification

```python
N_SLOTS = 3

@dataclass
class Slot:
    slot_id: int
    parent: Node | None = None
    stale: int = 0
    best: float = float("-inf")
    origin: str = "fresh"        # "fresh" | "revived"
    seed_note: str = ""
```

```
per turn:
    if elapsed >= wall_clock_limit:              stop("wall_clock")
    if converged(system_best_curve, 3, 0.002):   stop("converged")   # UNCHANGED

    proposals = [propose(slot) for slot in slots]          # SEQUENTIAL LLM calls
    results   = ThreadPoolExecutor(N_SLOTS).map(run, ...)  # PARALLEL execution
    for slot, res in zip(slots, results):
        metrics = SavedScoresEvaluator().evaluate(res, slot_iter_out)
        ... critic ... tree.add / tree.record_child / tree.record_failure

    system_best = max(system_best, best_scored_this_turn)
    system_best_curve.append(system_best)
```

**LLM calls stay sequential.** `llm.py` enforces `MIN_REQUEST_INTERVAL_S = 21` and its daily-cap
failover mutates shared key state; concurrent calls will trip rate limits and race.

**Iteration accounting.** Record both `turns` and `scripts` in `run_meta.json`. At n=3 converging in
~6 turns that is 18 scripts — under the 50 cap on either reading. Report both.

### Steps

1. Introduce `Slot` and a `slots: list[Slot]` of length `N_SLOTS`, all starting at the baseline node.
2. Extract the body of the current improve branch into `run_one_slot(slot, proposal) -> SlotResult`.
3. Wrap execution in `ThreadPoolExecutor(max_workers=N_SLOTS)`. Each slot gets its own `iter_out`.
4. Append one ledger `Entry` per slot per turn, carrying `slot_id` and `turn`.
5. `system_best_curve.append(system_best)` **once per turn**, after all slots report.
6. Add `--slots N` to `run_agent.py`, default 1 so the old path is the default until Phase 2 passes.

### Tests — `tests/test_portfolio.py` and `agent/demo.py`

- `test_three_slots_produce_three_ledger_entries_per_turn`
- `test_convergence_counter_advances_once_per_turn_not_once_per_slot`
- `test_a_slot_crash_does_not_stop_the_other_slots`
- `test_slots_write_to_disjoint_artifact_directories`
- `test_slots_equal_one_reproduces_the_sequential_loop` — same seed, same ledger

### Exit criteria

`--slots 3 --dry-run` completes. Wall-clock per turn ≤ 2× single-slot (expect 1.5–2×, not 3× — the
scripts are CPU-bound and contend).

### Rollback

`--slots 1`.

---

## Phase 2 — Diversity, and the go/no-go gate

**Goal.** Make the slots explore different families, and **measure whether they do**.

### Specification

```python
# agent/proposer.py — new block, improve phase only, placed with _BROADEN_PARENT
_SIBLINGS = """RUNNING IN PARALLEL WITH YOU THIS TURN:
{siblings}

These are being tried right now by other lines of work. Propose something in a different family or
targeting a different stage. A variation on any of the above is not a different direction, however
it is described."""
```

```python
# agent/portfolio.py
def pairwise_rank_correlation(score_arrays: dict[int, np.ndarray],
                              user_id) -> dict[tuple[int, int], float]:
    """Within-user rank correlation between slots' validation predictions."""

CORR_ALERT = 0.95
```

### Steps

1. Add the `_SIBLINGS` block to the improve prompt, listing the other slots' hypotheses for this turn.
2. After each turn, compute pairwise within-user rank correlation over the slots' `scores_valid.npy` — reuse `ensemble._within_user_rank`.
3. Append to `runs/<id>/portfolio.jsonl`: turn, per-pair correlation, mean, max.
4. Surface `mean_correlation` in the consultant/belief context so the system can see its own redundancy.

### Tests

- `test_sibling_block_lists_every_other_slot_and_not_itself`
- `test_sibling_block_absent_when_slots_equal_one`
- `test_correlation_of_identical_predictions_is_one`
- `test_correlation_logged_once_per_turn`

### Exit criteria — **this is the decision point**

Run 3 turns at `--slots 3`. Then:

- **mean pairwise correlation < 0.90** → proceed to Phase 3.
- **0.90–0.95** → tighten `_SIBLINGS`, re-run once, then decide.
- **> 0.95** → **stop**. The portfolio is three copies of one agent. Keep Phases 0–1 (parallelism is still useful for wall-clock) and abandon 3–5. Record the measurement; a negative result here is a genuine finding and belongs in the write-up.

---

## Phase 3 — Archive and refill

**Goal.** A slot that stalls is recycled, not stopped. Its work is kept.

### Specification

```python
SLOT_PATIENCE    = 2      # slot-local turns without a slot-best gain
FRESH_PER_REVIVE = 3      # revive on every 3rd refill
LAMBDA_CORR      = 0.5    # decorrelation weight in revival choice

@dataclass
class ArchiveEntry:
    entry_id: int
    slot_id: int
    turn_retired: int
    hypothesis: str
    code: str
    valid_scores: np.ndarray
    test_scores: np.ndarray
    primary: float
    note: str                 # consultant's note on why it stalled

class Archive:
    def add(self, slot, result, note: str) -> ArchiveEntry: ...
    def top(self, k: int = 5) -> list[ArchiveEntry]: ...
    def summary(self) -> str: ...          # for the prompt
    def best_revival(self, live_scores, lam: float = LAMBDA_CORR) -> ArchiveEntry | None:
        """argmax(primary - lam * max within-user rank corr with any live slot)."""
```

```python
def refill(archive, slots, state) -> Slot:
    state.fresh_since_revive += 1
    exhausted = state.no_untried_direction
    if archive and (state.fresh_since_revive >= FRESH_PER_REVIVE or exhausted):
        state.fresh_since_revive = 0
        e = archive.best_revival(live_scores_of(slots))
        return Slot(parent=node_of(e), origin="revived", seed_note=e.note)
    return Slot(parent=None, origin="fresh")     # drafts from scratch
```

**Slot stagnation never stops the run.** It only triggers `archive.add` + `refill`.

### Steps

1. Add `agent/portfolio.py::Archive` and `ArchiveEntry`; persist to `runs/<id>/archive.jsonl` plus `archive/entry_{id}_{valid,test}.npy`.
2. Track `slot.stale`: reset on a slot-best gain > ε, else increment.
3. On `slot.stale >= SLOT_PATIENCE`: archive and refill.
4. Implement `refill` with the alternation above; a revived slot's prompt carries `seed_note` and the revived source code as its parent.
5. Log every refill to `portfolio.jsonl` with the reason and the chosen entry.

### Tests

- `test_slot_archives_after_two_non_improving_turns`
- `test_archiving_a_slot_does_not_stop_the_run`
- `test_refill_is_fresh_twice_then_revived_on_the_third`
- `test_revival_prefers_a_decorrelated_entry_over_a_higher_scoring_correlated_one`
- `test_revival_falls_back_to_fresh_when_the_archive_is_empty`
- `test_all_slots_archived_in_one_turn_still_refills_to_n`

### Exit criteria

A 10-turn run archives ≥ 2 entries, performs ≥ 1 revival, and the system convergence curve is
unaffected by slot-level churn.

---

## Phase 4 — Consultant

**Goal.** Replace *n* independent belief revisions with one synthesis across slots and archive.

### Specification

```python
# agent/consultant.py
def revise(complete, archive, slots, turn_results, budget, catalogue) -> tuple[Knowledge, dict[int, str]]:
    """One LLM call per turn.

    Returns the shared belief set and one short note per slot_id. A note may say what a sibling
    is already covering, what the archive has ruled out, or which catalogue method composes with
    a stored result. It must not assert a dataset finding that no iteration measured.
    """
```

Reuses `knowledge.Claim` / `_coerce` unchanged. Notes are capped at 300 chars and land in the
proposer's existing `seed_note` slot.

### Steps

1. Create `agent/consultant.py`, generalising `knowledge.revise` to take the archive and all slots.
2. Replace the per-turn `_revise` call with a single consultant call.
3. Feed each returned note into that slot's next prompt.
4. Pass `mean_correlation` from Phase 2 into the consultant prompt so it can act on redundancy.

### Tests

- `test_consultant_makes_exactly_one_call_per_turn`
- `test_every_live_slot_receives_a_note`
- `test_malformed_reply_leaves_beliefs_and_notes_unchanged` (mirrors `knowledge.demo`)
- `test_consultant_prompt_carries_no_human_findings` — extend the `test_proposer` guard to this module's prompt constant

### Exit criteria

Tokens per turn ≤ 2.5× the single-slot baseline. Beliefs still revise (claims change status across
turns, not merely accumulate).

---

## Phase 5 — Portfolio ensemble and selection guard

**Goal.** Turn the archive into score. This is where the design is expected to pay.

### Specification

```python
# agent/ensemble.py
def blend_portfolio(archive, live_candidates, valid, fold_a, fold_b,
                    max_members: int = 5) -> dict:
    """Greedy forward selection over within-user ranks.

    Start from the incumbent. Repeatedly add the member that most improves fold-A primary; stop
    when no addition gains > epsilon on fold A, or `max_members` is reached. Confirm on fold B:
    if the selected blend loses on B by more than epsilon, fall back to the incumbent. Weights
    are chosen on validation only and applied unchanged to test.
    """
```

Greedy forward selection rather than a weight grid: it scales to an archive of any size, is
deterministic, and needs no optimiser.

### Steps

1. Generalise `retain_or_blend` to `blend_portfolio` over `{incumbent} ∪ archive ∪ live`.
2. Selection on fold A, confirmation on fold B, via `agent/selection.py::accept`.
3. Apply the identical member set and weights to test; never re-select there.
4. Log per turn to `harness_ensembles.jsonl`: members chosen, fold-A gain, fold-B delta, accepted or rejected.

### Tests

- `test_blend_never_scores_below_the_incumbent_on_fold_a`
- `test_blend_rejected_when_it_loses_on_fold_b`
- `test_test_scores_use_the_same_members_and_weights_as_validation`
- `test_duplicate_members_add_nothing` — two identical arrays must not both be selected
- `test_blend_is_deterministic_across_runs`

### Exit criteria

On a completed run, `blend_portfolio` beats the best single slot on **fold B**, not only fold A.
If it does not, the archive is not decorrelated enough and Phase 2's verdict was optimistic.

---

## Phase 6 — Reporting

**Goal.** Make the portfolio visible in the graded deliverables.

### Steps

1. `report_run.py`: render one lane per slot in the iteration table (`turn`, `slot`, score, status, hypothesis).
2. Render the archive: entry, retiring turn, score, whether revived, whether it entered the final blend.
3. Render the correlation trace from `portfolio.jsonl` — this is the evidence the portfolio earned its cost.
4. Report **turns** and **scripts** separately against the 50 cap.
5. Render the per-iteration **code diff** (`difflib.unified_diff` between a node and its parent) — still a required §2.5 deliverable and still missing.

### Tests

- `test_report_renders_every_slot_lane`
- `test_report_states_both_turn_and_script_counts`
- `test_report_renders_a_unified_diff_per_iteration`

---

## 5. Risk register

| Risk | Detection | Mitigation |
|---|---|---|
| Continuing past the convergence point | One `system_best_curve`, asserted by test in Phase 1 | Never give a slot its own stopping power |
| Slots are redundant | Phase 2 correlation gate | Abandon 3–5, keep parallelism, publish the negative result |
| Winner's curse | Phase 0 fold A/B guard | Cap n at 3; +0.0003 at 18 candidates is affordable, +0.0018 at 50 is not |
| Artifact races | Phase 0 isolation, asserted by test | Per-slot dirs, `shared/` read-only |
| Rate limits / key exhaustion | `llm.py` daily-cap failover | Keep LLM calls sequential |
| Wall-clock worse than expected | Phase 1 exit criterion | Expect 1.5–2×, not 3×; still ample against 35 min of a 6 h ceiling |
| Feasibility tier slips | Token count per turn | ~2.5× tokens; tiers are coarse and gated on beating the baseline — very likely the same tier |

## 6. Order of work

```
Phase 0  ──▶  Phase 1  ──▶  Phase 2 ── gate ──┬──▶ Phase 3 ──▶ Phase 4 ──▶ Phase 5 ──▶ Phase 6
 rails       parallel      diversity          │
                           + measurement      └──▶ (correlation > 0.95) stop, keep 0–1, report why
```

Phases 0–2 are the cheap, low-risk half and they answer the question the whole design rests on.
Do not build 3–5 before Phase 2 returns a number.

## 7. References

- **Population Based Training** — Jaderberg et al., [arXiv:1711.09846](https://arxiv.org/abs/1711.09846). A population that periodically *exploits* by copying a better member and *explores* by perturbing it. Closest published match to the archive-and-revive scheme.
- **ASHA** — Li et al., [arXiv:1810.05934](https://arxiv.org/abs/1810.05934). Asynchronous promotion under a fixed budget; the model for slot refill.
- **FML-bench (search dynamics)** — [arXiv:2605.17373](https://arxiv.org/abs/2605.17373). Breadth wins when improvements are sparse; adaptive switching beats fixed strategies. Also reports that a greedy hill-climber nearly matches tree search.
- **Gome** — [arXiv:2603.01692](https://arxiv.org/abs/2603.01692). At frontier model strength, directed updates beat exhaustive tree search. The counter-argument; keep it in view.
- **Iris** — [arXiv:2608.02143](https://arxiv.org/abs/2608.02143). Revisable claims with explicit scope and status; the model for the shared belief set.
- **Arbor** — [arXiv:2606.12563](https://arxiv.org/abs/2606.12563). A critic's contribution is measurement integrity, not score. Already implemented in `agent/critic.py`; it must run per slot.
- **Negative correlation learning** — Liu & Yao, *Neural Networks* 12(10):1399–1404, 1999. Train members to disagree rather than hoping they do. The principled version of Phase 2.
