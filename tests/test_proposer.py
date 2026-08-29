"""Offline tests: python -m tests.test_proposer"""
from agent.ledger import Entry
from agent.proposer import TASK_BRIEF, LLMProposer
from agent.tree import Node

OK_CODE = 'print("METRICS {\\"primary\\": 0.5}")'


def _reply(hyp, code=OK_CODE, ti=10, to=5):
    return (f"HYPOTHESIS: {hyp}\n```python\n{code}\n```", ti, to)


def _entry(i, hyp="h", metrics=None, status="ok"):
    return Entry(i, None, 0, hyp, "code", metrics or {"primary": 0.6}, 1.0, 10, 5, status)


def _capture(reply=None):
    """Returns (proposer, prompts list)."""
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        return reply if reply is not None else _reply("h")

    return LLMProposer(complete), prompts


def test_brief_carries_no_human_findings():
    """The guard on the whole design: the brief is task spec, not research.

    Everything here was measured by a human in an earlier version of this project and handed
    to the model. If any of it comes back, the agent stops being the one doing the research and
    the Innovation and Autonomy claims become false.
    """
    banned = ["blend", "ensembl", "rank aggregation", "decorrelat", "lambdarank", "deepfm",
              "already measured", "exhibit", "what to propose", "headroom might",
              "0.6045", "0.6021", "0.5935", "target encoding", "skeleton"]
    low = TASK_BRIEF.lower()
    leaked = [w for w in banned if w in low]
    assert not leaked, f"human findings leaked back into the brief: {leaked}"


def test_knowledge_base_carries_no_measured_results():
    """The brief is not the only channel into the prompt, and we learned that the hard way.

    The KB's expected_effect fields are injected on every improve iteration. They once held
    our own measurements -- including the exact winning blend and its weights -- so the agent
    was one retrieval hit away from being handed the answer while we claimed it had no priors.
    A paper entry may describe what its method claims; it may not report what we measured here.
    """
    import re
    from agent.kb import load_papers
    pat = re.compile(r"measured on this task|measured here|scored valid|already measured"
                     r"|0\.[56]\d{3}", re.I)
    leaked = [p["id"] for p in load_papers()
              if pat.search(p.get("expected_effect", "") + p.get("applies_to", ""))]
    assert not leaked, f"measured-on-this-task results present in KB entries: {leaked}"


def test_eda_phase_asks_for_inspection_and_no_metrics():
    p, prompts = _capture()
    p.propose(phase="eda")
    assert "INSPECT DATA" in prompts[0]
    assert "do NOT print a METRICS line" in prompts[0]
    assert "EXPERIMENTS THIS RUN" not in prompts[0], "there is no history to report yet"


def test_baseline_phase_names_the_official_number():
    p, prompts = _capture()
    p.propose(phase="baseline", context={"eda": "positives=0.33 fields=37"})
    assert "REPRODUCE THE OFFICIAL BASELINE" in prompts[0]
    assert "0.6016" in prompts[0]
    assert "positives=0.33" in prompts[0], "EDA findings must reach the baseline prompt"


def test_improve_phase_sends_the_parent_script_for_editing():
    p, prompts = _capture()
    parent = Node(4, None, "fm with author features", "print('parent script here')", 0.6042)
    p.propose(phase="improve", parent=parent, history=[_entry(4)])
    assert "print('parent script here')" in prompts[0]
    assert "iteration #4" in prompts[0] and "0.6042" in prompts[0]
    assert "Make ONE targeted change" in prompts[0]


def test_broaden_mode_demands_a_new_direction_and_lists_what_was_tried():
    """FML-bench: on stagnation, breadth beats refining the same line of attack again."""
    p, prompts = _capture()
    parent = Node(1, None, "baseline fm", "print('working plumbing')", 0.6019)
    hist = [_entry(2, hyp="expand to all 37 fields"), _entry(3, hyp="add tag feature")]
    p.propose(phase="improve", parent=parent, history=hist,
              context={"mode": "broaden", "stale": 2})
    t = prompts[0]
    assert "Change DIRECTION, not detail" in t
    assert "expand to all 37 fields" in t and "add tag feature" in t
    assert "print('working plumbing')" in t, "plumbing is reused even when direction changes"
    assert "Make ONE targeted change" not in t, "broaden must not ask for a targeted edit"


def test_refine_mode_asks_for_a_targeted_edit():
    p, prompts = _capture()
    parent = Node(4, None, "fm variant", "print('parent')", 0.6042)
    p.propose(phase="improve", parent=parent, history=[_entry(4)], context={"mode": "refine"})
    assert "Make ONE targeted change" in prompts[0]
    assert "Change DIRECTION" not in prompts[0]


def test_iterate_prompt_states_the_candidates_contract():
    p, prompts = _capture()
    p.propose(phase="improve")
    assert "CANDIDATES" in prompts[0]
    assert "not limited to a single alternative" in prompts[0]


def test_improve_without_a_parent_drafts_from_scratch():
    p, prompts = _capture()
    p.propose(phase="improve", parent=None)
    assert "from scratch" in prompts[0]
    assert "Make ONE targeted change" not in prompts[0]


def test_beliefs_and_memory_reach_the_improve_prompt():
    p, prompts = _capture()
    p.propose(phase="improve",
              context={"knowledge": "- (active) pointwise beats pairwise here\n"
                                    "- (invalidated) capacity is the bottleneck",
                       "memory": "PRIOR RUNS OF THIS AGENT (3 scored experiments)"})
    assert "pointwise beats pairwise here" in prompts[0]
    assert "capacity is the bottleneck" in prompts[0]
    assert "invalidated" in prompts[0], "a contradicted belief must be shown as contradicted"
    assert "PRIOR RUNS OF THIS AGENT" in prompts[0]


def test_controller_diagnosis_and_incumbent_predictions_reach_prompt():
    p, prompts = _capture()
    p.propose(phase="improve", context={
        "diagnosis": "16-40 users gap 0.42 share 63%",
        "incumbent_ready": True,
    })
    assert "WHERE THE TRUSTED INCUMBENT LOSES" in prompts[0]
    assert "16-40 users gap" in prompts[0]
    assert "incumbent_valid_scores.npy" in prompts[0]


def test_empty_belief_set_is_not_rendered():
    p, prompts = _capture()
    p.propose(phase="improve", context={"knowledge": "nothing established yet"})
    assert "WHAT YOU CURRENTLY BELIEVE" not in prompts[0]


def test_iterate_prompt_states_the_findings_contract():
    p, prompts = _capture()
    p.propose(phase="improve")
    assert "FINDINGS " in prompts[0]
    assert "not a wasted iteration" in prompts[0]


def test_prompt_size_bounded_as_history_grows():
    lengths = {}

    def make(key):
        def complete(prompt):
            lengths[key] = len(prompt)
            return _reply("h")
        return complete

    LLMProposer(make("small")).propose(phase="improve",
                                       history=[_entry(i) for i in range(5)])
    LLMProposer(make("large")).propose(phase="improve",
                                       history=[_entry(i) for i in range(500)])
    # history is summarised to a fixed window; token cost must not compound over a long run
    assert lengths["large"] - lengths["small"] < 400, lengths


def test_parent_code_is_truncated_not_unbounded():
    p, prompts = _capture()
    huge = Node(1, None, "h", "x" * 100_000, 0.6)
    p.propose(phase="improve", parent=huge)
    assert len(prompts[0]) < 40_000, len(prompts[0])


def test_blacklisted_hypothesis_never_returned():
    p, prompts = _capture(reply=_reply("banned"))
    assert p.propose(phase="improve", blacklist={"banned"}) is None
    assert len(prompts) == 2, "one retry, then give up"


def test_blacklist_does_not_block_baseline_reproduction():
    """Reproducing the baseline is a requirement, not an idea to retire."""
    p, _ = _capture(reply=_reply("reproduce official FM"))
    got = p.propose(phase="baseline", blacklist={"reproduce official FM"})
    assert got is not None


def test_malformed_output_returns_none_after_two_attempts():
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        return ("this is not the expected format at all", 5, 5)

    assert LLMProposer(complete).propose(phase="improve") is None
    assert len(prompts) == 2
    assert "did not parse" in prompts[1]


def test_kb_entries_appear_but_are_not_pinned():
    p, prompts = _capture()
    p.propose(phase="improve", history=[_entry(0, hyp="sample selection bias multi-task cvr")])
    assert "ESMM" in prompts[0] or "Entire Space" in prompts[0]
    assert "a retrieval hit is not a recommendation" in prompts[0]


def test_full_catalogue_is_offered_so_retrieval_cannot_hide_a_method():
    """v1 surfaced the same 3 papers every iteration, so whole families were never seen."""
    p, prompts = _capture()
    p.propose(phase="improve", history=[_entry(0, hyp="factorization machine embedding fields")])
    for pid in ("rank_aggregation", "bagging", "stacked_generalization", "din", "sasrec"):
        assert pid in prompts[0], f"{pid} missing from the catalogue"


def test_detailed_retrieval_rotates_instead_of_repeating():
    p, prompts = _capture()
    hist = [_entry(0, hyp="factorization machine embedding fields ndcg")]
    for _ in range(3):
        p.propose(phase="improve", history=hist)
    detail = [t.split("DETAIL ON A FEW OF THEM")[1].split("Respond EXACTLY")[0]
              for t in prompts if "DETAIL ON A FEW OF THEM" in t]
    assert len(detail) >= 3
    import re
    shown = [set(re.findall(r"- (\w+):", d)) for d in detail]
    assert not (shown[0] & shown[1]), f"same papers twice: {shown[0]} {shown[1]}"
    assert not ((shown[0] | shown[1]) & shown[2]), shown


def test_token_counts_summed_across_retries():
    replies = iter([("garbage", 20, 3), _reply("h2", ti=15, to=4)])
    p = LLMProposer(lambda _: next(replies))
    r = p.propose(phase="improve")
    assert (r.tokens_in, r.tokens_out) == (35, 7)


def test_history_reports_the_score_outcome_not_the_loop_status():
    """The loop marks any sub-epsilon gain "kept"; the score is still kept and submittable.

    r59 set its record on three consecutive iterations the loop labelled that way, and the
    prompt printed the label verbatim -- telling the agent its best work had been discarded.
    """
    from agent.proposer import _summarize_history

    rows = [
        Entry(1, None, 0, "baseline", "", {"primary": 0.6016}, 1, 0, 0, "ok", None, "baseline"),
        Entry(2, None, 0, "deepfm", "", {"primary": 0.6049}, 1, 0, 0, "ok", None, "improve"),
        Entry(3, None, 0, "record set on a sub-epsilon row", "", {"primary": 0.6056}, 1, 0, 0, "kept",
              None, "improve"),
        Entry(4, None, 0, "a genuinely worse idea", "", {"primary": 0.6002}, 1, 0, 0, "kept",
              None, "improve"),
        Entry(5, None, 0, "a crash", "", {}, 1, 0, 0, "failed", "boom", "improve"),
    ]
    out = _summarize_history(rows, 10)
    record = next(l for l in out.splitlines() if "record set on a sub-epsilon row" in l)
    assert "BEST SO FAR" in record, record
    assert "kept" not in record, ("the highest score in the run must not be reported to the "
                                  "agent by the label the loop used to skip it: " + record)
    worse = next(l for l in out.splitlines() if "genuinely worse" in l)
    assert "below best 0.6056" in worse, worse
    assert "failed" in next(l for l in out.splitlines() if "a crash" in l)


def test_the_static_head_of_the_prompt_is_identical_across_calls():
    """Providers discount input that repeats as a stable PREFIX, and only as a prefix.

    The brief and the method catalogue are byte-identical on every call of a run, but while
    the catalogue sat after the volatile blocks none of it could be cached. Keeping the head
    stable is worth thousands of input tokens a run and removes nothing from the prompt.
    """
    proposer, prompts = _capture()
    proposer.propose(phase="improve", history=[_entry(1)], context={"eda": "first eda"})
    proposer.propose(phase="improve", history=[_entry(1), _entry(2, "other")],
                     context={"eda": "a completely different eda report"})
    assert len(prompts) == 2
    a, b = prompts
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    assert common > 8000, (
        f"only {common} characters of stable prefix; the static head must stay contiguous "
        "at the front or none of it can be cached")
    assert "AVAILABLE LITERATURE" in a[:common], (
        "the catalogue is static and belongs inside the cacheable prefix")


def test_the_first_improve_iteration_asks_for_breadth():
    """Measured on this data: distinct families differ by 0.0019 primary, two seeds of one
    family by 0.0002. The family is worth more than the tuning, and one script is one
    iteration however many models it holds -- so the first experiment sweeps families and
    later ones refine the winner.
    """
    proposer, prompts = _capture()
    parent = Node(1, None, "baseline", "print(1)", 0.6016)
    proposer.propose(phase="improve", history=[_entry(1)], parent=parent,
                     context={"mode": "sweep"})
    assert "FIRST experiment" in prompts[0] and "breadth" in prompts[0], prompts[0][:200]
    assert "CANDIDATES" in prompts[0], "the comparison has to be recorded"

    proposer2, p2 = _capture()
    proposer2.propose(phase="improve", history=[_entry(1)], parent=parent,
                      context={"mode": "refine"})
    assert "FIRST experiment" not in p2[0], "later iterations refine, they do not re-sweep"


def test_the_parent_script_sits_inside_the_cacheable_prefix():
    """The parent script is the largest block and repeats whenever the search stays on one
    node, so it belongs in front of the blocks rewritten every iteration. Measured on a
    same-parent pair: 20,012 shared characters before the reorder, 25,830 after -- about
    1,600 tokens a call that no longer have to be re-sent uncached.
    """
    code = "# parent script" + chr(10) + ("x = 1" + chr(10)) * 900
    parent = Node(3, None, "the parent hypothesis", code, 0.6045)

    def build(hist, know):
        proposer, prompts = _capture()
        proposer.propose(phase="improve", history=hist, parent=parent,
                         context={"mode": "refine", "eda": "eda " * 400,
                                  "memory": "PRIOR RUNS " * 300, "knowledge": know,
                                  "diagnosis": "gap table " * 40})
        return prompts[0]

    a = build([_entry(1, "first")], "claim one")
    b = build([_entry(1, "first"), _entry(2, "second"), _entry(3, "third")],
              "claim one" + chr(10) + "claim two")
    shared = 0
    for x, y in zip(a, b):
        if x != y:
            break
        shared += 1
    starts_at = a.index("# parent script")
    assert shared > starts_at + len(code) - 50, (
        f"parent code spans {starts_at}..{starts_at + len(code)} but only {shared} chars are "
        "shared; a volatile block was moved in front of it and the cache prefix broke")


if __name__ == "__main__":
    for t in (
        test_brief_carries_no_human_findings,
        test_knowledge_base_carries_no_measured_results,
        test_eda_phase_asks_for_inspection_and_no_metrics,
        test_baseline_phase_names_the_official_number,
        test_improve_phase_sends_the_parent_script_for_editing,
        test_broaden_mode_demands_a_new_direction_and_lists_what_was_tried,
        test_refine_mode_asks_for_a_targeted_edit,
        test_iterate_prompt_states_the_candidates_contract,
        test_improve_without_a_parent_drafts_from_scratch,
        test_beliefs_and_memory_reach_the_improve_prompt,
        test_controller_diagnosis_and_incumbent_predictions_reach_prompt,
        test_empty_belief_set_is_not_rendered,
        test_iterate_prompt_states_the_findings_contract,
        test_prompt_size_bounded_as_history_grows,
        test_parent_code_is_truncated_not_unbounded,
        test_blacklisted_hypothesis_never_returned,
        test_blacklist_does_not_block_baseline_reproduction,
        test_malformed_output_returns_none_after_two_attempts,
        test_kb_entries_appear_but_are_not_pinned,
        test_full_catalogue_is_offered_so_retrieval_cannot_hide_a_method,
        test_detailed_retrieval_rotates_instead_of_repeating,
        test_token_counts_summed_across_retries,
        test_history_reports_the_score_outcome_not_the_loop_status,
        test_the_static_head_of_the_prompt_is_identical_across_calls,
        test_the_first_improve_iteration_asks_for_breadth,
        test_the_parent_script_sits_inside_the_cacheable_prefix,
    ):
        t()
        print(f"ok  {t.__name__}")
    print("all passed")
