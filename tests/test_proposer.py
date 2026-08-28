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
    ):
        t()
        print(f"ok  {t.__name__}")
    print("all passed")
