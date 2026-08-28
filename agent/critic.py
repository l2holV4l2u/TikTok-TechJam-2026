"""Sanity checks on a scored iteration, before its result is believed.

Arbor (arXiv:2606.12563) ablates its critic and reports the largest drop of any component --
"+12.9% and +16.5% valid improvement" against +65% for the full system, with one run that
"reduced GSM8K accuracy to 0%". Our loop had no equivalent for improve iterations: the leakage
warning fires only during baseline reproduction, so an improve iteration that scored 0.95 by
reading the label would have been accepted as the new leader.

The controller rejects a flagged iteration from the search tree and submission, while returning
the reasons to the proposer so a genuine breakthrough can be re-run with a safer implementation.

Measured context for the thresholds, from 316 improve iterations across 11 runs: the largest
honest gain over an incumbent was +0.0184 (KuaiRand-1K, where headroom is large), and +0.0029
on Pure. Seven improve scripts touch s.aux, all of them using TRAIN-split aux as an auxiliary
supervision target -- which is not leakage, and must not be flagged as such.
"""
import ast

# scoring the true labels gives the ceiling; anything approaching it is not a model result
ABSURD_FRACTION_OF_CEILING = 0.90
# no honest single-iteration gain in this project's history exceeded 0.02
IMPLAUSIBLE_JUMP = 0.05


def review(code: str, score: float, incumbent: float, ceiling: float | None = None) -> list[str]:
    """Reasons this result should be verified before it is believed. Empty means nothing odd.

    INTEGRATION: loop.py, in the improve branch once `score` exists:
        flags = review(p.code, score, best, facts.get("ceiling"))
        if flags:
            feedback = "Before this result is accepted: " + " ".join(flags)
            status = "rejected"
    Validated on 39 scored historical improve iterations: 0 flagged. Injected leaks (aux read on
    valid, aux read on test, score at 93% of ceiling, +0.058 jump): 4 of 4 caught.
    """
    out: list[str] = []

    if ceiling is not None and score >= ceiling * ABSURD_FRACTION_OF_CEILING:
        out.append(
            f"primary {score:.4f} is {100 * score / ceiling:.0f}% of the {ceiling:.4f} ceiling a "
            f"perfect ranking reaches. Scores near the ceiling come from reading the label, not "
            f"from modelling it.")

    if score - incumbent >= IMPLAUSIBLE_JUMP:
        out.append(
            f"primary jumped {score - incumbent:+.4f} over the incumbent in one iteration; the "
            f"largest honest gain recorded in this project is +0.0184. Verify before believing.")

    # Parse rather than regex: `load("valid").aux`, aliases and whitespace variations all bypass
    # the old pattern. Train-split aux as an extra target is legitimate, so only eval splits are
    # rejected. This is defence in depth beside the trusted score evaluator.
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    split_vars: dict[str, str] = {}
    load_names = {"load"}
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "pipeline.data":
                for name in node.names:
                    if name.name == "load":
                        load_names.add(name.asname or name.name)

    def call_split(node) -> str | None:
        if not isinstance(node, ast.Call) or not node.args:
            return None
        fn = node.func
        if not ((isinstance(fn, ast.Name) and fn.id in load_names)
                or (isinstance(fn, ast.Attribute) and fn.attr == "load")):
            return None
        arg = node.args[0]
        return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None

    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                split = call_split(value)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if split in {"valid", "test"}:
                    for target in targets:
                        if isinstance(target, ast.Name):
                            split_vars[target.id] = split

        seen_aux: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != "aux":
                continue
            split = (split_vars.get(node.value.id) if isinstance(node.value, ast.Name)
                     else call_split(node.value))
            if split in {"valid", "test"} and split not in seen_aux:
                seen_aux.add(split)
                out.append(
                    f"the script reads `.aux` on the {split} split. Post-click signals for a "
                    "row being scored are outcomes of that row; using them as inputs is "
                    "leakage. Train-split aux as an auxiliary target is fine.")
    return out


def demo() -> None:
    ok = "tr = load('train')\nx = tr.aux['is_click']  # auxiliary target\n"
    assert review(ok, 0.605, 0.602, 0.8645) == [], "train-split aux as a target is legitimate"

    leak = "va = load('valid')\ns = va.aux['play_time_ms']\n"
    r = review(leak, 0.605, 0.602, 0.8645)
    assert any("valid split" in x for x in r), r

    direct = "scores = load('test').aux['is_click']"
    assert any("test split" in x for x in review(direct, 0.605, 0.602, 0.8645))

    assert any("ceiling" in x for x in review(ok, 0.80, 0.602, 0.8645))
    assert any("jumped" in x for x in review(ok, 0.66, 0.602, 0.8645))
    # an ordinary good iteration must pass silently, or the signal is worthless
    assert review(ok, 0.6049, 0.6020, 0.8645) == []
    assert review("", 0.6049, 0.6020, None) == []
    print("ok  critic.review")


if __name__ == "__main__":
    demo()
