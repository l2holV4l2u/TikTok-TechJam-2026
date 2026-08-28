"""Search tree over solution scripts.

The loop used to restart from a fixed skeleton every iteration, so iteration N+1 could not
build on N. Here each scored script is a node, and a proposal is a targeted edit of a chosen
node. A node that keeps producing children no better than itself is retired, and search falls
back to the next-best node -- that is the backtracking a linear greedy walk cannot do.
"""
from dataclasses import dataclass, field


@dataclass
class Node:
    iter_id: int
    parent_id: int | None
    hypothesis: str
    code: str
    score: float
    misses: int = 0          # children that failed to beat this node
    crashes: int = 0         # children that crashed or timed out before producing a score
    child_seconds: float = 0.0   # wall time this node's children have consumed
    dead: bool = False


@dataclass
class Tree:
    max_misses: int = 3      # retire a node after this many non-improving children
    epsilon: float = 0.002   # organizer convergence epsilon; smaller deltas are noise
    nodes: list[Node] = field(default_factory=list)

    def add(self, node: Node) -> None:
        self.nodes.append(node)

    def get(self, iter_id: int) -> Node | None:
        return next((n for n in self.nodes if n.iter_id == iter_id), None)

    def select(self, mode: str = "refine") -> Node | None:
        """Best live node, or None when the tree is empty or fully retired (draft from scratch).

        FML-bench found that agents exploring broadly beat agents refining one line of attack
        deeply (arXiv:2510.10472), and that switching to broader exploration on detecting
        stagnation outperforms every fixed strategy (arXiv:2605.17373).

        Both modes return the best live node; what differs is what the proposer is asked to do
        with it. Breadth belongs in method space, not in which file you start from. An earlier
        version sent `broaden` back to the earliest node instead, and a run showed the cost
        directly: the agent rebuilt a genuinely useful idea (recency weighting, worth +0.0005 in
        its own internal sweep) on top of the plain baseline rather than on the best model it
        had, so the iteration scored below a leader it should have improved.
        """
        live = [n for n in self.nodes if not n.dead]
        return max(live, key=lambda n: n.score) if live else None

    def record_child(self, parent_id: int | None, score: float, seconds: float = 0.0) -> None:
        """A child that does not clear the parent by epsilon is a miss, not progress."""
        parent = self.get(parent_id) if parent_id is not None else None
        if parent is None:
            return
        parent.child_seconds += seconds
        if score <= parent.score + self.epsilon:
            parent.misses += 1
            if parent.misses >= self.max_misses:
                parent.dead = True

    def record_failure(self, parent_id: int | None, seconds: float = 0.0) -> None:
        """A child that crashed or timed out is evidence against its parent too.

        record_child only ever ran after a score existed, so a crash counted for nothing. In
        r38_1k node #8 produced five children: two of them crashed -- one after 4,373 s at the
        timeout -- and neither moved the node any closer to retirement. Arbor (arXiv:2606.12563)
        scores an action by expected gain over cost, discounted by an explicit crash-rate term;
        this is that term at the scale of a run that gets a handful of iterations.

        A crash is weaker evidence than a miss: the idea may be sound and the code wrong, and
        recovery already retries it. So it counts half.

        INTEGRATION: loop.py must call this on the failure path, beside the existing
        `tree.record_child(parent_id, score)` on the success path:
            tree.record_failure(parent_id, res.seconds)
        """
        parent = self.get(parent_id) if parent_id is not None else None
        if parent is None:
            return
        parent.crashes += 1
        parent.child_seconds += seconds
        if parent.crashes >= 2 * self.max_misses:
            parent.dead = True

    @property
    def best(self) -> Node | None:
        return max(self.nodes, key=lambda n: n.score) if self.nodes else None

    def lineage(self, node: Node) -> list[Node]:
        out, cur = [node], node
        while cur.parent_id is not None:
            cur = self.get(cur.parent_id)
            if cur is None:
                break
            out.append(cur)
        return list(reversed(out))

    def render(self) -> str:
        """ASCII tree for the run report; judges read the search shape, not just the scores."""
        kids: dict[int | None, list[Node]] = {}
        for n in self.nodes:
            kids.setdefault(n.parent_id, []).append(n)
        lines: list[str] = []

        def walk(pid: int | None, depth: int) -> None:
            for n in sorted(kids.get(pid, []), key=lambda x: x.iter_id):
                mark = " [retired]" if n.dead else ""
                lines.append(f"{'  ' * depth}#{n.iter_id} {n.score:.4f}{mark}  {n.hypothesis[:70]}")
                walk(n.iter_id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)


def demo() -> None:
    t = Tree(max_misses=2, epsilon=0.002)
    t.add(Node(0, None, "root", "", 0.600))
    assert t.select().iter_id == 0

    assert t.select("broaden") is t.select("refine"), "one node: both modes agree"

    t.add(Node(1, 0, "child a", "", 0.601))   # +0.001, inside epsilon -> a miss
    t.record_child(0, 0.601)
    assert t.get(0).misses == 1 and not t.get(0).dead
    assert t.select().iter_id == 1, "1 scores higher, so it is the next parent"

    t.add(Node(2, 0, "child b", "", 0.599))
    t.record_child(0, 0.599)
    assert t.get(0).dead, "two misses retires the root"
    assert t.select().iter_id == 1

    t.add(Node(3, 1, "child c", "", 0.610))   # clears epsilon -> not a miss
    t.record_child(1, 0.610)
    assert t.get(1).misses == 0
    assert t.select().iter_id == 3
    # both modes anchor on the leader; breadth is expressed in what the proposer is asked to
    # do with it, not by discarding the best structure found so far
    assert t.select("refine").iter_id == 3
    assert t.select("broaden").iter_id == 3
    assert [n.iter_id for n in t.lineage(t.get(3))] == [0, 1, 3]

    for i, sc in ((4, 0.6), (5, 0.6), (6, 0.6)):
        t.add(Node(i, 3, "d", "", sc))
        t.record_child(3, sc)
    assert t.get(3).dead
    # every node retired except the losers, which are still live and selectable
    assert t.select() is not None
    for n in t.nodes:
        n.dead = True
    assert t.select() is None, "fully retired tree must fall back to a fresh draft"
    print(t.render())
    print("ok")


if __name__ == "__main__":
    demo()
