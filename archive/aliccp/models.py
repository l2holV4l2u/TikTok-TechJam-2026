"""Multi-task CTR/CVR model zoo. build(name, cardinalities, **hp) -> nn.Module."""
import torch
import torch.nn as nn


class _Embeddings(nn.Module):
    """Concats one embedding table per sparse feature into a flat vector."""

    def __init__(self, cardinalities: dict[str, int], embed_dim: int):
        super().__init__()
        self.fields = list(cardinalities.keys())
        self.embs = nn.ModuleDict({f: nn.Embedding(cardinalities[f], embed_dim) for f in self.fields})
        self.out_dim = len(self.fields) * embed_dim

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.embs[f](x[f]) for f in self.fields], dim=-1)


def _stack(in_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    """MLP feature extractor: Linear+ReLU per hidden layer, no final activation stripped."""
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers)


def _tower(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    """MLP head: same as _stack plus a final unactivated Linear to out_dim (a logit)."""
    return nn.Sequential(_stack(in_dim, hidden), nn.Linear(hidden[-1] if hidden else in_dim, out_dim))


class SharedBottom(nn.Module):
    def __init__(self, cardinalities, embed_dim=16, bottom_hidden=(128, 64), tower_hidden=(32,)):
        super().__init__()
        self.emb = _Embeddings(cardinalities, embed_dim)
        self.bottom = _stack(self.emb.out_dim, bottom_hidden)
        self.ctr_tower = _tower(bottom_hidden[-1], tower_hidden, 1)
        self.cvr_tower = _tower(bottom_hidden[-1], tower_hidden, 1)

    def forward(self, x):
        h = self.bottom(self.emb(x))
        return self.ctr_tower(h).squeeze(-1), self.cvr_tower(h).squeeze(-1)


class ESMM(nn.Module):
    def __init__(self, cardinalities, embed_dim=16, tower_hidden=(128, 64, 32)):
        super().__init__()
        self.emb = _Embeddings(cardinalities, embed_dim)
        self.ctr_tower = _tower(self.emb.out_dim, tower_hidden, 1)
        self.cvr_tower = _tower(self.emb.out_dim, tower_hidden, 1)

    def forward(self, x):
        z = self.emb(x)
        # CVR tower is only ever supervised via CTCVR = sigmoid(ctr)*sigmoid(cvr) trained over ALL impressions
        # (see train.py loss), not just clicks, which is what fixes AliCCP's CVR sample-selection bias.
        return self.ctr_tower(z).squeeze(-1), self.cvr_tower(z).squeeze(-1)


class MMoE(nn.Module):
    def __init__(self, cardinalities, embed_dim=16, num_experts=4, expert_hidden=(128, 64), tower_hidden=(32,)):
        super().__init__()
        self.emb = _Embeddings(cardinalities, embed_dim)
        in_dim = self.emb.out_dim
        self.experts = nn.ModuleList([_stack(in_dim, expert_hidden) for _ in range(num_experts)])
        self.gate_ctr = nn.Linear(in_dim, num_experts)
        self.gate_cvr = nn.Linear(in_dim, num_experts)
        self.ctr_tower = _tower(expert_hidden[-1], tower_hidden, 1)
        self.cvr_tower = _tower(expert_hidden[-1], tower_hidden, 1)

    def forward(self, x):
        z = self.emb(x)
        expert_outs = torch.stack([e(z) for e in self.experts], dim=1)  # (B, E, H)
        w_ctr = torch.softmax(self.gate_ctr(z), dim=-1).unsqueeze(-1)
        w_cvr = torch.softmax(self.gate_cvr(z), dim=-1).unsqueeze(-1)
        h_ctr = (expert_outs * w_ctr).sum(dim=1)
        h_cvr = (expert_outs * w_cvr).sum(dim=1)
        return self.ctr_tower(h_ctr).squeeze(-1), self.cvr_tower(h_cvr).squeeze(-1)


class PLE(nn.Module):
    """Single-layer CGC (task-specific + shared experts per task gate); full PLE stacks several of these."""

    def __init__(self, cardinalities, embed_dim=16, n_task_experts=2, n_shared_experts=2,
                 expert_hidden=(128, 64), tower_hidden=(32,)):
        super().__init__()
        self.emb = _Embeddings(cardinalities, embed_dim)
        in_dim = self.emb.out_dim
        self.ctr_experts = nn.ModuleList([_stack(in_dim, expert_hidden) for _ in range(n_task_experts)])
        self.cvr_experts = nn.ModuleList([_stack(in_dim, expert_hidden) for _ in range(n_task_experts)])
        self.shared_experts = nn.ModuleList([_stack(in_dim, expert_hidden) for _ in range(n_shared_experts)])
        self.gate_ctr = nn.Linear(in_dim, n_task_experts + n_shared_experts)
        self.gate_cvr = nn.Linear(in_dim, n_task_experts + n_shared_experts)
        self.ctr_tower = _tower(expert_hidden[-1], tower_hidden, 1)
        self.cvr_tower = _tower(expert_hidden[-1], tower_hidden, 1)

    def _gated(self, z, task_experts, gate):
        outs = torch.stack([e(z) for e in task_experts] + [e(z) for e in self.shared_experts], dim=1)
        w = torch.softmax(gate(z), dim=-1).unsqueeze(-1)
        return (outs * w).sum(dim=1)

    def forward(self, x):
        z = self.emb(x)
        h_ctr = self._gated(z, self.ctr_experts, self.gate_ctr)
        h_cvr = self._gated(z, self.cvr_experts, self.gate_cvr)
        return self.ctr_tower(h_ctr).squeeze(-1), self.cvr_tower(h_cvr).squeeze(-1)


_MODELS = {"shared_bottom": SharedBottom, "esmm": ESMM, "mmoe": MMoE, "ple": PLE}


def build(name: str, cardinalities: dict[str, int], **hp) -> nn.Module:
    if name not in _MODELS:
        raise ValueError(f"unknown model {name!r}, choices: {list(_MODELS)}")
    return _MODELS[name](cardinalities, **hp)
