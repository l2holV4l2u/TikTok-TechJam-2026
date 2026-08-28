"""Reference model implementations for the human-side baseline reproduction.

Deliberately NOT referenced from the agent's task brief: naming four architectures would be a
prior on method space, and the agent's best run independently proposed DCN-V2 crosses without
one. 0 of 333 agent scripts import this module; every iteration writes its own model.
"""
"""CTR ranking model zoo. build(name, cardinalities, **hp) -> nn.Module, forward(x: dict[str,LongTensor]) -> logit (B,)."""
import torch
import torch.nn as nn


class _Embeddings(nn.Module):
    """One embedding table per sparse field, gathered into a (B, F, D) stack."""

    def __init__(self, cardinalities: dict[str, int], embed_dim: int):
        super().__init__()
        self.fields = list(cardinalities.keys())
        self.embs = nn.ModuleDict({f: nn.Embedding(cardinalities[f], embed_dim) for f in self.fields})
        # torch defaults embeddings to N(0,1). Summed over fields the FM interaction term then
        # starts orders of magnitude too large: measured on Pure, the default init plateaus at
        # 0.5533 valid after 40 epochs where std=0.01 reaches 0.6020 in under 15.
        for emb in self.embs.values():
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([self.embs[f](x[f]) for f in self.fields], dim=1)  # (B, F, D)


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class FM(nn.Module):
    """2nd-order factorization machine: linear term + pairwise term via the standard O(k) sum-square trick."""

    def __init__(self, cardinalities: dict[str, int], embed_dim: int = 16):
        super().__init__()
        self.fields = list(cardinalities.keys())
        self.linear = nn.ModuleDict({f: nn.Embedding(cardinalities[f], 1) for f in self.fields})
        for w in self.linear.values():
            nn.init.zeros_(w.weight)  # FM first-order weights start at zero, as in libFM
        self.emb = _Embeddings(cardinalities, embed_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        linear = torch.cat([self.linear[f](x[f]) for f in self.fields], dim=-1).sum(-1)
        e = self.emb(x)  # (B, F, D)
        sum_sq = e.sum(dim=1).pow(2)
        sq_sum = e.pow(2).sum(dim=1)
        interaction = 0.5 * (sum_sq - sq_sum).sum(dim=-1)
        return self.bias + linear + interaction


class DeepFM(nn.Module):
    """FM term (wide, memorizes cross features) plus an MLP over the flattened embeddings (deep)."""

    def __init__(self, cardinalities: dict[str, int], embed_dim: int = 16, hidden: tuple[int, ...] = (128, 64)):
        super().__init__()
        self.fm = FM(cardinalities, embed_dim)
        n_fields = len(cardinalities)
        self.mlp = _mlp(n_fields * embed_dim, hidden, 1)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        fm_logit = self.fm(x)
        e = self.fm.emb(x).flatten(start_dim=1)
        deep_logit = self.mlp(e).squeeze(-1)
        return fm_logit + deep_logit


class _CrossNetV2(nn.Module):
    """DCN-v2 cross layers: x_{l+1} = x_0 * (W_l x_l + b_l) + x_l, low-rank-free full-matrix form."""

    def __init__(self, in_dim: int, n_layers: int = 3):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(in_dim, in_dim)) for _ in range(n_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(in_dim)) for _ in range(n_layers)])
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for w, b in zip(self.weights, self.biases):
            x = x0 * (x @ w.T + b) + x
        return x


class DCNv2(nn.Module):
    """Cross network (explicit bounded-degree feature crosses) run in parallel with a deep MLP, then concat."""

    def __init__(self, cardinalities: dict[str, int], embed_dim: int = 16, cross_layers: int = 3, hidden: tuple[int, ...] = (128, 64)):
        super().__init__()
        self.emb = _Embeddings(cardinalities, embed_dim)
        in_dim = len(cardinalities) * embed_dim
        self.cross = _CrossNetV2(in_dim, cross_layers)
        self.deep = _mlp(in_dim, hidden, hidden[-1] if hidden else in_dim)
        self.head = nn.Linear(in_dim + (hidden[-1] if hidden else in_dim), 1)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        e = self.emb(x).flatten(start_dim=1)
        cross_out = self.cross(e)
        deep_out = self.deep(e)
        return self.head(torch.cat([cross_out, deep_out], dim=-1)).squeeze(-1)


class DIN(nn.Module):
    """Simplified Deep Interest Network: video_id embedding attends over the other fields as a pseudo-history."""

    def __init__(self, cardinalities: dict[str, int], embed_dim: int = 16, hidden: tuple[int, ...] = (128, 64)):
        super().__init__()
        self.fields = list(cardinalities.keys())
        self.target_field = "video_id" if "video_id" in cardinalities else self.fields[0]
        self.emb = _Embeddings(cardinalities, embed_dim)
        self.attn = _mlp(embed_dim * 4, (64,), 1)  # [target, other, target-other, target*other]
        self.mlp = _mlp(len(self.fields) * embed_dim, hidden, 1)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        e = self.emb(x)  # (B, F, D)
        t_idx = self.fields.index(self.target_field)
        target = e[:, t_idx:t_idx + 1, :].expand_as(e)  # (B, F, D), broadcast target to every field slot
        attn_in = torch.cat([target, e, target - e, target * e], dim=-1)
        weights = torch.softmax(self.attn(attn_in).squeeze(-1), dim=-1).unsqueeze(-1)  # (B, F, 1)
        weighted = (e * weights).flatten(start_dim=1)
        return self.mlp(weighted).squeeze(-1)


_MODELS = {"fm": FM, "deepfm": DeepFM, "dcnv2": DCNv2, "din": DIN}


def build(name: str, cardinalities: dict[str, int], **hp) -> nn.Module:
    if name not in _MODELS:
        raise ValueError(f"unknown model {name!r}, choices: {list(_MODELS)}")
    return _MODELS[name](cardinalities, **hp)
