"""Plain-assert tests for pipeline.models, tiny random tensors only. Run with: python -m tests.test_models"""
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from pipeline.models import FM, build

CARD = {"user_id": 11, "video_id": 7, "tab": 4}
MODEL_NAMES = ["fm", "deepfm", "dcnv2", "din"]


def _batch(n: int, device="cpu"):
    return {f: torch.randint(0, c, (n,), device=device) for f, c in CARD.items()}


def test_models_build_and_forward_finite():
    for name in MODEL_NAMES:
        model = build(name, CARD, embed_dim=4)
        x = _batch(9)
        out = model(x)
        assert out.shape == (9,), (name, out.shape)
        assert torch.isfinite(out).all(), (name, out)


def test_gradients_flow_to_all_embeddings():
    for name in MODEL_NAMES:
        model = build(name, CARD, embed_dim=4)
        x = _batch(9)
        model(x).sum().backward()
        for f in CARD:
            table = model.emb.embs[f] if hasattr(model, "emb") else model.fm.emb.embs[f]
            assert table.weight.grad is not None, (name, f)
            assert torch.isfinite(table.weight.grad).all(), (name, f)
            assert table.weight.grad.abs().sum() > 0, (name, f, "zero gradient")


def _naive_fm_logit(model: "FM", x: dict) -> "torch.Tensor":
    """O(k n^2) double-loop reference for the FM pairwise term, same batch, one row at a time."""
    n = len(next(iter(x.values())))
    fields = model.fields
    out = torch.empty(n)
    for row in range(n):
        xi = {f: x[f][row:row + 1] for f in fields}
        e = {f: model.emb.embs[f](xi[f]).squeeze(0) for f in fields}
        linear = sum(model.linear[f](xi[f]).squeeze(0) for f in fields).squeeze(0)
        interaction = torch.zeros(())
        for a in range(len(fields)):
            for b in range(a + 1, len(fields)):
                interaction = interaction + torch.dot(e[fields[a]], e[fields[b]])
        out[row] = (model.bias.squeeze(0) + linear + interaction).item()
    return out


def test_fm_pairwise_term_matches_naive_double_loop():
    model = build("fm", CARD, embed_dim=3)
    model.eval()
    x = _batch(5)
    with torch.no_grad():
        fast = model(x)
        naive = _naive_fm_logit(model, x)
    assert torch.allclose(fast, naive, atol=1e-5), (fast, naive)


def test_cpu_cuda_consistent_shapes():
    if not torch.cuda.is_available():
        return "skip (no CUDA device)"
    for name in MODEL_NAMES:
        model_cpu = build(name, CARD, embed_dim=4)
        x_cpu = _batch(6, device="cpu")
        out_cpu = model_cpu(x_cpu)

        model_gpu = build(name, CARD, embed_dim=4).to("cuda")
        x_gpu = _batch(6, device="cuda")
        out_gpu = model_gpu(x_gpu)

        assert out_cpu.shape == out_gpu.shape, (name, out_cpu.shape, out_gpu.shape)
        assert out_gpu.device.type == "cuda", name
        assert torch.isfinite(out_gpu).all(), name


def test_initial_logits_are_small_enough_to_train():
    """Default torch N(0,1) embedding init makes the FM interaction term explode at step zero.

    Measured on Pure: default init plateaus at 0.5533 valid after 40 epochs; small init reaches
    0.6020 in under 15. The symptom is visible before any training -- initial logits are huge --
    so assert on that rather than on a slow convergence run.
    """
    cards = {f: 50 for f in ("user_id", "video_id", "author_id", "tab", "duration_bucket")}
    x = {f: torch.randint(0, 50, (256,)) for f in cards}
    for name in ("fm", "deepfm", "dcnv2", "din"):
        torch.manual_seed(0)
        out = build(name, cards)(x)
        assert out.abs().max().item() < 5.0, (name, out.abs().max().item())


if __name__ == "__main__":
    if torch is None:
        print("torch not installed -- skipping pipeline.models tests")
    else:
        for t in (
            test_models_build_and_forward_finite,
            test_gradients_flow_to_all_embeddings,
            test_fm_pairwise_term_matches_naive_double_loop,
            test_cpu_cuda_consistent_shapes,
            test_initial_logits_are_small_enough_to_train,
        ):
            note = t()
            print(f"ok  {t.__name__}" + (f" {note}" if note else ""))
        print("all passed")
