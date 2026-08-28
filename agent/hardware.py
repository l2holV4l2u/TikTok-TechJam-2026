"""Runtime hardware discovery shared by the CLI and proposer prompt."""
from __future__ import annotations


def resolve_device(requested: str = "auto") -> dict:
    """Resolve ``auto|cpu|cuda`` and return serialisable facts for prompts/reports."""
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported device {requested!r}; choose auto, cpu, or cuda")

    try:
        import torch
    except ImportError as exc:
        if requested == "cuda":
            raise RuntimeError("--device cuda requires a CUDA-enabled PyTorch installation") from exc
        return {
            "requested": requested, "device": "cpu", "torch_version": None,
            "cuda_runtime": None, "gpu_name": None, "gpu_memory_gb": None,
            "cuda_available": False,
        }

    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError(
            "--device cuda was requested, but this Python has no usable CUDA runtime. "
            "Run the harness with the CUDA-enabled virtual environment."
        )
    device = "cuda" if available and requested != "cpu" else "cpu"
    gpu_name = None
    gpu_memory_gb = None
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_memory_gb = round(props.total_memory / (1024 ** 3), 2)

    return {
        "requested": requested, "device": device, "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb, "cuda_available": available,
    }


def prompt_hardware_note(hardware: dict) -> str:
    """Describe available compute without prescribing a modelling conclusion."""
    if hardware["device"] == "cuda":
        return (
            f"- CUDA is selected: {hardware['gpu_name']} with "
            f"{hardware['gpu_memory_gb']:.2f} GiB, PyTorch {hardware['torch_version']} "
            f"(CUDA runtime {hardware['cuda_runtime']}). AGENT_DEVICE=cuda.\n"
            "  Use batched transfers and mixed precision where numerically safe. NumPy metrics "
            "and feature preparation remain CPU work. LightGBM remains CPU unless GPU support "
            "is explicitly verified."
        )
    return (
        f"- This run selects CPU (PyTorch {hardware.get('torch_version') or 'not installed'}). "
        "AGENT_DEVICE=cpu; CUDA and AMP must not be used."
    )
