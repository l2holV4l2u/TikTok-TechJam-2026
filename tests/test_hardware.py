"""Offline tests for device selection and prompt disclosure."""
from unittest.mock import patch

from agent.hardware import prompt_hardware_note, resolve_device


def test_cpu_can_be_forced_even_when_cuda_exists():
    with patch("torch.cuda.is_available", return_value=True):
        got = resolve_device("cpu")
    assert got["device"] == "cpu"


def test_cuda_request_fails_loudly_in_a_cpu_build():
    with patch("torch.cuda.is_available", return_value=False):
        try:
            resolve_device("cuda")
        except RuntimeError as exc:
            assert "CUDA-enabled virtual environment" in str(exc)
        else:
            raise AssertionError("a CPU-only torch build must not silently accept --device cuda")


def test_prompt_note_distinguishes_cpu_and_gpu():
    cpu = prompt_hardware_note({"device": "cpu", "torch_version": "2.x"})
    gpu = prompt_hardware_note({
        "device": "cuda", "gpu_name": "test GPU", "gpu_memory_gb": 6.0,
        "torch_version": "2.x+cu", "cuda_runtime": "13.0",
    })
    assert "AGENT_DEVICE=cpu" in cpu and "must not be used" in cpu
    assert "AGENT_DEVICE=cuda" in gpu and "6.00 GiB" in gpu


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} tests passed")
