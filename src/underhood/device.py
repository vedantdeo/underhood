"""Pick the compute device and prove it works.

uv run underhood-check
"""

from __future__ import annotations

import time

import torch


def pick_device() -> torch.device:
    """MPS on Apple silicon when available, else CPU. CUDA is for the rented GPU in Week 5."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _time_matmul(device: torch.device, n: int = 2048, repeats: int = 5) -> float:
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    _ = a @ b  # warm-up, includes kernel compilation on MPS
    if device.type == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = a @ b
    if device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    device = pick_device()
    print(f"torch {torch.__version__}  device={device}")
    cpu = _time_matmul(torch.device("cpu"))
    print(f"2048x2048 matmul on cpu: {cpu * 1000:.1f} ms")
    if device.type != "cpu":
        fast = _time_matmul(device)
        print(f"2048x2048 matmul on {device}: {fast * 1000:.1f} ms  ({cpu / fast:.1f}x faster)")


if __name__ == "__main__":
    main()
