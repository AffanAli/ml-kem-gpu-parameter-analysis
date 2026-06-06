"""
Device selection for ML-KEM GPU implementation.

Supports:
- CUDA on NVIDIA GPU, e.g. Windows GTX 1660 Ti
- MPS on Apple Silicon, e.g. Mac M3
- CPU fallback
"""

import torch


def get_device() -> torch.device:
    """
    Select the best available device.

    Priority:
    1. CUDA for NVIDIA GPU
    2. MPS for Apple Silicon GPU
    3. CPU fallback
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


def print_device_info() -> None:
    """
    Print active PyTorch device information.
    """
    print(f"Selected device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    elif DEVICE.type == "mps":
        print("Apple Silicon MPS device available")

    else:
        print("Using CPU fallback")