from __future__ import annotations
from typing import Iterable
import torch
from mlkem_gpu.device import DEVICE


def to_tensor(
    data: Iterable[int] | torch.Tensor,
    dtype: torch.dtype = torch.int64,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    Convert input data to a PyTorch tensor on the selected device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)

    return torch.tensor(data, dtype=dtype, device=device)


def zeros(
    size: int | tuple[int, ...],
    dtype: torch.dtype = torch.int64,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    Create a zero tensor on the selected device.
    """
    return torch.zeros(size, dtype=dtype, device=device)


def empty(
    size: int | tuple[int, ...],
    dtype: torch.dtype = torch.int64,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    Create an empty tensor on the selected device.
    """
    return torch.empty(size, dtype=dtype, device=device)


def to_cpu_list(tensor: torch.Tensor) -> list[int]:
    """
    Move tensor to CPU and convert it to a Python list.
    """
    return tensor.detach().cpu().tolist()