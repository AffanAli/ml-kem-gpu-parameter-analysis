"""
GPU-based modular reduction operations for ML-KEM-512.
"""

import torch

from mlkem_gpu.params import KYBER_Q
from mlkem_gpu.gpu_utils import to_tensor


# From PQClean reduce.h
MONT = -1044  # 2^16 mod q
QINV = -3327  # q^-1 mod 2^16

def montgomery_reduce(a: torch.Tensor) -> torch.Tensor:
    """
    Montgomery reduction.

    Computes a value congruent to: a * R^-1 mod q
    where :
        R = 2^16.
    Input:
        a: tensor of integers

    Output:
        tensor reduced modulo KYBER_Q
    """
    a = to_tensor(a, dtype=torch.int64)

    t = (a.to(torch.int16) * QINV).to(torch.int16)
    t = (a - t.to(torch.int64) * KYBER_Q) >> 16

    return t.to(torch.int16)


def barrett_reduce(a: torch.Tensor) -> torch.Tensor:
    """
    Barrett reduction.

    Computes centered representative of a modulo q.
    """
    a = to_tensor(a, dtype=torch.int64)

    v = ((1 << 26) + KYBER_Q // 2) // KYBER_Q

    t = ((v * a + (1 << 25)) >> 26)
    t = t * KYBER_Q

    return (a - t).to(torch.int16)


def freeze(a: torch.Tensor) -> torch.Tensor:
    """
    Convenience wrapper for reducing coefficients into centered range.
    """
    return barrett_reduce(a)