"""
GPU polynomial-vector operations for ML-KEM-512.

A polyvec contains KYBER_K polynomials.
For ML-KEM-512, KYBER_K = 2.
"""

from __future__ import annotations
import torch
from mlkem_gpu.params import KYBER_K, KYBER_N
from mlkem_gpu.gpu_utils import to_tensor, zeros, to_cpu_list
from mlkem_gpu.reduce_gpu import barrett_reduce


class PolyVec:
    """
    Represents one ML-KEM polynomial vector.
    """

    def __init__(self, coeffs=None):
        if coeffs is None:
            self.coeffs = zeros((KYBER_K, KYBER_N), dtype=torch.int16)
        else:
            coeffs_tensor = to_tensor(coeffs, dtype=torch.int16)

            expected_numel = KYBER_K * KYBER_N

            if coeffs_tensor.numel() != expected_numel:
                raise ValueError(
                    f"PolyVec must have exactly {expected_numel} coefficients, "
                    f"got {coeffs_tensor.numel()}"
                )

            self.coeffs = coeffs_tensor.reshape(KYBER_K, KYBER_N)

    def reduce(self) -> "PolyVec":
        """
        Reduce all coefficients in all polynomials.
        """
        self.coeffs = barrett_reduce(self.coeffs)
        return self

    def clone(self) -> "PolyVec":
        """
        Return a copy of the polynomial vector.
        """
        return PolyVec(self.coeffs.clone())

    def tolist(self) -> list[list[int]]:
        """
        Convert polyvec coefficients to nested CPU Python lists.
        """
        return to_cpu_list(self.coeffs)

    def __repr__(self) -> str:
        return f"PolyVec(coeffs_shape={tuple(self.coeffs.shape)}, device={self.coeffs.device})"


def polyvec_add(a: PolyVec, b: PolyVec) -> PolyVec:
    """
    Add two polyvecs coefficient-wise.

    Equivalent to PQClean polyvec_add().
    """
    result = a.coeffs.to(torch.int32) + b.coeffs.to(torch.int32)
    return PolyVec(result).reduce()


def polyvec_reduce(a: PolyVec) -> PolyVec:
    """
    Reduce all coefficients in a polyvec.

    Equivalent to PQClean polyvec_reduce().
    """
    return a.clone().reduce()