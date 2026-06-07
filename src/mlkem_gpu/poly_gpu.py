"""
GPU polynomial operations for ML-KEM-512.

A polynomial is represented as 256 int16 coefficients.
"""

from __future__ import annotations
import torch
from mlkem_gpu.params import KYBER_N
from mlkem_gpu.gpu_utils import to_tensor, zeros, to_cpu_list
from mlkem_gpu.reduce_gpu import barrett_reduce


class Poly:
    """
    Represents one ML-KEM polynomial.
    """

    def __init__(self, coeffs=None):
        if coeffs is None:
            self.coeffs = zeros(KYBER_N, dtype=torch.int16)
        else:
            coeffs_tensor = to_tensor(coeffs, dtype=torch.int16)

            if coeffs_tensor.numel() != KYBER_N:
                raise ValueError(
                    f"Poly must have exactly {KYBER_N} coefficients, "
                    f"got {coeffs_tensor.numel()}"
                )

            self.coeffs = coeffs_tensor.reshape(KYBER_N)

    def reduce(self) -> "Poly":
        """
        Reduce all coefficients modulo KYBER_Q using Barrett reduction.
        """
        self.coeffs = barrett_reduce(self.coeffs)
        return self

    def tolist(self) -> list[int]:
        """
        Convert polynomial coefficients to a CPU Python list.
        """
        return to_cpu_list(self.coeffs)

    def clone(self) -> "Poly":
        """
        Return a copy of the polynomial.
        """
        return Poly(self.coeffs.clone())

    def __repr__(self) -> str:
        return f"Poly(coeffs_shape={tuple(self.coeffs.shape)}, device={self.coeffs.device})"


def poly_add(a: Poly, b: Poly) -> Poly:
    """
    Add two polynomials coefficient-wise.

    Equivalent to PQClean poly_add().
    """
    return Poly(a.coeffs.to(torch.int32) + b.coeffs.to(torch.int32)).reduce()


def poly_sub(a: Poly, b: Poly) -> Poly:
    """
    Subtract two polynomials coefficient-wise.

    Equivalent to PQClean poly_sub().
    """
    return Poly(a.coeffs.to(torch.int32) - b.coeffs.to(torch.int32)).reduce()


def poly_reduce(a: Poly) -> Poly:
    """
    Reduce polynomial coefficients.

    Equivalent to PQClean poly_reduce().
    """
    return a.clone().reduce()