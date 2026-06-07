"""
GPU polynomial-vector operations for ML-KEM-512.
"""
from __future__ import annotations

import torch
from mlkem_gpu.params import KYBER_K, KYBER_N
from mlkem_gpu.reduce_gpu import barrett_reduce
from mlkem_gpu.gpu_utils import to_tensor, zeros, to_cpu_list
from mlkem_gpu.poly_gpu import (
    Poly,
    poly_add,
    poly_reduce,
    poly_basemul_montgomery,
)


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


def polyvec_basemul_acc_montgomery(a: PolyVec, b: PolyVec) -> Poly:
    """
    Multiply two PolyVecs in NTT domain and accumulate the result.

    Equivalent to PQClean:

        polyvec_basemul_acc_montgomery(poly *r, const polyvec *a, const polyvec *b)

    For ML-KEM-512:

        r = basemul(a[0], b[0]) + basemul(a[1], b[1])
    """

    r = poly_basemul_montgomery(
        Poly(a.coeffs[0]),
        Poly(b.coeffs[0]),
    )

    for i in range(1, KYBER_K):
        t = poly_basemul_montgomery(
            Poly(a.coeffs[i]),
            Poly(b.coeffs[i]),
        )

        r = poly_add(r, t)

    return poly_reduce(r)