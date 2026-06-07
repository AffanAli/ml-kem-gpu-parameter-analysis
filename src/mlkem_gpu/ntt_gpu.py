"""
GPU-based Number Theoretic Transform for ML-KEM-512.
"""

from __future__ import annotations
import torch
from mlkem_gpu.gpu_utils import to_tensor
from mlkem_gpu.poly_gpu import Poly
from mlkem_gpu.reduce_gpu import montgomery_reduce, barrett_reduce


ZETAS = [
    -1044, -758, -359, -1517, 1493, 1422, 287, 202,
    -171, 622, 1577, 182, 962, -1202, -1474, 1468,
    573, -1325, 264, 383, -829, 1458, -1602, -130,
    -681, 1017, 732, 608, -1542, 411, -205, -1571,
    1223, 652, -552, 1015, -1293, 1491, -282, -1544,
    516, -8, -320, -666, -1618, -1162, 126, 1469,
    -853, -90, -271, 830, 107, -1421, -247, -951,
    -398, 961, -1508, -725, 448, -1065, 677, -1275,
    -1103, 430, 555, 843, -1251, 871, 1550, 105,
    422, 587, 177, -235, -291, -460, 1574, 1653,
    -246, 778, 1159, -147, -777, 1483, -602, 1119,
    -1590, 644, -872, 349, 418, 329, -156, -75,
    817, 1097, 603, 610, 1322, -1285, -1465, 384,
    -1215, -136, 1218, -1335, -874, 220, -1187, -1659,
    -1185, -1530, -1278, 794, -1510, -854, -870, 478,
    -108, -308, 996, 991, 958, -1460, 1522, 1628,
]


def zetas_tensor(device=None) -> torch.Tensor:
    """
    Return zetas as tensor on selected device.
    """
    return to_tensor(ZETAS, dtype=torch.int16, device=device)


def fqmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Finite-field multiplication.
    """
    product = a.to(torch.int64) * b.to(torch.int64)
    return montgomery_reduce(product)


def ntt_coeffs(r: torch.Tensor) -> torch.Tensor:
    """
    Forward NTT on one polynomial.

    Input:
        Tensor shape: (256,)

    Output:
        Tensor shape: (256,)
    """
    r = to_tensor(r, dtype=torch.int16).clone()

    if r.numel() != 256:
        raise ValueError(f"NTT expects 256 coefficients, got {r.numel()}")

    r = r.reshape(256)

    zetas = zetas_tensor(device=r.device)

    k = 1
    length = 128

    while length >= 2:
        for start in range(0, 256, 2 * length):
            zeta = zetas[k]
            k += 1

            left_idx = torch.arange(
                start,
                start + length,
                device=r.device,
            )

            right_idx = left_idx + length

            t = fqmul(zeta, r[right_idx])

            left = r[left_idx].to(torch.int32)
            t32 = t.to(torch.int32)

            r[right_idx] = (left - t32).to(torch.int16)
            r[left_idx] = (left + t32).to(torch.int16)

        length //= 2

    return r


def poly_ntt(a: Poly) -> Poly:
    """
    Forward NTT for a Poly object.
    """
    coeffs = ntt_coeffs(a.coeffs)
    coeffs = barrett_reduce(coeffs)

    return Poly(coeffs)


def invntt_coeffs(r: torch.Tensor) -> torch.Tensor:
    """
    Inverse NTT on one polynomial.

    Input:
        Tensor shape: (256,)

    Output:
        Tensor shape: (256,)
    """
    r = to_tensor(r, dtype=torch.int16).clone()

    if r.numel() != 256:
        raise ValueError(f"Inverse NTT expects 256 coefficients, got {r.numel()}")

    r = r.reshape(256)

    zetas = zetas_tensor(device=r.device)

    f = torch.tensor(
        1441,
        dtype=torch.int16,
        device=r.device,
    )

    k = 127
    length = 2

    while length <= 128:
        for start in range(0, 256, 2 * length):
            zeta = zetas[k]
            k -= 1

            left_idx = torch.arange(
                start,
                start + length,
                device=r.device,
            )

            right_idx = left_idx + length

            t = r[left_idx].clone()

            r[left_idx] = barrett_reduce(
                t.to(torch.int32) + r[right_idx].to(torch.int32)
            )

            diff = r[right_idx].to(torch.int32) - t.to(torch.int32)
            r[right_idx] = fqmul(zeta, diff)

        length *= 2

    r = fqmul(r, f)

    return r


def poly_invntt_tomont(a: Poly) -> Poly:
    """
    Inverse NTT for a Poly object.

    Equivalent to PQClean:

        poly_invntt_tomont(poly *r) {
            invntt(r->coeffs);
        }
    """
    coeffs = invntt_coeffs(a.coeffs)
    return Poly(coeffs)


def basemul(
    a: torch.Tensor,
    b: torch.Tensor,
    zeta: int | torch.Tensor,
) -> torch.Tensor:
    """
    Equivalent to PQClean basemul().

    Input:
        a shape = (2,)
        b shape = (2,)

    Output:
        shape = (2,)
    """

    a = to_tensor(a, dtype=torch.int16)
    b = to_tensor(b, dtype=torch.int16)

    if a.numel() != 2:
        raise ValueError("a must contain exactly 2 coefficients")

    if b.numel() != 2:
        raise ValueError("b must contain exactly 2 coefficients")

    zeta = to_tensor(
        zeta,
        dtype=torch.int16,
        device=a.device,
    )

    r0 = fqmul(a[1], b[1])
    r0 = fqmul(r0, zeta)
    r0 = r0 + fqmul(a[0], b[0])

    r1 = fqmul(a[0], b[1])
    r1 = r1 + fqmul(a[1], b[0])

    return torch.stack(
        [
            r0.to(torch.int16),
            r1.to(torch.int16),
        ]
    )

def poly_basemul_montgomery(a: Poly, b: Poly) -> Poly:
    """
    Polynomial base multiplication in NTT domain.

    Equivalent to PQClean:

        poly_basemul_montgomery(poly *r, const poly *a, const poly *b)

    Input:
        a, b: Poly objects already in NTT domain

    Output:
        Poly object
    """
    result = torch.empty(
        256,
        dtype=torch.int16,
        device=a.coeffs.device,
    )

    zetas = zetas_tensor(device=a.coeffs.device)

    for i in range(256 // 4):
        zeta = zetas[64 + i]

        result[4 * i : 4 * i + 2] = basemul(
            a.coeffs[4 * i : 4 * i + 2],
            b.coeffs[4 * i : 4 * i + 2],
            zeta,
        )

        result[4 * i + 2 : 4 * i + 4] = basemul(
            a.coeffs[4 * i + 2 : 4 * i + 4],
            b.coeffs[4 * i + 2 : 4 * i + 4],
            -zeta,
        )

    return Poly(result)