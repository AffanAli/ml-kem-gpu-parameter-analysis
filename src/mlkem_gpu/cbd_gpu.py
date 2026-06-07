"""
GPU-based Centered Binomial Distribution sampling for ML-KEM-512.

CBD converts random bytes into small noise polynomial coefficients.
"""

from __future__ import annotations
import torch
from mlkem_gpu.params import KYBER_N, KYBER_ETA1, KYBER_ETA2
from mlkem_gpu.gpu_utils import to_tensor
from mlkem_gpu.poly_gpu import Poly


def _validate_byte_buffer(buf: torch.Tensor, expected_len: int) -> torch.Tensor:
    """
    Validate and convert input byte buffer.
    """
    buf = to_tensor(buf, dtype=torch.uint8)

    if buf.numel() != expected_len:
        raise ValueError(
            f"Expected buffer length {expected_len}, got {buf.numel()}"
        )

    return buf.reshape(expected_len)


def load32_littleendian(buf: torch.Tensor) -> torch.Tensor:
    """
    Load 4 bytes as one uint32 value in little-endian order.

    Equivalent to PQClean load32_littleendian().
    """
    buf = _validate_byte_buffer(buf, 4).to(torch.int64)

    return (
        buf[0]
        | (buf[1] << 8)
        | (buf[2] << 16)
        | (buf[3] << 24)
    )


def load24_littleendian(buf: torch.Tensor) -> torch.Tensor:
    """
    Load 3 bytes as one uint32-like value in little-endian order.

    Equivalent to PQClean load24_littleendian().
    """
    buf = _validate_byte_buffer(buf, 3).to(torch.int64)

    return (
        buf[0]
        | (buf[1] << 8)
        | (buf[2] << 16)
    )


def cbd2(buf: torch.Tensor) -> Poly:
    """
    CBD with eta = 2.

    Produces coefficients in range [-2, 2].
    """
    expected_len = KYBER_ETA2 * KYBER_N // 4
    buf = _validate_byte_buffer(buf, expected_len)

    coeffs = torch.empty(
        KYBER_N,
        dtype=torch.int16,
        device=buf.device,
    )

    for i in range(KYBER_N // 8):
        t = load32_littleendian(buf[4 * i : 4 * i + 4])

        d = t & 0x55555555
        d = d + ((t >> 1) & 0x55555555)

        for j in range(8):
            a = (d >> (4 * j + 0)) & 0x3
            b = (d >> (4 * j + 2)) & 0x3

            coeffs[8 * i + j] = (a - b).to(torch.int16)

    return Poly(coeffs)


def cbd3(buf: torch.Tensor) -> Poly:
    """
    CBD with eta = 3.

    Produces coefficients in range [-3, 3].
    """
    expected_len = KYBER_ETA1 * KYBER_N // 4
    buf = _validate_byte_buffer(buf, expected_len)

    coeffs = torch.empty(
        KYBER_N,
        dtype=torch.int16,
        device=buf.device,
    )

    for i in range(KYBER_N // 4):
        t = load24_littleendian(buf[3 * i : 3 * i + 3])

        d = t & 0x00249249
        d = d + ((t >> 1) & 0x00249249)
        d = d + ((t >> 2) & 0x00249249)

        for j in range(4):
            a = (d >> (6 * j + 0)) & 0x7
            b = (d >> (6 * j + 3)) & 0x7

            coeffs[4 * i + j] = (a - b).to(torch.int16)

    return Poly(coeffs)


def poly_cbd_eta1(buf: torch.Tensor) -> Poly:
    """
    Equivalent to PQClean poly_cbd_eta1().
    """
    return cbd3(buf)


def poly_cbd_eta2(buf: torch.Tensor) -> Poly:
    """
    Equivalent to PQClean poly_cbd_eta2().
    """
    return cbd2(buf)


## ==============================================================================
## Following are optimized function for tensor
## ==============================================================================
def cbd2_batched(buf: torch.Tensor) -> torch.Tensor:
    """
    Batched CBD with eta = 2.

    Input shape:
        (batch_size, 128)

    Output shape:
        (batch_size, 256)

    Each row produces one ML-KEM polynomial.
    """
    expected_len = KYBER_ETA2 * KYBER_N // 4

    buf = to_tensor(buf, dtype=torch.uint8)

    if buf.dim() != 2 or buf.shape[1] != expected_len:
        raise ValueError(
            f"Expected shape (batch_size, {expected_len}), got {tuple(buf.shape)}"
        )

    b = buf.to(torch.int64)
    batch_size = b.shape[0]

    x0 = b[:, 0::4]
    x1 = b[:, 1::4]
    x2 = b[:, 2::4]
    x3 = b[:, 3::4]

    t = x0 | (x1 << 8) | (x2 << 16) | (x3 << 24)

    d = t & 0x55555555
    d = d + ((t >> 1) & 0x55555555)

    coeffs = torch.empty(
        (batch_size, KYBER_N),
        dtype=torch.int16,
        device=buf.device,
    )

    for j in range(8):
        a = (d >> (4 * j + 0)) & 0x3
        c = (d >> (4 * j + 2)) & 0x3
        coeffs[:, j::8] = (a - c).to(torch.int16)

    return coeffs


def cbd3_batched(buf: torch.Tensor) -> torch.Tensor:
    """
    Batched CBD with eta = 3.

    Input shape:
        (batch_size, 192)

    Output shape:
        (batch_size, 256)

    Each row produces one ML-KEM polynomial.
    """
    expected_len = KYBER_ETA1 * KYBER_N // 4

    buf = to_tensor(buf, dtype=torch.uint8)

    if buf.dim() != 2 or buf.shape[1] != expected_len:
        raise ValueError(
            f"Expected shape (batch_size, {expected_len}), got {tuple(buf.shape)}"
        )

    b = buf.to(torch.int64)
    batch_size = b.shape[0]

    x0 = b[:, 0::3]
    x1 = b[:, 1::3]
    x2 = b[:, 2::3]

    t = x0 | (x1 << 8) | (x2 << 16)

    d = t & 0x00249249
    d = d + ((t >> 1) & 0x00249249)
    d = d + ((t >> 2) & 0x00249249)

    coeffs = torch.empty(
        (batch_size, KYBER_N),
        dtype=torch.int16,
        device=buf.device,
    )

    for j in range(4):
        a = (d >> (6 * j + 0)) & 0x7
        c = (d >> (6 * j + 3)) & 0x7
        coeffs[:, j::4] = (a - c).to(torch.int16)

    return coeffs


def poly_cbd_eta1_batched(buf: torch.Tensor) -> torch.Tensor:
    """
    Batched version of poly_cbd_eta1().

    Returns a tensor, not a Poly object.

    Shape:
        (batch_size, 256)
    """
    return cbd3_batched(buf)


def poly_cbd_eta2_batched(buf: torch.Tensor) -> torch.Tensor:
    """
    Batched version of poly_cbd_eta2().

    Returns a tensor, not a Poly object.

    Shape:
        (batch_size, 256)
    """
    return cbd2_batched(buf)