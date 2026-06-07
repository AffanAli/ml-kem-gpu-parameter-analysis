import torch

from mlkem_gpu.device import DEVICE
from mlkem_gpu.params import KYBER_N
from mlkem_gpu.poly_gpu import Poly
from mlkem_gpu.ntt_gpu import ntt_coeffs, poly_ntt


def test_ntt_output_shape():
    x = torch.arange(
        KYBER_N,
        dtype=torch.int16,
        device=DEVICE,
    )

    y = ntt_coeffs(x)

    assert y.shape == (KYBER_N,)


def test_ntt_output_device():
    x = torch.arange(
        KYBER_N,
        dtype=torch.int16,
        device=DEVICE,
    )

    y = ntt_coeffs(x)

    assert y.device.type == DEVICE.type


def test_ntt_zero_polynomial_remains_zero():
    x = torch.zeros(
        KYBER_N,
        dtype=torch.int16,
        device=DEVICE,
    )

    y = ntt_coeffs(x)

    assert torch.all(y == 0)


def test_poly_ntt_returns_poly():
    p = Poly(torch.arange(KYBER_N, dtype=torch.int16, device=DEVICE))

    result = poly_ntt(p)

    assert isinstance(result, Poly)
    assert result.coeffs.shape == (KYBER_N,)


def test_poly_ntt_known_output_first_10():
    p = Poly(torch.arange(KYBER_N, dtype=torch.int16, device=DEVICE))

    result = poly_ntt(p)

    expected_first_10 = torch.tensor(
        [-900, -484, 425, 795, -1464, 1356, 624, 31, -846, -1132],
        dtype=torch.int16,
        device=DEVICE,
    )

    assert torch.equal(result.coeffs[:10], expected_first_10)


def test_poly_ntt_output_is_centered_range():
    p = Poly(torch.arange(KYBER_N, dtype=torch.int16, device=DEVICE))

    result = poly_ntt(p)

    assert torch.all(result.coeffs >= -1664)
    assert torch.all(result.coeffs <= 1664)