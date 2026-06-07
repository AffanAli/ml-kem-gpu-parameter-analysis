import torch

from mlkem_gpu.device import DEVICE
from mlkem_gpu.params import KYBER_K, KYBER_N
from mlkem_gpu.poly_gpu import (
    Poly, 
    poly_add, 
    poly_sub,
    poly_ntt,
    poly_invntt_tomont,
    poly_basemul_montgomery    
)

def test_poly_shape():
    p = Poly()
    assert p.coeffs.shape == (KYBER_N,)


def test_poly_add():
    a = Poly([3000] * KYBER_N)
    b = Poly([1000] * KYBER_N)

    c = poly_add(a, b)

    assert c.tolist()[:5] == [671, 671, 671, 671, 671]


def test_poly_sub():
    a = Poly([3000] * KYBER_N)
    b = Poly([1000] * KYBER_N)

    c = poly_sub(a, b)

    assert c.tolist()[:5] == [-1329, -1329, -1329, -1329, -1329]


def test_poly_ntt_returns_poly():
    p = Poly(torch.arange(KYBER_N, dtype=torch.int16, device=DEVICE))

    result = poly_ntt(p)

    assert isinstance(result, Poly)
    assert result.coeffs.shape == (KYBER_N,)
    assert result.coeffs.device.type == DEVICE.type


def test_poly_ntt_known_output_first_10():
    p = Poly(torch.arange(KYBER_N, dtype=torch.int16, device=DEVICE))

    result = poly_ntt(p)

    expected = torch.tensor(
        [-900, -484, 425, 795, -1464, 1356, 624, 31, -846, -1132],
        dtype=torch.int16,
        device=DEVICE,
    )

    assert torch.equal(result.coeffs[:10], expected)


def test_poly_invntt_tomont_returns_poly():
    p = Poly(torch.arange(KYBER_N, dtype=torch.int16, device=DEVICE))

    p_ntt = poly_ntt(p)
    result = poly_invntt_tomont(p_ntt)

    assert isinstance(result, Poly)
    assert result.coeffs.shape == (KYBER_N,)
    assert result.coeffs.device.type == DEVICE.type


def test_poly_invntt_tomont_known_output_first_10():
    p = Poly(torch.arange(KYBER_N, dtype=torch.int16, device=DEVICE))

    p_ntt = poly_ntt(p)
    result = poly_invntt_tomont(p_ntt)

    expected = torch.tensor(
        [0, -1044, 1241, 197, -847, 1438, 394, -650, 1635, 591],
        dtype=torch.int16,
        device=DEVICE,
    )

    assert torch.equal(result.coeffs[:10], expected)


def test_poly_basemul_montgomery_returns_poly():
    a = Poly(torch.arange(KYBER_N, dtype=torch.int16, device=DEVICE))
    b = Poly(torch.arange(KYBER_N, dtype=torch.int16, device=DEVICE))

    a_ntt = poly_ntt(a)
    b_ntt = poly_ntt(b)

    result = poly_basemul_montgomery(a_ntt, b_ntt)

    assert isinstance(result, Poly)
    assert result.coeffs.shape == (KYBER_N,)
    assert result.coeffs.device.type == DEVICE.type


def test_poly_basemul_montgomery_known_output_first_10():
    a = Poly(torch.arange(KYBER_N, dtype=torch.int16, device=DEVICE))
    b = Poly(torch.arange(KYBER_N, dtype=torch.int16, device=DEVICE))

    a_ntt = poly_ntt(a)
    b_ntt = poly_ntt(b)

    result = poly_basemul_montgomery(a_ntt, b_ntt)

    expected = torch.tensor(
        [1736, -2212, -2751, -2924, -1401, 1048, -1406, 116, 868, 1150],
        dtype=torch.int16,
        device=DEVICE,
    )

    assert torch.equal(result.coeffs[:10], expected)