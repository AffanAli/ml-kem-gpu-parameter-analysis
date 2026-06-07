import torch
from mlkem_gpu.device import DEVICE
from mlkem_gpu.params import KYBER_K, KYBER_N
from mlkem_gpu.poly_gpu import (
    Poly,
    poly_ntt,
)
from mlkem_gpu.polyvec_gpu import (
    PolyVec, 
    polyvec_add,
    PolyVec,
    polyvec_basemul_acc_montgomery,
)


def test_polyvec_shape():
    pv = PolyVec()
    assert pv.coeffs.shape == (KYBER_K, KYBER_N)


def test_polyvec_add():
    a = PolyVec([
        [3000] * KYBER_N,
        [1000] * KYBER_N,
    ])

    b = PolyVec([
        [1000] * KYBER_N,
        [3000] * KYBER_N,
    ])

    c = polyvec_add(a, b)

    assert c.coeffs.shape == (KYBER_K, KYBER_N)
    assert c.tolist()[0][:5] == [671, 671, 671, 671, 671]
    assert c.tolist()[1][:5] == [671, 671, 671, 671, 671]

def test_polyvec_basemul_acc_returns_poly():
    """
    verify Return type
    """
    a = PolyVec(
        torch.arange(
            KYBER_K * KYBER_N,
            dtype=torch.int16,
            device=DEVICE,
        )
    )

    b = PolyVec(
        torch.arange(
            KYBER_K * KYBER_N,
            dtype=torch.int16,
            device=DEVICE,
        )
    )

    for i in range(KYBER_K):
        a.coeffs[i] = poly_ntt(Poly(a.coeffs[i])).coeffs
        b.coeffs[i] = poly_ntt(Poly(b.coeffs[i])).coeffs

    result = polyvec_basemul_acc_montgomery(a, b)

    assert isinstance(result, Poly)


def test_polyvec_basemul_acc_shape():
    """
    verify shape
    """
    a = PolyVec(
        torch.arange(
            KYBER_K * KYBER_N,
            dtype=torch.int16,
            device=DEVICE,
        )
    )

    b = PolyVec(
        torch.arange(
            KYBER_K * KYBER_N,
            dtype=torch.int16,
            device=DEVICE,
        )
    )

    for i in range(KYBER_K):
        a.coeffs[i] = poly_ntt(Poly(a.coeffs[i])).coeffs
        b.coeffs[i] = poly_ntt(Poly(b.coeffs[i])).coeffs

    result = polyvec_basemul_acc_montgomery(a, b)

    assert result.coeffs.shape == (KYBER_N,)


def test_polyvec_basemul_acc_known_output_first_10():
    """
    verify with Known Output
    """
    a = PolyVec(
        torch.arange(
            KYBER_K * KYBER_N,
            dtype=torch.int16,
            device=DEVICE,
        )
    )

    b = PolyVec(
        torch.arange(
            KYBER_K * KYBER_N,
            dtype=torch.int16,
            device=DEVICE,
        )
    )

    for i in range(KYBER_K):
        a.coeffs[i] = poly_ntt(Poly(a.coeffs[i])).coeffs
        b.coeffs[i] = poly_ntt(Poly(b.coeffs[i])).coeffs

    result = polyvec_basemul_acc_montgomery(a, b)

    expected = torch.tensor(
        [-498, 961, 955, -43, -145, -197, -531, -285, -378, -1444],
        dtype=torch.int16,
        device=DEVICE,
    )

    assert torch.equal(
        result.coeffs[:10],
        expected,
    )


def test_polyvec_basemul_acc_output_range():
    """
    verify coefficients are centered.
    """
    a = PolyVec(
        torch.arange(
            KYBER_K * KYBER_N,
            dtype=torch.int16,
            device=DEVICE,
        )
    )

    b = PolyVec(
        torch.arange(
            KYBER_K * KYBER_N,
            dtype=torch.int16,
            device=DEVICE,
        )
    )

    for i in range(KYBER_K):
        a.coeffs[i] = poly_ntt(Poly(a.coeffs[i])).coeffs
        b.coeffs[i] = poly_ntt(Poly(b.coeffs[i])).coeffs

    result = polyvec_basemul_acc_montgomery(a, b)

    assert torch.all(result.coeffs >= -1664)
    assert torch.all(result.coeffs <= 1664)