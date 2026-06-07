import torch

from mlkem_gpu.params import KYBER_K, KYBER_N
from mlkem_gpu.poly_gpu import Poly, poly_add, poly_sub
from mlkem_gpu.polyvec_gpu import PolyVec, polyvec_add


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