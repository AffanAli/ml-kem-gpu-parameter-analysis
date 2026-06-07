import torch
from mlkem_gpu.params import KYBER_K, KYBER_N
from mlkem_gpu.polyvec_gpu import PolyVec, polyvec_add


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