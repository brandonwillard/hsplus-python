import pytest
import numpy as np
from mpmath import mp, fp

from hsplus.hib_stats import SURE_hib, DIC_hib, E_kappa, m_hib
from hsplus.horn_function import horn_phi1


def horn_phi1_bad(a, b, g, x, y):
    return mp.hyper2d({'m+n': a, 'm': b}, {'m+n': g}, x, y)


def horn_phi1_bad_lim(a, b, g, x, y):
    return mp.limit(lambda z: mp.appellf1(a, b, 1. / z, g, z * x, y), 0)


def horn_phi1_naive(a, b, g, x, y):
    return mp.nsum(lambda m, n: mp.rf(a, m + n) * mp.rf(b, m) /
                   mp.rf(g, m + n) * x**m * y**n / mp.fac(m) /
                   mp.fac(n), [0, mp.inf], [0, mp.inf])


def test_horn_phi1():
    assert np.allclose(float(horn_phi1(0.5, 1, 1, 0, 0)),
                       float(horn_phi1_bad(0.5, 1, 1, 0, 0)))


@pytest.mark.parametrize("func", [SURE_hib, DIC_hib, E_kappa, m_hib])
def test_vectorized(func):

    val_1 = func(1., sigma=1., tau=1.)
    assert np.shape(val_1) == ()

    val_2 = func(np.array([1., 1.]), sigma=1., tau=1.)
    assert np.shape(val_2) == (2,)

    np.testing.assert_array_equal(val_1, val_2)

    val_3 = func(np.array([1., 1.]), sigma=1., tau=np.array([0.1, 1.]))
    assert np.shape(val_3) == (2,)

    assert val_3[0] != val_2[0] and val_3[1] == val_2[0]


def test_sure_points():

    # SURE_hib(np.array([1., 2.]), np.ones(2), np.array([1., 2.]))

    sure_val_1 = SURE_hib(1.53522076e+01, sigma=1., tau=1.)
    np.testing.assert_almost_equal(sure_val_1, 2.03431435, decimal=4)

    sure_val_6 = SURE_hib(np.array([1e3, 1e4, -1e20, 1e20]),
                          sigma=1.,
                          tau=np.array([1., 1., 1., 10.5]))
    np.testing.assert_almost_equal(sure_val_6, 2.0, decimal=4)

    sure_val_7 = SURE_hib(np.array([1e2, 1e3]), sigma=1., tau=0.5)
    np.testing.assert_almost_equal(sure_val_7[0], 2.0, decimal=2)
    np.testing.assert_almost_equal(sure_val_7[1], 2.0, decimal=3)
