import pytest
import numpy as np
from mpmath import mp, fp

from hsplus.hib_stats import (SURE_hib, DIC_hib, E_kappa, m_hib)
import hsplus.horn_numeric
from hsplus.horn_numeric import (
    horn_phi1,
    phi1_T1_x,
    phi1_T2_x,
    phi1_T3_y,
    phi1_T4_y)


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


@pytest.mark.parametrize("mp_test_ctx", [mp, fp])
def test_horn_y_singular(mp_test_ctx):
    r""" Test singular points.
    """
    hsplus.horn_numeric.mp_ctx = mp_test_ctx

    phi_args = [1.0, 1, 1.5]
    xys = [[0.5, 1], [0, 1], [10, 1], [-10, 1]]

    for xy in xys:
        assert horn_phi1(*(phi_args + xy)) == mp_test_ctx.inf

    # hsplus.horn_numeric.mp_ctx = fp
    # hsplus.horn_numeric.mp_ctx = mp

    # phi_args = (2.0, 1, 2.5, -1e10, 1)

    # horn_phi1(*phi_args)
    # # %timeit horn_phi1(*phi_args)

    # mp.nsum(lambda n: phi1_T1_x(n, *phi_args), [0, mp.inf])
    # # %timeit mp.nsum(lambda n: phi1_T1_x(n, *phi_args), [0, mp.inf])

    # mp.exp(phi_args[-2]) * mp.nsum(lambda n: phi1_T2_x(n, *phi_args), [0, mp.inf])
    # # %timeit mp.exp(phi_args[-2]) * mp.nsum(lambda n: phi1_T2_x(n, *phi_args), [0, mp.inf])

    # mp.nsum(lambda n: phi1_T3_y(n, *phi_args), [0, mp.inf])
    # # %timeit mp.nsum(lambda n: phi1_T3_y(n, *phi_args), [0, mp.inf])

    # mp.exp(phi_args[-2]) * mp.nsum(lambda n: phi1_T4_y(n, *phi_args), [0, mp.inf])
    # # %timeit mp.exp(phi_args[-2]) * mp.nsum(lambda n: phi1_T4_y(n, *phi_args), [0, mp.inf])

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
