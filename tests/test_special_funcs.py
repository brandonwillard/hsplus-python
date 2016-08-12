import random
import numpy as np
import mpmath as mp
#import sympy as sp

from hsplus.special_funcs import *


def horn_phi1_bad(a, b, g, x, y):
    return mp.hyper2d({'m+n': a, 'm': b}, {'m+n': g}, x, y)


def horn_phi1_bad_lim(a, b, g, x, y):
    return mp.limit(lambda z: mp.appellf1(a, b, 1 / z, g, x, z * y), 0)


def horn_phi1_naive(a, b, g, x, y):
    return mp.nsum(lambda m, n: mp.rf(a, m + n) * mp.rf(b, n) /
                   mp.rf(g, m + n) * x**m * y**n / mp.fac(m) /
                   mp.fac(n), [0, mp.inf], [0, mp.inf])


def test_horn_phi1():
    assert np.allclose(float(horn_phi1(0.5, 1, 1, 0, 0)),
                       float(horn_phi1_bad(0.5, 1, 1, 0, 0)))


def test_horn_phi1_numeric():

    #
    # Test HS marginal integrals against numeric values using
    #
    assert np.allclose([float(hs_marg_int_num([0.0], 1.0, 1.0, [1.0]))],
                       [float(hs_marg_phi1([0.0], 1.0, 1.0, [1.0]))])

    assert np.allclose([float(hs_marg_int_num([0.0], 1.0, 1.0, [2.0]))],
                       [float(hs_marg_phi1([0.0], 1.0, 1.0, [2.0]))])

    #
    # Test HS marginal integrals against numeric values using
    # numeric differentiation.
    #
    assert np.allclose(float(mp.diff(lambda x: hs_marg_int_num([x], 1.0, 1.0,
                                                               [1.0]), 0.0)),
                       float(mp.diff(lambda x: hs_marg_phi1([x], 1.0, 1.0,
                                                            [1.0]), 0.0)))

    assert np.allclose(float(mp.diff(lambda x: hs_marg_int_num([x], 1.0, 1.0,
                                                               [1.0]), 1.0)),
                       float(mp.diff(lambda x: hs_marg_phi1([x], 1.0, 1.0,
                                                            [1.0]), 1.0)))


def test_horn_phi1_sim():
    N_reps = 5
    N = 5
    tau = 1.0
    sigma = 1.0
    beta_true = mp.matrix(N, 1)
    beta_true[1, 0] = 4.0
    y_til_reps = N_reps * [None]
    for n_rep in range(0, N_reps):
        X = mp.randmatrix(N)
        U, D, V = mp.svd_r(X)
        # TODO: could use np.random.multivariate_normal
        eps = mp.matrix([random.normalvariate(0, 1) for i in xrange(0, N)])
        y = X * beta_true + tau**2 * eps
        D_inv = mp.diag(D.apply(lambda x: 1.0/x))
        y_til = ((V.transpose() * D_inv) * U.transpose()) * y
        y_til_reps[n_rep] = y_til
        # this is more precise, but since we already calculated the svd...
        #y_til = mp.lu_solve(X, y)

    assert np.allclose(float(hs_marg_int_num(y_til_reps[0].transpose().tolist()[0],
                                             tau, sigma,
                                             D.transpose().tolist()[0])),
                       float(hs_marg_phi1(y_til_reps[0].transpose().tolist()[0], tau,
                                          sigma, D.transpose().tolist()[0])))
