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


def test_sure():
    #from hsplus.special_funcs import m_hib, sure_hib, E_kappa, horn_phi1

    tau = 1.0
    sigma = 1.

    alpha_d_val_1 = 1.53522076e+01
    sure_val_1 = sure_hib(alpha_d_val_1, 1., 1.)
    np.testing.assert_almost_equal(sure_val_1, 2.03431435, decimal=3)

    alpha_d_val_2 = 1.53522076246375913655128897516988217830657958984375e+01
    sure_val_2 = sure_hib(alpha_d_val_2, 1., 1.)
    np.testing.assert_almost_equal(sure_val_2, 2.03431435, decimal=3)


def sure_scratch():
    alpha_d_val_3 = 114.
    sure_hib(alpha_d_val_3, 1., 1.)

    # horn_phi1_single args:
    a = 0.5
    a = mp.fraction(1, 2)
    b = 1.0
    g = 4.5
    g = mp.fraction(9, 2)
    # for the denominator Phi1 term...
    g = 3.5
    g = mp.fraction(7, 2)
    x = 6498.0
    y = 0.0

    #res_phis = horn_phi1(b, 1., a_p + b + n, s_p, 1. - tau**(-2))
    #res_phis /= horn_phi1(b, 1., a_p + b, s_p, 1. - tau**(-2))

    #mp.hyper2d({'m+n':a, 'n':b}, {'m+n':g}, x, y)

    def T1(m):
        res = mp.rf(a, m) / mp.rf(g, m)

        # This term is very large until about
        # m >= 3 * np.ceil(x)
        res *= mp.power(x, m) / mp.fac(m)

        res *= mp.hyp2f1(b, a + m, g + m, y)

        return res

    m_x = int(3 * np.ceil(x))

    m_range = xrange(0, m_x, m_x // 100)
    t1_vals = np.array([T1(m_) for m_ in m_range], dtype=np.object)

    fig, ax = plt.subplots(1, 1)
    ax.clear()
    ax.plot(m_range, t1_vals)
    ax.set_yscale('log', basey=1000)
    fig.tight_layout()

    res = mp.nsum(T1, [0, mp.inf],
                  method='l+s',  # 'r+s+l+a+e+d'
                  maxterms=10 * m_x,  # 10 * mp.dps
                  verbose=True,
                  steps=m_x // 3 * np.arange(1, 4, 1)
                  #steps=[int(3 * np.ceil(x))] + np.arange(10, 100, 10)
                  )

    res = mp.nsum(lambda m: (mp.rf(a, m) / mp.rf(g, m)) *
                  # This term is very large until about
                  # m >= 3 * np.ceil(x)
                  (x**m / mp.fac(m)) *
                  mp.hyp2f1(b, a + m, g + m, y), #, asymp_tol=1e-4),
                  [0, mp.inf],
                  method='l+s',  # 'r+s+l+a+e+d'
                  maxterms=10 * int(3 * np.ceil(x)),  # 10 * mp.dps
                  verbose=True,
                  steps=int(3 * np.ceil(x)) // 3 * np.arange(1, 4, 1)
                  #steps=[int(3 * np.ceil(x))] + np.arange(10, 100, 10)
                  )

    mp.mpf('7.5911198071895191e+2810') / mp.mpf('4.0890989741902681e+2807')

    mp.hyp2f1(a, 1, g, x, asymp_tol=1e-4)
    (1. - x)**(-a) * mp.hyp2f1(a, g-1, g, x/(x-1.), asymp_tol=1e-4)

    / mp.hyp2f1(a, 1, g - 1, x, asymp_tol=1e-4)

    # (T3)
    res = mp.nsum(lambda m: (mp.rf(a, m) * mp.rf(b, m) / mp.rf(g, m)) *
                  (y**m / mp.fac(m)) *
                  mp.hyp1f1(a + m, g + m, x),
                  [0, mp.inf],
                  method='e' # 'r+s+l+a+e+d'
                  )

    res = mp.exp(x)
    res *= mp.nsum(lambda m: (mp.rf(g - a, m) / mp.rf(g, m)) *
                    ((-x)**m / mp.fac(m)) *
                    mp.hyp2f1(b, a, g + m, y),
                    [0, mp.inf])
