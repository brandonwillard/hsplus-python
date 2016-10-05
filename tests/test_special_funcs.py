import numpy as np
from mpmath import mp, fp

from hsplus.hib_stats import SURE_hib
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


def test_sure_point():
    alpha_d_val_1 = 1.53522076e+01
    sure_val_1 = SURE_hib(alpha_d_val_1, 1., 1.)
    np.testing.assert_almost_equal(sure_val_1, 2.03431435, decimal=5)


def _test_asymptotics():
    """ The ratios of marginal posteriors and their derivatives,
    .. math:
        \frac{m^\prime(y)}{m(y)} \;,

    (e.g. SURE) given by

    .. math:
        m(y) \propto \Phi_1(b, 1, a^\prime + b; s^\prime, 1 - \tau^{-2})
        \\
        \frac{d m}{dy}(y) \equiv m^\prime(y) \propto
        \Phi_1(b, 1, a^\prime + b + 1; s^\prime, 1 - \tau^{-2})
        \;,

    for :math:`s^\prime = s + y^2/(2 \sigma^2)` and
    :math:`a^\prime = a + 1/2`, are sensitive to large :math:`y`.
    Sometimes these ratios converge, so we need to a stable, fast numeric
    approach that also converges.

    .. note:
        :math:`a = b = 1/2, s = 0` for the Horseshoe prior.

    """
    # CCH/m(y) params:
    mp.dps = 15
    z_1, z_2 = 421.25, 0.
    #z_1, z_2 = 10., 0.

    (horn_phi1_naive(1/2., 1., 3/2., z_1, z_2) /
     horn_phi1_naive(1/2., 1., 1., z_1, z_2))
    # mpf('0.16716563382409313')

    (horn_phi1_naive(1/2., 1., 3/2., z_1, z_2) /
     horn_phi1_naive(1/2., 1., 3/2, z_1, z_2))

    # Denominator of derivative term
    mp.appellf1(alpha, beta_, 1., gamma_d1, z_1, z_2)

    # m(y) calc has this ratio:
    mp.appellf1(alpha, beta_, 1., gamma_, z_1, z_2)
    mp.appellf1(alpha, beta_, 1., gamma_denom, z_1_denom, z_2)

    # Here's an F1 expansion around z_1 -> inf
    mp.appellf1(alpha, beta_, 1., gamma_, z_1, z_2)

    def F1_z1_inf(alpha, beta_1, beta_2, gamma_, z_1, z_2):
        res_1 = mp.gamma(gamma_) * mp.gamma(beta_1 - alpha)
        res_1 /= mp.gamma(gamma_ - alpha) * mp.gamma(beta_1)
        res_1 *= (-z_1)**(-alpha)
        res_1 *= mp.appellf1(alpha, alpha - gamma_ + 1,
                             beta_2, alpha - beta_1 + 1,
                             1./z_1, z_2/z_1)
        res_2 = mp.gamma(gamma_) / mp.gamma(alpha)
        res_2 *= (-z_1)**(-beta_1)
        res_2 *= mp.nsum(lambda k:
                         mp.gamma(alpha + k - beta_1) * mp.rf(beta_2, k) /
                         (mp.gamma(gamma_ + k - beta_1) * mp.fac(k)) *
                         mp.hyp2f1(beta_1,
                                   beta_1 - gamma_ - k + 1,
                                   beta_1 - alpha - k + 1,
                                   1./z_1) *
                         z_2**k, [0, mp.inf])

        return res_1 + res_2

    mp.appellf1(alpha, beta, 1., gamma_, z_1, z_2)
    F1_z1_inf(alpha, beta, 1., gamma_, z_1, z_2)
