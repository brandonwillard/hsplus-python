from itertools import izip_longest

import numpy as np
import mpmath as mp

# TODO: mpmath defaults?
mp.dps = 7


def horn_phi1_single(a, b, g, x, y):
    r""" Evaluate the Horn Phi1 function.  Uses the approach of Gordy (1998).

    The general expression in `mpmath` is

    ```
      nsum(lambda m,n: rf(a,m+n)*rf(b,n)/rf(g,m+n)*\
        x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
    ```

    Parameters
    ==========
    a: float
        This value must satisfy `0 < a < g`.
    b: float
    g: float
    x: float
    y: float
        This value must satisfy `0 <= y < 1`.

    Returns
    =======
      The real value of the Horn Phi1 function at the given points.
    """

    if 0 <= y and y < 1:
        if x < 0:
            # (T2)
            res = mp.exp(x) * mp.nsum(lambda m: mp.rf(g - a, m) /
                                      mp.rf(g, m) * (-x)**m / mp.fac(m) *
                                      mp.hyp2f1(b, a, g + m, y), [0, mp.inf])
        else:
            # (T1)
            res = mp.nsum(lambda m: mp.rf(a, m) / mp.rf(g, m) *
                          x**m / mp.fac(m) *
                          mp.hyp2f1(b, a + m, g + m, y), [0, mp.inf])
    elif y < 0:
        # (T4)
        res = mp.exp(x) * (1 - y)**(-b) * horn_phi1(g - a, b,
                                                    g, -x, y / (y - 1))
    else:
        # not defined for y >= 1
        res = np.nan

    return float(res)


horn_phi1 = np.vectorize(horn_phi1_single)


def m_hs(y, sigma, tau, deriv_ord=None):
    r""" Exact evaluation of the marginal posterior of the HS prior via
    special functions.

    Parameters
    ==========
    y: float
        A single observation.
    sigma: float
        Observation variance.
    tau: float
        Prior variance scale factor.
    deriv_ord: int or None
        Order of the derivative.

    Returns
    =======
      Numeric value of `m(y; sigma, tau)`.
    """
    s_p = np.square(y) / (2. * sigma**2)
    if deriv_ord is None:
        C = 1./(tau * np.sqrt(2. * np.pi * sigma**2))
        res = C * np.exp(-s_p) * horn_phi1(0.5, 1., 1.5, s_p, 1.-1./tau**2)
    elif deriv_ord == 1:
        C = 2./(3. * tau * np.sqrt(2. * np.pi * sigma**2))
        res = C * np.exp(-s_p) * horn_phi1(0.5, 1., 2.5, s_p, 1.-1./tau**2)
    else:
        return None

    return res


def m_hsp(y, sigma, tau, deriv_ord=None):
    r""" Exact evaluation of the marginal posterior for
    the HS+ prior via special functions.

    Parameters
    ==========
    y: float
        A single observation.
    sigma: float
        Observation variance.
    tau: float
        Prior variance scale factor.
    deriv_ord: int or None
        Order of the derivative.

    Returns
    =======
    Numeric value of `m(y; sigma, tau)`.
    """
    # TODO: What form?
    pass


def E_k_Z(Z, n=1., tauv=1., sigv=1., hsp=False):
    r"""
    TODO:

    Parameters
    ==========
    TODO

    Results
    =======
    TODO
    """
    if not hsp:
        s_p = Z/(2.0*sigv**2)
        res = (1./mp.rf(1.5, n) *
               horn_phi1(0.5, 1., 1.5+n, s_p, 1.-1./tauv**2) /
               horn_phi1(0.5, 1., 1.5, s_p, 1.-1./tauv**2))

        assert res >= 0 and res <= 1.0

    return float(res)


def m_p_Z(Z, p, tauv=1, sigv=1, hsp=False):
    r""" Computes
    .. math:

        \int_0^1 z^{p/2} e^{-z/2} p(z) dz

    with special functions.

    a=1/2, b=1/2, s=0 s_p=Z/(2*sigv**2), a_p=1

    Parameters
    ==========
    TODO

    Results
    =======
    TODO
    """
    if not hsp:
        s_p = Z/(2.0*sigv**2)
        C = (2.0 * mp.pi * sigv**2)**(-p/2.0)

        res = C * mp.exp(-s_p) * 2.0/mp.pi *\
            horn_phi1(0.5, 1.0, 1.5, s_p, 1-1.0/tauv**2) /\
            horn_phi1(0.5, 1.0, 1.0, 0.0, 1-1.0/tauv**2)

        assert res >= 0 and res <= 1.0

    else:
        return None

    return float(res)


def m_p_Z_num(z, p, tauv=1, hsp=False):
    r""" Numerically integrate
    .. math:

        \int_0^1 z^{p/2} e^{-z/2} p(z) dz


    Parameters
    ==========
    hsp: boolean
        If True, compute using the HS+ prior, otherwise HS.

    Returns
    =======
        TODO
    """
    from .symbolic_funcs import symbol_obj as sym

    k_pow = mp.fraction(p, 2)
    if hsp:
        mid_singularity = mp.fraction(1, tauv**4+1)
        res = mp.quad(lambda k: k**k_pow * mp.exp(z*k/2) *
                      sym.lam_hsp_prior_kap(k, tauv),
                      [0, mid_singularity, 1])
    else:
        # TODO: Also check that this singularity is in [0,1]
        tau_4 = mp.mpmathify(tauv)**4
        if tauv >= 1 or 2 < tau_4:
            res = mp.quad(lambda k: k**k_pow * mp.exp(z*k/2) *
                          sym.lam_hs_prior_kap(k, tauv), [0, 1])
        else:
            mid_singularity = mp.fraction(1, tau_4-1)
            res = mp.quad(lambda k: k**k_pow * mp.exp(z*k/2) *
                          sym.lam_hs_prior_kap(k, tauv),
                          [0, mid_singularity, 1])

    return float(res)


def hs_mse_z(z, p):
    r"""
    Computes HS MSE conditional on Z=z.

    Parameters
    ==========
    z: float
        The sum-of-squared observations upon which we condition.
    p: float or int
        The degrees of freedom

    Returns
    =======
    TODO
    """

    part_1 = z * m_p_Z(z, p + 4)/m_p_Z(z, p)
    E_part = E_k_Z(z)
    return part_1 - p * E_part - z * E_part**2 / 2.


def hs_mse_mc(beta2, p, N=1000):
    r""" Samples observations sum-of-squares and computes the mean
    MSE for HS.

    Parameters
    ==========
    beta2: float
        The magnitude of the true signal squared.
    p: float or int
        Degrees of freedom/number of observations.
    N: int
        Number of samples used to approximate the posterior expected value.

    Returns
    =======
    TODO
    """
    Z = np.random.noncentral_chisquare(p, beta2, N)
    hs_MSE = p + 2 * np.mean([hs_mse_z(z, p) for z in Z])
    return hs_MSE


def E_beta_y(y, sigma, tau):
    r""" Tweedie's formula for the HS marginal posterior.

    Parameters
    ==========
    y: np.array
        TODO
    sigma: float
        TODO
    tau: float
        TODO

    Returns
    =======
    TODO
    """
    return y + m_hs(y, sigma, tau, deriv_ord=1)/m_hs(y, sigma, tau)


def Var_beta_y(y, sigma, tau):
    r""" Tweedie's formula for the HS marginal posterior.

    Parameters
    ==========
    y: np.array
        TODO
    sigma: float
        TODO
    tau: float
        TODO

    Returns
    =======
    TODO
    """
    return 1. - m_hs(y, sigma, tau, deriv_ord=2)/m_hs(y, sigma, tau) +\
        (m_hs(y, sigma, tau, deriv_rd=1)/m_hs(y, sigma, tau))**2


def m_hs_num_single(y, tau, sigma):
    r""" HS marginal integral evaluated by numeric integration.

    Given
    .. math:

        (y \mid \sigma^2, \tau^2, \lambda_i^2) \sim N(0, \sigma^2 (1 + \tau^2 \lambda_i^2))

    this function computes

    .. math:

        \prod_i \int_0^\infty p(y \mid \sigma^2, \tau^2, \lambda_i^2) p(\lambda_i) d\lambda_i


    Parameters
    ==========
    y: float
        Observation.
    tau: float
        Positive prior variance term.
    sigma: float
        Positive observation variance term.

    Returns
    =======
    Value of numeric marginal integral.
    """

    C = 1./mp.sqrt(2. * mp.pi)

    def quad_expr(lam):
        var_res = sigma**2 * (1. + tau**2 * lam**2)
        res = mp.exp(-y**2 / (2. * var_res)) /\
            ((1. + lam**2) * mp.sqrt(var_res))
        return res

    return float(C * mp.quad(lambda lam: quad_expr(lam), [0, mp.inf]))


m_hs_num = np.vectorize(m_hs_num_single)
