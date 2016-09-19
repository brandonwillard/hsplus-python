import numpy as np
import mpmath as mp

# TODO: mpmath defaults?
mp.dps = 7


def phi1_T1_x(m, a_, b_, g_, x_, y_):
    res = mp.rf(a_, m) / mp.rf(g_, m)
    res *= mp.power(x_, m) / mp.fac(m)
    res *= mp.hyp2f1(b_, a_ + m, g_ + m, y_)
    return res


def phi1_T2_x(m, a_, b_, g_, x_, y_):
    res = mp.rf(g_ - a_, m) / mp.rf(g_, m)
    res *= mp.power(-x_, m) / mp.fac(m)
    res *= mp.hyp2f1(b_, a_, g_ + m, y_)
    res *= mp.exp(x_)
    return res


def phi1_T3_x(n, a_, b_, g_, x_, y_):
    res = mp.rf(a_, n) * mp.rf(b_, n) / mp.rf(g_, n)
    res *= mp.power(y_, n) / mp.fac(n)
    res *= mp.hyp1f1(a_ + n, g_ + n, x_)
    return res


def phi1_T4_x(n, a_, b_, g_, x_, y_):
    res = mp.rf(a_, n) * mp.rf(b_, n) / mp.rf(g_, n)
    res *= mp.power(y_, n) / mp.fac(n)
    res *= mp.hyp1f1(g_ - a_, g_ + n, -x_)
    res *= mp.exp(x_)
    return res


def horn_phi1_single(a, b, g, x, y):
    r""" Evaluate the Horn Phi1 function.  Uses the approach of Gordy (1998).

    .. math:

        \Phi_1(\alpha, \beta; \gamma; x, y) =
        \sum_{m=0}^\infty \sum_{n=0}^\infty
        \frac{(\alpha)_{m+n} (\beta)_n}{(\gamma)_{m+n} m! n!}
        y^n x^m

    The general expression in `mpmath` is

    ```
      nsum(lambda m,n: rf(a,m+n)*rf(b,n)/rf(g,m+n)*\
        x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
    ```

    Parameters
    ==========
    a: float
        The ::math::`\alpha` parameter.  This value must satisfy
        ::math::`0 < a < g`.
    b: float
        The ::math::`\beta` parameter.
    g: float
        The ::math::`\gamma` parameter.
    x: float
        The ::math::`x` parameter.
    y: float
        The ::math::`y` parameter.  This value must satisfy
        ::math::`0 \leq y < 1`.

    Returns
    =======
    ndarray of mpmath.mpf
      The real value of the Horn Phi1 function at the given points.
    """

    if not (0 < a and a < g):
        raise ValueError("Parameter a must be 0 < a < g")

    if y >= 1:
        raise ValueError("Parameter y must be 0 <= y < 1")

    if (0 <= y and y < 1):
        #if mp.chop(y) == 0:
        #    res = mp.hyp2f1(a, 1, g, x)
        phi_args = (a, b, g, x, y)
        if x < 0:
            if x > -1:
                res = mp.nsum(lambda n: phi1_T4_x(n, *phi_args), [0, mp.inf])
            else:
                res = mp.nsum(lambda n: phi1_T2_x(n, *phi_args), [0, mp.inf])
        else:
            if x > 1:
                res = mp.nsum(lambda n: phi1_T3_x(n, *phi_args), [0, mp.inf])
            else:
                res = mp.nsum(lambda n: phi1_T1_x(n, *phi_args), [0, mp.inf])
    elif mp.isfinite(y):
        res = mp.exp(x) * mp.power(1 - y, -b)
        res *= horn_phi1(g - a, b, g, -x, y / (y - 1.))
    else:
        raise ValueError("Unhandled y value: {}". format(y))

    return res


horn_phi1 = np.vectorize(horn_phi1_single)


def m_hib(y, sigma, tau=1., a=0.5, b=0.5, s=0):
    r""" Exact evaluation of the marginal posterior for
    the hypergeometric inverted-beta model.

    In its most general form, we have for the hypergeometric
    inverted-beta model given by

    .. math:

        p(y_i, \kappa_i) \propto \kappa_i^{a^\prime - 1} (1-\kappa_i)^{b-1}
        \left(1/\tau^2 + (1 - 1/\tau^2) \kappa_i\right)^{-1}
        e^{-\kappa_i s^\prime}
        \;.

    where ::math::`s^\prime = s + y_i^2 / (2\sigma^2)` and
    ::math::`a^\prime = a + 1/2`.

    The marginal posterior is

    .. math:
        m(y_i; \sigma, \tau) = \frac{1}{\sqrt(2 \pi \sigma^2}
        \exp\left(-\frac{y_i^2}{2 \sigma^2}\right)
        \frac{\operatorname{B}(a^\prime, b)}{\operatorname{B}(a,b)}
        \frac{\Phi_1(b, 1, a^\prime + b, s^\prime, 1 - 1/\tau^2)}{
            \Phi_1(b, 1, a + b, s, 1 - 1/\tau^2)}

    The Horseshoe prior has ::math::`a = b = 1/2, s = 0` and
    ::math::`\tau = 1`.

    Parameters
    ==========
    y: float
        A single observation.
    sigma: float
        Observation variance.
    tau: float
        Prior variance scale factor.
    a: float
        Hypergeometric inverted-beta model parameter
    b: float
        Hypergeometric inverted-beta model parameter
    s: float
        Hypergeometric inverted-beta model parameter

    Returns
    =======
    ndarray of mpmath.mpf
      Numeric value of `m(y; sigma, tau)`.
    """
    y_2_sig = 0.5 * np.square(y / sigma)
    s_p = s + y_2_sig
    a_p = a + 0.5
    C = 1. / np.sqrt(2. * np.pi) / sigma
    res = C * np.exp(-y_2_sig)
    res *= mp.beta(a_p, b) / mp.beta(a, b)

    if tau > 0:
        tau_term = 1. - tau**(-2)
    else:
        tau_term = mp.ninf

    res *= horn_phi1(b, 1., a_p + b, s_p, tau_term)
    res /= horn_phi1(b, 1., a + b, s, tau_term)

    return res


def m_hs(y, sigma, tau=1.):
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
    ndarray of mpmath.mpf
      Numeric value of `m(y; sigma, tau)`.
    """
    return m_hib(y, sigma, tau, 0.5, 0.5, 0)


def E_kappa(y, sigma, tau=1., a=0.5, b=0.5, s=0, n=1):
    r""" Moments of the hypergeometric inverted-beta model
    in the ::math::`\kappa` parameterization.

    In its most general form, we have

    .. math:

        E(\kappa^n \mid y, \sigma, \tau) &=
        \frac{(a^\prime)_n}{(a^\prime + b)_n}
        \frac{\Phi_1(b, 1, a^\prime + b + n, s^\prime, 1 - 1/\tau^2)}{
            \Phi_1(b, 1, a^\prime + b, s^\prime, 1 - 1/\tau^2)}
            \;,

    where ::math::`s^\prime = s + y_i^2 / (2\sigma^2)` and
    ::math::`a^\prime = a + 1/2` for the hypergeometric inverted-beta
    given by

    .. math:

        p(y_i, \kappa_i) \propto \kappa_i^{a^\prime - 1} (1-\kappa_i)^{b-1}
        \left(1/\tau^2 + (1 - 1/\tau^2) \kappa_i\right)^{-1}
        e^{-\kappa_i s^\prime}
        \;.

    Parameters
    ==========
    y: float
        A single observation.
    sigma: float
        Observation variance.
    tau: float
        Prior variance scale factor.
    a: float
        Hypergeometric inverted-beta model parameter
    b: float
        Hypergeometric inverted-beta model parameter
    s: float
        Hypergeometric inverted-beta model parameter
    n: int
        Order of the moment.

    Results
    =======
    ndarray of mpmath.mpf
    """
    s_p = s + 0.5 * np.square(y / sigma)
    a_p = a + 0.5
    res = mp.rf(a_p, n) / mp.rf(a_p + b, n)

    if tau > 0:
        tau_term = 1. - tau**(-2)
    else:
        tau_term = mp.ninf

    res_phis = horn_phi1(b, 1., a_p + b + n, s_p, tau_term)
    res_phis /= horn_phi1(b, 1., a_p + b, s_p, tau_term)
    res *= res_phis

    return res


def E_beta(y, sigma, tau=1., a=0.5, b=0.5, s=0, n=1):
    r""" Moments of the hypergeometric inverted-beta model
    in the ::math::`\beta` parameterization.

    In its most general form, we have

    .. math:

        E(\beta^n \mid y, \sigma, \tau) &=
        \left( 1 - \frac{a^\prime}{a^\prime + b}
        \frac{\Phi_1(b, 1, a^\prime + b + n, s^\prime, 1 - 1/\tau^2)}{
            \Phi_1(b, 1, a^\prime + b, s^\prime, 1 - 1/\tau^2)}
            \right) y
            \;,

    where ::math::`s^\prime = s + y_i^2 / (2\sigma^2)` and
    ::math::`a^\prime = a + 1/2` for the hypergeometric inverted-beta
    given by

    .. math:

        p(y_i, \kappa_i) \propto \kappa_i^{a^\prime - 1} (1-\kappa_i)^{b-1}
        \left(1/\tau^2 + (1 - 1/\tau^2) \kappa_i\right)^{-1}
        e^{-\kappa_i s^\prime}
        \;.

    Parameters
    ==========
    y: float
        A single observation.
    sigma: float
        Observation variance.
    tau: float
        Prior variance scale factor.
    a: float
        Hypergeometric inverted-beta model parameter
    b: float
        Hypergeometric inverted-beta model parameter
    s: float
        Hypergeometric inverted-beta model parameter
    n: int
        Order of the moment.

    Results
    =======
    ndarray of mpmath.mpf
    """
    s_p = s + 0.5 * np.square(y / sigma)
    a_p = a + 0.5
    res = a_p / (a_p + b)

    if tau > 0:
        tau_term = 1. - tau**(-2)
    else:
        tau_term = mp.ninf

    res *= horn_phi1(b, 1., a_p + b + n, s_p, tau_term)
    res /= horn_phi1(b, 1., a_p + b, s_p, tau_term)
    res *= y
    res = y - res

    return res


def sure_hib(y, sigma, tau=1., a=0.5, b=0.5, s=0, d=1.):
    r""" Compute the SURE value for the HIB prior model.

    Parameters
    ==========
    y: float
        A single observation.
    sigma: float
        Observation variance.
    tau: float
        Prior variance scale factor.
    a: float
        Hypergeometric inverted-beta model parameter
    b: float
        Hypergeometric inverted-beta model parameter
    s: float
        Hypergeometric inverted-beta model parameter
    d: float
        Optional observation scaling parameter.

    Returns
    =======
    TODO
    """
    res = 2 * sigma**2
    E_1 = E_kappa(y, sigma, tau, a, b, s, n=1)
    y_d_2 = np.square(y * d)
    res -= y_d_2 * np.square(E_1)
    E_2 = E_kappa(y, sigma, tau, a, b, s, n=2)
    res2 = -sigma**2 * E_1 + y_d_2 * E_2
    res += 2 * res2
    return res


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
