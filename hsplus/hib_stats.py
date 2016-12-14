# -*- coding: utf-8 -*-
import numpy as np
from mpmath import mp, fp

from .horn_numeric import horn_phi1, mp_ctx


def m_hib_single(y, sigma=1., tau=1., a=0.5, b=0.5, s=0.,
                 horn_phi1_fn=horn_phi1):
    r""" Computation of the marginal posterior for the hypergeometric
    inverted-beta model.

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
    res *= mp_ctx.beta(a_p, b) / mp_ctx.beta(a, b)

    if tau > 0:
        tau_term = 1. - tau**(-2)
    else:
        tau_term = mp.ninf

    # TODO, FIXME: Replace with direct computation of this ratio.
    res *= horn_phi1_fn(b, 1., a_p + b, s_p, tau_term, keep_exp_const=False)
    res /= horn_phi1_fn(b, 1., a + b, s, tau_term, keep_exp_const=False)

    return res


m_hib = np.vectorize(m_hib_single)


def m_hs(y, sigma=1., tau=1.):
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

    Returns
    =======
    ndarray of mpmath.mpf
      Numeric value of `m(y; sigma, tau)`.
    """
    return m_hib(y, sigma, tau, 0.5, 0.5, 0)


def E_kappa(y, sigma=1., tau=1., a=0.5, b=0.5, s=0., n=1.,
            horn_phi1_fn=horn_phi1):
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
    res = mp_ctx.rf(a_p, n) / mp_ctx.rf(a_p + b, n)

    if np.iterable(tau):
        # TODO: mp_ctx.mpf(t_) ?
        tau_term = np.fromiter((1. - t_**(-2) if t_ > 0 else mp.ninf
                                for t_ in tau), dtype=np.float)
    else:
        if tau > 0:
            # TODO: mp_ctx.mpf(t_) ?
            tau_term = 1. - tau**(-2)
        else:
            tau_term = mp.ninf

    # TODO: Replace with direct computation of this ratio.
    res_num = horn_phi1_fn(b, 1., a_p + b + n, s_p, tau_term,
                           keep_exp_const=False)
    res_denom = horn_phi1_fn(b, 1., a_p + b, s_p, tau_term,
                             keep_exp_const=False)

    # XXX FIXME: A hack, but not completely without sense.  The numerator
    # should diverge slower than the denominator.  We'll keep the
    # division so that a warning is emitted, but this needs to be
    # handled properly.
    res_ratio = res_num/res_denom
    res_ratio[np.isinf(res_num)] = 1

    res *= res_ratio

    return res


def E_beta(y, sigma=1., tau=1., a=0.5, b=0.5, s=0.,
           horn_phi1_fn=horn_phi1):
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

    Results
    =======
    ndarray of mpmath.mpf
    """
    res = E_kappa(y, sigma, tau, a, b, s, n=1,
                  horn_phi1_fn=horn_phi1)

    res *= y
    res = y - res

    return res


def SURE_hib(y, sigma=1., tau=1., a=0.5, b=0.5, s=0., d=1.):
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
    y_d_2 = (y * d)**2
    res = res - y_d_2 * E_1**2
    E_2 = E_kappa(y, sigma, tau, a, b, s, n=2)
    res2 = -sigma**2 * E_1 + y_d_2 * E_2
    res = res + 2 * res2
    return res


def DIC_hib(y, sigma=1., tau=1., a=0.5, b=0.5, s=0., d=1.):
    r""" Computes DIC estimate for the hypergeometric inverted-beta model.

    .. math::
        \text{DIC} = \sum_{i=1}^N \left\{
        2 \left(1 - E\left[\kappa_i \mid y_i\right] \right) +
        \frac{y_i^2}{\sigma^2} \left( 2 E\left[\kappa_i^2 \mid y_i\right] -
            {E\left[\kappa_i \mid y_i\right]}^2 \right)
        \right\}


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
    ndarray (float)

    """
    E_1 = E_kappa(y, sigma, tau, a, b, s, n=1)
    E_2 = E_kappa(y, sigma, tau, a, b, s, n=2)
    res = 2.*(1. - E_1) + (y * d / sigma)**2 * (2. * E_2 - E_1**2)
    return res
