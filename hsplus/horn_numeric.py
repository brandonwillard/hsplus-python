# -*- coding: utf-8 -*-
import numpy as np
from mpmath import mp, fp

mp_ctx = mp


def phi1_T1_x(m, a_, b_, g_, x_, y_):
    r""" Series terms for :math:`Phi_1(a, b, g; x, y)`.

    This series is in powers of :math:`x`.

    TODO: If |y| >> 1, we could try:

      res *= exp(y_) * hyp1f1(g_ - a_, g_ + n, -y_)

    and move the `exp(x_)` outside the sum.  In ratios we
    might even be able to do this to both terms and divide out
    `exp(x_)` entirely.
    """
    res = mp_ctx.rf(a_, m) / mp_ctx.rf(g_, m)
    res *= mp_ctx.power(x_, m) / mp_ctx.fac(m)
    res *= mp_ctx.hyp2f1(b_, a_ + m, g_ + m, y_)
    return res


def phi1_T2_x(m, a_, b_, g_, x_, y_):
    r""" Scaled series terms for :math:`\exp(-x) Phi_1(a, b, g; x, y)`.

    This series is in powers of :math:`-x`.
    """
    res = mp_ctx.rf(g_ - a_, m) / mp_ctx.rf(g_, m)
    res *= mp_ctx.power(-x_, m) / mp_ctx.fac(m)
    res *= mp_ctx.hyp2f1(b_, a_, g_ + m, y_)
    return res


def phi1_T3_y(n, a_, b_, g_, x_, y_):
    r""" Series terms for :math:`Phi_1(a, b, g; x, y)`.

    This series is in powers of :math:`y`.

    TODO: It might be possible to divide out the
    `mp_ctx.exp(x_)` terms in ratios.
    """
    r"""
    TODO: If x >> 1, we could try:

      res *= mp_ctx.exp(x_) * mp_ctx.hyp1f1(g_ - a_, g_ + n, -x_)

    and move the `mp_ctx.exp(x_)` outside the sum.  In ratios we
    might even be able to do this to both terms and divide out
    `mp_ctx.exp(x_)` entirely.
    """
    res = mp_ctx.rf(a_, n) * mp_ctx.rf(b_, n) / mp_ctx.rf(g_, n)
    res *= mp_ctx.power(y_, n) / mp_ctx.fac(n)
    #if mp_ctx.abs(x_) >= 1.:
    #    res *= mp_ctx.hyp1f1(g_ - a_, g_ + n, -x_)
    #    res *= mp_ctx.exp(x_)
    #else:
    #    res *= mp_ctx.hyp1f1(a_ + n, g_ + n, x_)
    res *= mp_ctx.hyp1f1(a_ + n, g_ + n, x_)
    return res


def phi1_T4_y(n, a_, b_, g_, x_, y_):
    r""" Scaled series terms for :math:`\exp(-x) Phi_1(a, b, g; x, y)`.

    This series is in powers of :math:`y`.
    """
    res = mp_ctx.rf(a_, n) * mp_ctx.rf(b_, n) / mp_ctx.rf(g_, n)
    res *= mp_ctx.power(y_, n) / mp_ctx.fac(n)
    res *= mp_ctx.hyp1f1(g_ - a_, g_ + n, -x_)
    return res


def horn_phi1_gordy_single(a, b, g, x, y, keep_exp_const=True, **kwargs):
    r""" Infinite precision computation of the Horn :math:`Phi_1` function.
    Uses the approach of [Gordy]_.

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
        ::math::`0 < \alpha < \gamma`.
    b: float
        The ::math::`\beta` parameter.
    g: float
        The ::math::`\gamma` parameter.
    x: float
        The ::math::`x` parameter.
    y: float
        The ::math::`y` parameter.  This value must satisfy
        ::math::`y < 1`.
    keep_exp_const: bool (optional)
        Include constant multipliers terms like ::math::`\exp(x)` and
        ::math::`(1-y)^{-\beta}`.  Set this to `False` when computing
        ratios with equal ``b``, ``x`` and ``y`` parameters.

    Returns
    =======
    ndarray (mpmath.mpf)
      The real value of the Horn Phi1 function at the given points.

    .. [Gordy] Gordy, Michael B. “A Generalization of Generalized Beta
        Distributions,” 1998.
    """

    if not (0 <= a and a < g):
        raise ValueError("Parameter a must be 0 < a < g")

    if y >= 1:
        # TODO: We could use
        #   hyp2f1(a, b, g, 1) =
        #     gamma(g) gamma(g-a-b) /gamma(g-a) / gamma(g-b)
        # for g-a-b > 0, so
        #   hyp2f1(b, a+m, g+m, 1) =
        #     gamma(g+m) gamma(g-b-a) /gamma(g+m-b) / gamma(g-a)
        # in phi1_T1_x and phi1_T2_x.
        # The whole result is
        # res = mp_ctx.gamma(g) * mp_ctx.gamma(g-a-b)
        # res /= mp_ctx.gamma(g-a) * mp_ctx.gamma(g-b)
        # res *= mp_ctx.hyp1f1(a, g-b, x)
        raise ValueError("Parameter y must be 0 <= y < 1")

    if (0 <= y and y < 1):
        #if mp_ctx.chop(y) == 0:
        #    res = mp_ctx.hyp2f1(a, 1, g, x)
        phi_args = (a, b, g, x, y)
        if x < 0:
            if x > -1:
                # -1 < x < 0

                # Only the second properly diverges at y ~ 1.
                if y < 0.9:
                    res = mp_ctx.nsum(lambda n: phi1_T4_y(n, *phi_args),
                                      [0, mp_ctx.inf])
                else:
                    res = mp_ctx.nsum(lambda n: phi1_T2_x(n, *phi_args),
                                      [0, mp_ctx.inf])
                if keep_exp_const:
                    res *= mp_ctx.exp(x)
            else:
                # x <= -1
                if y < 0.9:
                    res = mp_ctx.nsum(lambda n: phi1_T4_y(n, *phi_args),
                                      [0, mp_ctx.inf])
                else:
                    # res = mp_ctx.nsum(lambda n: phi1_T1_x(n, *phi_args),
                    #                   [0, mp_ctx.inf])

                    res = mp_ctx.nsum(lambda n: phi1_T2_x(n, *phi_args),
                                      [0, mp_ctx.inf])
                if keep_exp_const:
                    res *= mp_ctx.exp(x)
        else:
            # x > 0
            if x >= 1:
                if y < 0.5:
                    res = mp_ctx.nsum(lambda n: phi1_T4_y(n, *phi_args),
                                      [0, mp_ctx.inf])
                else:
                    res = mp_ctx.nsum(lambda n: phi1_T2_x(n, *phi_args),
                                      [0, mp_ctx.inf])
                if keep_exp_const:
                    res *= mp_ctx.exp(x)
            else:
                # 0 <= x <= 1
                if y < 0.5:
                    if x < 0.5:
                        res = mp_ctx.nsum(lambda n: phi1_T1_x(n, *phi_args),
                                          [0, mp_ctx.inf])
                    else:
                        res = mp_ctx.nsum(lambda n: phi1_T3_y(n, *phi_args),
                                          [0, mp_ctx.inf])
                else:
                    res = mp_ctx.nsum(lambda n: phi1_T1_x(n, *phi_args),
                                      [0, mp_ctx.inf])
    elif mp.isfinite(y):
        res = horn_phi1_gordy_single(g - a, b, g, -x, y / (y - 1.),
                                     keep_exp_const=keep_exp_const)
        if keep_exp_const:
            res *= mp_ctx.power(1 - y, -b) * mp_ctx.exp(x)
    else:
        raise ValueError("Unhandled y value: {}". format(y))

    return res


def horn_phi1_quad_single(a, b, g, x, y, **kwargs):
    r""" Finite precision quadrature computation of Humbert :math:`\Phi_1`.

    See Also
    --------
    horn_phi1_gordy_single: Series computation of Humbert :math:`\Phi_1`.
    """
    if not (0 <= a and a < g):
        # FIXME: Too restrictive: c-a < 0 can be non-integer.
        raise ValueError("Parameter a must be 0 < a < g")

    if y >= 1:
        raise ValueError("Parameter y must be 0 <= y < 1")

    def phi_1_integrand_num(t):
        res_ = t**(a-1.) * (1.-t)**(g-a-1.)
        res_ *= fp.exp(y * t) / (1.-x*t)**b
        return res_

    try:
        res = fp.gamma(g)/fp.gamma(a)/fp.gamma(g-a)
        res *= fp.quad(phi_1_integrand_num, [0, 1])

        return res
    except:
        # TODO: Could check |y| >> 1 and guess at the result being inf.
        return fp.nan


horn_phi1 = np.vectorize(horn_phi1_gordy_single)
horn_phi1_quad = np.vectorize(horn_phi1_quad_single)
