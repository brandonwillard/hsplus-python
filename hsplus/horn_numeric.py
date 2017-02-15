# -*- coding: utf-8 -*-
import numpy as np
from mpmath import mp, fp

mp_ctx = mp


def phi1_T1_x(m, a_, b_, g_, x_, y_):
    r""" Series terms for an expansion of ::math::`Phi_1`
    in powers of ::math::`x`.
    """
    res = mp_ctx.rf(a_, m) / mp_ctx.rf(g_, m)
    res *= mp_ctx.power(x_, m) / mp_ctx.fac(m)
    res *= mp_ctx.hyp2f1(b_, a_ + m, g_ + m, y_)
    return res


def phi1_T1_x_gen(a_, b_, g_, x_, y_, pow_terms=True):
    r""" Generator of series terms for an expansion of ::math::`Phi_1`
    in powers of ::math::`x`.

    """
    coef = 1
    x_pow = 1
    F_term = mp_ctx.hyp2f1(b_, a_, g_, y_)
    m = 0
    while True:
        res = coef * x_pow * F_term
        yield res
        m += 1
        coef *= (a_ + m - 1) / (g_ + m - 1) / m
        if pow_terms:
            x_pow *= x_
        F_term = mp_ctx.hyp2f1(b_, a_ + m, g_ + m, y_)


def phi1_T2_x(m, a_, b_, g_, x_, y_):
    r""" Regularized series terms for an expansion of ::math::`Phi_1`
    in powers of ::math::`-x`.
    FYI: 'regularized' means that the ::math::`\exp(-x)` term is excluded.
    """
    res = mp_ctx.rf(g_ - a_, m) / mp_ctx.rf(g_, m)
    res *= mp_ctx.power(-x_, m) / mp_ctx.fac(m)
    res *= mp_ctx.hyp2f1(b_, a_, g_ + m, y_)
    return res


def phi1_T2_x_gen(a_, b_, g_, x_, y_, pow_terms=True):
    r""" Generator of regularized series terms for an expansion of
    ::math::`Phi_1` in powers of ::math::`-x`.
    FYI: 'regularized' means that the ::math::`\exp(-x)` term is excluded.
    """
    coef = 1.
    x_pow = 1.
    F_term = mp_ctx.hyp2f1(b_, a_, g_, y_)
    m = 0
    while True:
        res = coef * x_pow * F_term
        yield res
        m += 1
        coef *= (g_ - a_ + m - 1) / (g_ + m - 1) / m
        if pow_terms:
            x_pow *= -x_
        F_term = mp_ctx.hyp2f1(b_, a_, g_ + m, y_)


def phi1_T3_y(n, a_, b_, g_, x_, y_):
    r""" Series terms for an expansion of ::math::`Phi_1`
    in powers of ::math::`y`.
    """
    try:
        res = mp_ctx.rf(a_, n) * mp_ctx.rf(b_, n) / mp_ctx.rf(g_, n)
        res *= mp_ctx.power(y_, n) / mp_ctx.fac(n)
    except OverflowError:
        res = mp_ctx.inf

    try:
        res *= mp_ctx.hyp1f1(a_ + n, g_ + n, x_)
    except OverflowError:
        # XXX, FIXME: This is hack.  When using finite precision,
        # `math.exp(x)` will overflow for large `x`, instead of returing
        # `float('inf')`.  That's why we do the following:
        if x_ > 0:
            res = mp_ctx.inf
        else:
            raise
    return res


def phi1_T3_y_gen(a_, b_, g_, x_, y_, pow_terms=True):
    r""" Generator of series terms for an expansion of ::math::`Phi_1`
    in powers of ::math::`y`.
    """
    coef = 1
    y_pow = 1
    F_term = mp_ctx.hyp1f1(a_, g_, x_)
    n = 0
    while True:
        res = coef * y_pow * F_term
        yield res
        n += 1
        coef *= (a_ + n - 1) * (b_ + n - 1) / (g_ + n - 1) / n
        if pow_terms:
            y_pow *= y_
        F_term = mp_ctx.hyp1f1(a_ + n, g_ + n, x_)


def phi1_T4_y(n, a_, b_, g_, x_, y_):
    r""" Regularized series terms for an expansion of ::math::`Phi_1` in
    powers of ::math::`y`.
    FYI: 'regularized' means that the ::math::`\exp(-x)` term is excluded.
    """
    try:
        res = mp_ctx.rf(a_, n) * mp_ctx.rf(b_, n) / mp_ctx.rf(g_, n)
        res *= mp_ctx.power(y_, n) / mp_ctx.fac(n)
    except OverflowError:
        res = mp_ctx.inf

    try:
        res *= mp_ctx.hyp1f1(g_ - a_, g_ + n, -x_)
    except OverflowError:
        # XXX, FIXME: This is hack.  When using finite precision,
        # `math.exp(x)` will overflow for large `x`, instead of returing
        # `float('inf')`.  That's why we do the following:
        if -x_ > 0:
            res = mp_ctx.inf
        else:
            raise
    return res


def phi1_T4_y_gen(a_, b_, g_, x_, y_, pow_terms=True):
    r""" Generator of regularized series terms for an expansion of
    ::math::`Phi_1` in powers of ::math::`y`.

    FYI: 'regularized' means that the ::math::`\exp(-x)` term is excluded.
    """
    coef = 1
    # coef *= mp_ctx.exp(y_)
    y_pow = 1
    F_term = mp_ctx.hyp1f1(g_ - a_, g_, -x_)
    n = 0
    while True:
        res = coef * y_pow * F_term
        yield res
        n += 1
        coef *= (a_ + n - 1) * (b_ + n - 1) / (g_ + n - 1) / n
        if pow_terms:
            y_pow *= y_
        F_term = mp_ctx.hyp1f1(g_ - a_, g_ + n, -x_)


def horn_phi1_gordy_single(a, b, g, x, y,
                           keep_exp_const=True,
                           **kwargs):
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

    if y > 1:
        raise ValueError("Parameter y must be 0 <= y < 1")
    elif y == 1:
        return mp_ctx.inf

    if (0 <= y and y < 1):

        phi_args = (a, b, g, x, y)
        if x < 0:
            if x > -1:
                # -1 < x < 0
                series_gen = phi1_T2_x_gen(*phi_args)
                res = mp_ctx.nsum(lambda n: next(series_gen),
                                  [0, mp_ctx.inf])
                if keep_exp_const and not mp_ctx.isinf(res):
                    res *= mp_ctx.exp(x)
            else:
                # x <= -1
                if y == 0 and mp_ctx is fp:
                    # FIXME: Yet another finite precision hack.
                    return mp_ctx.inf
                else:
                    series_gen = phi1_T4_y_gen(*phi_args)
                    res = mp_ctx.nsum(lambda n: next(series_gen),
                                      [0, mp_ctx.inf])
                    if keep_exp_const and not mp_ctx.isinf(res):
                        res *= mp_ctx.exp(x)
        else:
            # x > 0
            if x >= 1:
                if y == 0 and mp_ctx is fp:
                    # FIXME: Yet another finite precision hack.
                    return mp_ctx.inf
                else:
                    series_gen = phi1_T4_y_gen(*phi_args)
                    res = mp_ctx.nsum(lambda n: next(series_gen),
                                      [0, mp_ctx.inf])
                    if keep_exp_const and not mp_ctx.isinf(res):
                        res *= mp_ctx.exp(x)
            else:
                # 0 <= x <= 1
                if y < 0.5:
                    if x < 0.5:
                        series_gen = phi1_T1_x_gen(*phi_args)
                        res = mp_ctx.nsum(lambda n: next(series_gen),
                                          [0, mp_ctx.inf])
                    else:
                        series_gen = phi1_T3_y_gen(*phi_args)
                        res = mp_ctx.nsum(lambda n: next(series_gen),
                                          [0, mp_ctx.inf])
                else:
                    if x < 0.5:
                        series_gen = phi1_T2_x_gen(*phi_args)
                        res = mp_ctx.nsum(lambda n: next(series_gen),
                                          [0, mp_ctx.inf])
                        if keep_exp_const and not mp_ctx.isinf(res):
                            res *= mp_ctx.exp(x)
                    else:
                        series_gen = phi1_T1_x_gen(*phi_args)
                        res = mp_ctx.nsum(lambda n: next(series_gen),
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
