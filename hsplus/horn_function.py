# -*- coding: utf-8 -*-
import sympy as sp
import numpy as np
from mpmath import mp, fp

from sympy.core.function import Function, ArgumentIndexError
from sympy.core.containers import Tuple
from sympy.core.symbol import Dummy
from sympy.functions.special.hyper import TupleParametersBase, _prep_tuple


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
    keep_exp_const: bool (optional)
        Include constant multipliers in `exp(x)` and ``exp(y)``.  Set this to
        `False` when computing ratios with equal ``x`` and ``y`` parameters.

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
        raise ValueError("Parameter y must be 0 <= y < 1")

    if (0 <= y and y < 1):
        #if mp_ctx.chop(y) == 0:
        #    res = mp_ctx.hyp2f1(a, 1, g, x)
        phi_args = (a, b, g, x, y)
        if x < 0:
            if x > -1:
                # -1 < x < 0
                res = mp_ctx.nsum(lambda n: phi1_T4_y(n, *phi_args),
                                  [0, mp_ctx.inf])
                if keep_exp_const:
                    res *= mp_ctx.exp(x)
            else:
                # x <= -1

                #res = mp_ctx.nsum(lambda n: phi1_T2_x(n, *phi_args),
                #                  [0, mp_ctx.inf])
                #if keep_exp_const:
                #    res *= mp_ctx.exp(x)
                res = mp_ctx.nsum(lambda n: phi1_T3_y(n, *phi_args),
                                  [0, mp_ctx.inf])
        else:
            if x >= 1:
                # TODO, XXX: Large values aren't handled well!  This is
                # probably where quadrature will (slightly) outperform:
                #   res = horn_phi1_quad_single(*phi_args)

                #res = mp_ctx.nsum(lambda n: phi1_T3_y(n, *phi_args),
                #                  [0, mp_ctx.inf])

                res = mp_ctx.nsum(lambda n: phi1_T4_y(n, *phi_args),
                                  [0, mp_ctx.inf])
                if keep_exp_const:
                    res *= mp_ctx.exp(x)
            else:
                # 0 <= x <= 1
                res = mp_ctx.nsum(lambda n: phi1_T1_x(n, *phi_args),
                                  [0, mp_ctx.inf])
    elif mp.isfinite(y):
        res = mp_ctx.power(1 - y, -b)
        res *= horn_phi1_gordy_single(g - a, b, g, -x, y / (y - 1.),
                                      keep_exp_const=keep_exp_const)
        if keep_exp_const:
            res *= mp_ctx.exp(x)
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


class HornPhi1(TupleParametersBase):
    r""" The Horn :math:`\Phi_1` function (also known as the [Humbert series]_).

    It is a bivariate confluent hypergeometric function and takes numerator
    parameters :math:`a_p`, denominator parameters :math:`b_q`
    and arguments :math:`x` and :math:`y`.

    This class definition follows `sympy.functions.special.hyper`.

    XXX: The arguments (x, y) do *not* match Gordy's, but instead they
    correspond to the usual definition of the Horn :math:`\Phi_1`.
    This means that the :math:`b` or :math:`\beta` rising
    factorial term in the summand's numerator shares its index variable
    with the :math:`x` term, and *not* the :math:`y` term--as alphabetical
    order (and reason) would have it.

    Examples
    ========
        >>> from hsplus.horn_function import HornPhi1
        >>> import sympy as sp
        >>> a, b, g = sp.symbols('a b g', real=True)
        >>> x, y = sp.symbols('x, y', real=True)
        >>> phi1 = HornPhi1((a, b), (g,), x, y)
        >>> sp.pprint(phi1)
        HornPhi₁((a, b), (g,), x, y)
        >>> phi1.evalf(subs={a: sp.S.Half, b: sp.S.One,
        ...                  g: sp.Rational(3, 2), x: 1, y: 0})
        1.46265174590718
        >>> sp.pprint(phi1_int)
             1
             ⌠
             ⎮  a - 1         -a + g - 1           -b  λ⋅y
        Γ(g)⋅⎮ λ     ⋅(-λ + 1)          ⋅(-λ⋅x + 1)  ⋅ℯ    dλ
             ⌡
             0
        ─────────────────────────────────────────────────────
                            Γ(a)⋅Γ(-a + g)

    .. todo:
        Add to hyperexpand table:
        http://docs.sympy.org/latest/modules/simplify/hyperexpand.html#extending-the-hypergeometric-tables

    .. [Humbert series] http://en.wikipedia.org/wiki/Humbert_series
    """

    def __new__(cls, ap, bq, x, y):
        # TODO: Convert negative a, b and negative non-integer g indices.
        #if any(sp.ask(sp.Q.integer(t_) & sp.Q.negative) for t_ in ap[0:2])

        return Function.__new__(cls, _prep_tuple(ap), _prep_tuple(bq), x, y)

    def _eval_expand_func(self, **hints):
        pass

    def _polarify(self, num):
        r""" Is this really the same as sympy.polarify?
        """
        from sympy.functions import exp_polar, ceiling

        if num.has(exp_polar):
            num, branch = num.as_coeff_mul(exp_polar)
            if len(branch) != 1:
                return
            branch = branch[0].args[0]/sp.I
        else:
            branch = sp.S(0)

        n = ceiling(abs(branch/sp.S.Pi)) + 1
        num = num**(sp.S(1)/n) * sp.exp(sp.I*branch / n)

        return num, n

    def _eval_evalf(self, prec):
        r"""
        From `sympy.functions.hyper.meijerg`:

            The default code is insufficient for polar arguments.
            mpmath provides an optional argument "r", which evaluates
            G(z**(1/r)). I am not sure what its intended use is, but we hijack it
            here in the following way: to evaluate at a number z of |argument|
            less than (say) n*pi, we put r=1/n, compute z' = root(z, n)
            (carefully so as not to loose the branch information), and evaluate
            G(z'**(1/r)) = G(z'**n) = G(z).
        """
        from sympy import Expr
        import mpmath as mp

        # Numerically evaluate arguments
        #args_num = [a_._eval_evalf(prec) for a_ in self.argument]

        (xnum, xn), (ynum, yn) = map(self._polarify, self.argument)

        # Convert all args to mpf or mpc
        try:
            [x, y, xr, yr, ap, bq] = [arg._to_mpmath(prec)
                                      for arg in [xnum, ynum, 1/xn, 1/yn,
                                                  self.args[0], self.args[1]]]
        except ValueError:
            return

        a, b = ap
        g, = bq
        with mp.workprec(prec):
            v = horn_phi1_gordy_single(a, b, g, x, y)

        return Expr._from_mpmath(v, prec)

    def fdiff(self, argindex=3):
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

        if argindex != 3:
            raise ArgumentIndexError(self, argindex)

        nap = Tuple(*[a + 1 for a in self.ap])
        nbq = Tuple(*[b + 1 for b in self.bq])
        fac = sp.Mul(*self.ap)/sp.Mul(*self.bq)
        return fac*sp.hyper(nap, nbq, self.argument)

    def summand(self, m, n):
        r""" Returns the summand that defines this confluent hypergeometric series.

        XXX: The arguments (x, y) do *not* match Gordy's, but instead they
        correspond to the usual definition of the Horn or Humbert
        :math:`\Phi_1`.  This means that the :math:`b` or :math:`\beta` rising
        factorial term in the summand's numerator shares its index variable
        with the :math:`x` term, and *not* the :math:`y` term--as alphabetical
        order (and reason) would have it.
        """
        from sympy import RisingFactorial, factorial
        return RisingFactorial(self.ap[0], m+n) \
            * RisingFactorial(self.ap[1], m) \
            / RisingFactorial(self.bq[0], m+n) \
            / factorial(m) \
            / factorial(n)

    def _eval_rewrite_as_Sum(self, ap, bq, x, y):
        from sympy.functions import Piecewise
        from sympy import Sum, oo
        m = Dummy("m", integer=True, nonnegative=True)
        n = Dummy("n", integer=True, nonnegative=True)
        return Piecewise((Sum(self.summand(m, n) * x**m * y**n,
                              (m, 0, oo), (n, 0, oo)),
                          self.convergence_statement), (self, True))

    def _eval_rewrite_as_Integral(self, ap, bq, x, y):
        r""" One integral form for a :math:`\Phi_1` function.
        This is better put in a matching library, among the other
        possible integral expressions.
        """
        from sympy import gamma
        a, b = ap
        g, = bq

        lam = Dummy('lambda', real=True, nonnegative=True)
        res = gamma(g) / gamma(a) / gamma(g - a)
        integrand = sp.Mul(lam**(a-1), (1 - lam)**(g - a - 1),
                           (1 - x * lam)**(-b), sp.exp(y * lam))
        res *= sp.Integral(integrand, (lam, 0, 1))
        return res

    @property
    def argument(self):
        """ Arguments of the Horn Phi1 function. """
        return (self.args[2], self.args[3])

    @property
    def ap(self):
        """ Numerator parameters of series rising factorial terms. """
        return Tuple(*self.args[0])

    @property
    def bq(self):
        """ Denominator parameters of series rising factorial terms. """
        return Tuple(*self.args[1])

    @property
    def radius_of_convergence(self):
        """
        From `sympy.functions.hyper`:

        Compute the radius of convergence of the defining series.
        Note that even if this is not oo, the function may still be evaluated
        outside of the radius of convergence by analytic continuation. But if
        this is zero, then the function is not actually defined anywhere else.
        >>> from sympy.functions import hyper
        >>> from sympy.abc import z
        >>> hyper((1, 2), [3], z).radius_of_convergence
        1
        >>> hyper((1, 2, 3), [4], z).radius_of_convergence
        0
        >>> hyper((1, 2), (3, 4), z).radius_of_convergence
        oo
        """
        if any(a.is_integer and (a <= 0) is True for a in self.ap + self.bq):
            aints = [a for a in self.ap if a.is_Integer and (a <= 0) is True]
            bints = [a for a in self.bq if a.is_Integer and (a <= 0) is True]
            if len(aints) < len(bints):
                return sp.S(0)
            popped = False
            for b in bints:
                cancelled = False
                while aints:
                    a = aints.pop()
                    if a >= b:
                        cancelled = True
                        break
                    popped = True
                if not cancelled:
                    return sp.S(0)
            if aints or popped:
                # There are still non-positive numerator parameters.
                # This is a polynomial.
                return sp.oo
        if len(self.ap) == len(self.bq) + 1:
            return sp.S(1)
        elif len(self.ap) <= len(self.bq):
            return sp.oo
        else:
            return sp.S(0)

    @property
    def convergence_statement(self):
        """ Return a condition on z under which the series converges. """
        from sympy import And, Or, re, Ne, oo
        R = self.radius_of_convergence
        if R == 0:
            return False
        if R == oo:
            return True
        # The special functions and their approximations, page 44
        e = self.eta
        z = self.argument
        c1 = And(re(e) < 0, abs(z) <= 1)
        c2 = And(0 <= re(e), re(e) < 1, abs(z) <= 1, Ne(z, 1))
        c3 = And(re(e) >= 1, abs(z) < 1)
        return Or(c1, c2, c3)

    def _latex(self, printer, exp=None):
        if len(self.args) != 4:
            raise ValueError("Args length should be 1")
        return (r'\operatorname{{\Phi_1}}'
                r'{{\left({}, {}; {}, {} \right)}}').format(
                    printer._print(*self.args))


def phi1_int_match(expr):
    r""" The following pattern matches against a :math:`\Phi_1` integral form.

    Examples
    ========
        from hsplus.hornfunction import HornPhi1, phi1_sym_int
        import sympy as sp
        a, b, g = sp.symbols('a b g', real=True)
        x, y = sp.symbols('x, y', real=True)

        phi1 = HornPhi1((a, b), (g,), x, y)
        phi1_int = phi1.rewrite("Integral")

        sp.pprint(phi1_int)

        phi1_clone = phi1_int_match(phi1_int)

        sp.pprint(phi1_clone)

        assert phi1_clone.equals(phi1)
    """
    from sympy import gamma, Wild
    int_expr, = expr.find(sp.Integral)
    const_part = expr.extract_multiplicatively(int_expr)

    integrand = int_expr.args[0]
    lam = int_expr.args[1][0]

    a_w = Wild("a", exclude=[lam])
    b_w = Wild("b", exclude=[lam, a_w])
    g_w = Wild("g", exclude=[lam, b_w, a_w])
    y_w = Wild("y", exclude=[lam, b_w, a_w, g_w])
    x_w = Wild("x", exclude=[lam, b_w, a_w, g_w, y_w])
    C_w = Wild("C", exclude=[lam, b_w, a_w, g_w, y_w, x_w])

    phi1_integrand = sp.Mul(C_w, lam**(a_w-1), (1 - lam)**(g_w - a_w - 1),
                            (1 - x_w * lam)**(-b_w), sp.exp(y_w * lam))

    #integrand = sp.expand_power_base(integrand)
    #integrand = sp.powsimp(sp.collect(integrand, kap), combine="exp")

    phi1_match_terms = integrand.match(phi1_integrand)

    if phi1_match_terms is None:
        return None

    C = 1/(C_w * gamma(g_w) / gamma(a_w) /
           gamma(g_w - a_w)).subs(phi1_match_terms)

    if const_part is not None:
        C *= const_part

    phi1_this_term = (C * HornPhi1((a_w, b_w),
                                   (g_w,), x_w, y_w)).subs(phi1_match_terms)

    return phi1_this_term
