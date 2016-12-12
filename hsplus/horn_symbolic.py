# -*- coding: utf-8 -*-
import sympy as sp
from sympy.core.containers import Tuple
from sympy.core.symbol import Dummy
from sympy.functions.special.hyper import TupleParametersBase, _prep_tuple
from sympy import cacheit

from .horn_numeric import horn_phi1


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

    def __new__(cls, *args, **kwargs):

        try:
            if len(args) == 4:
                ap, bq, x, y = args
                ap = _prep_tuple(ap)
                bq = _prep_tuple(bq)
                a, b = ap
                q, = bq
            elif len(args) == 5:
                a, b, q, x, y = args
        except:
            raise ValueError(("HornPhi1 takes (a,b,g,x,y) or ((a,b),(g,),x,y) "
                              "non-keyword arguments only"))

        # TODO: Convert negative a, b and negative non-integer g indices.
        #if any(sp.ask(sp.Q.integer(t_) & sp.Q.negative) for t_ in ap[0:2])

        return super(HornPhi1, cls).__new__(cls, a, b, q, x, y)

    def equals(self, other, failing_expression=False):
        r""" Compare Sum forms.

        """
        res = (self == other)

        if res is True:
            return True

        this_sum_form = self._eval_rewrite_as_Sum(*self.args)

        if isinstance(other, HornPhi1):

            other_sum_form = other.rewrite(sp.Sum)
            res = this_sum_form.equals(other_sum_form,
                                       failing_expression=failing_expression)

        elif isinstance(other, sp.Piecewise):

            res = this_sum_form.equals(other,
                                       failing_expression=failing_expression)

        elif isinstance(other, sp.Mul) and other.has(sp.Integral):

            this_int_form = self._eval_rewrite_as_Integral(*self.args)
            res = this_int_form.equals(other,
                                       failing_expression=failing_expression)

        else:
            res = super(HornPhi1, self).equals(other, failing_expression)

        return res

    def _eval_expand_func(self, **hints):
        r"""
        Is this also the route by which we provide the recursion relations?
        If so, we'll have to make extensive use of the hints to determine
        exactly which recurrence relation to use.

        .. todo:
            Implement standard reductions, e.g. limits of [F1_values]_ and
            [F1_idents]_.

        .. [F1_values] http://functions.wolfram.com/HypergeometricFunctions/AppellF1/03/ShowAll.html
        .. [F1_idents] http://functions.wolfram.com/HypergeometricFunctions/AppellF1/17/ShowAll.html
        """
        #res = super(HornPhi1, self)._eval_expand_func(**hints)
        res = self
        return res

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
        from mpmath import mp

        # Numerically evaluate arguments
        #args_num = [a_._eval_evalf(prec) for a_ in self.argument]

        (xnum, xn), (ynum, yn) = map(self._polarify, self.argument)

        # Convert all args to mpf or mpc
        try:
            [x, y, xr, yr, ap, bq] = [arg._to_mpmath(prec)
                                      for arg in [xnum, ynum, 1/xn, 1/yn,
                                                  self.ap, self.bq]]
        except ValueError:
            return

        a, b = ap
        g, = bq
        with mp.workprec(prec):
            try:
                v = horn_phi1(a, b, g, x, y)
            except ValueError:
                # FIXME: This isn't really correct; could be mp.inf or mp.ninf.
                v = mp.nan

        return Expr._from_mpmath(v, prec)

    def fdiff(self, argindex=3):
        r"""
        .. todo:
            No order of differentiation?  Those formulas should be implemented
            somewhere.
        """

        if argindex not in (3, 4):
            raise NotImplementedError()

        # TODO: n-th derivative is just `a/b + n`.
        nap = Tuple(*[a + 1 for a in self.ap])
        nbq = Tuple(*[b + 1 for b in self.bq])

        if argindex is 3:
            # TODO: n-th derivative
            #fac = sp.Mul([sp.RisingFactorial(a_, n) for a_ in self.ap],
            #             [1/sp.RisingFactorial(b_, n) for b_ in self.bq])
            fac = sp.Mul(self.ap[0], self.ap[1], 1/self.bq[0])
        else:
            raise NotImplementedError()

        return fac * HornPhi1(*((nap, nbq) + self.argument))

    @cacheit
    def summand(self, m, n):
        r""" Returns the summand that defines this confluent hypergeometric series.

        XXX: The arguments (x, y) do *not* match Gordy's, but instead they
        correspond to the usual definition of the Horn or Humbert
        :math:`\Phi_1`.  This means that the :math:`b` or :math:`\beta` rising
        factorial term in the summand's numerator shares its index variable
        with the :math:`x` term, and *not* the :math:`y` term--as alphabetical
        order (and reason) would have it.
        """
        from sympy import RisingFactorial, factorial, Mul

        res = Mul(RisingFactorial(self.ap[0], m+n),
                  RisingFactorial(self.ap[1], m),
                  1/RisingFactorial(self.bq[0], m+n),
                  1/factorial(m), 1/factorial(n),
                  evaluate=False)

        return res

    @cacheit
    def _eval_rewrite_as_Sum(self, ap, bq, x, y):
        r""" Expand to double sum form.

        .. todo:
            Could also expand to specialized series forms, e.g. limits of
            [F1_series]_.

        .. [F1_series] http://functions.wolfram.com/HypergeometricFunctions/AppellF1/06/ShowAll.html
        """
        from sympy.functions import Piecewise
        from sympy import Sum, oo
        m = Dummy("m", integer=True, nonnegative=True)
        n = Dummy("n", integer=True, nonnegative=True)
        return Piecewise((Sum(self.summand(m, n) * x**m * y**n,
                              (m, 0, oo), (n, 0, oo)),
                          self.convergence_statement), (self, True))

    def _eval_nseries(self, x, n, logx):
        r"""
        .. todo:
            Could also use the series forms from here [F1_series]_.

        .. [F1_series] http://functions.wolfram.com/HypergeometricFunctions/AppellF1/06/ShowAll.html
        """
        res = super(HornPhi1, self)._eval_nseries(x, n, logx)
        return res

    @cacheit
    def _eval_rewrite_as_Integral(self, ap, bq, x, y):
        r""" One integral form for a :math:`\Phi_1` function.
        This is better put in a matching library, among the other
        possible integral expressions.

        TODO: Can also be written as double integral and a bivariate G-function [1]_

        .. [1] http://mathworld.wolfram.com/AppellHypergeometricFunction.html
        """
        from sympy import gamma
        a, b = ap
        g, = bq

        lam = Dummy('lambda', real=True, nonnegative=True)
        res = gamma(g) / gamma(a) / gamma(g - a)
        integrand = sp.Mul(lam**(a-1), (1 - lam)**(g - a - 1),
                           (1 - x * lam)**(-b), sp.exp(y * lam),
                           evaluate=False)
        res *= sp.Integral(integrand, (lam, 0, 1))
        return res

    @property
    def argument(self):
        """ Arguments of the Horn Phi1 function. """
        return Tuple(*self.args[3:])

    @property
    def ap(self):
        """ Numerator parameters of series rising factorial terms. """
        return Tuple(self.args[0], self.args[1])

    @property
    def bq(self):
        """ Denominator parameters of series rising factorial terms. """
        return Tuple(self.args[2])

    @property
    def _diffargs(self):
        return self.ap + self.bq
    @property
    def eta(self):
        """ A quantity related to the convergence of the series. """
        return sum(self.ap) - sum(self.bq)

    @property
    def radius_of_convergence(self):
        """
        XXX: Need to check that this still holds for the bivariate case!

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
        r"""
        XXX: Need to check that this still holds for the bivariate case!

        Return a condition on z under which the series converges.
        """
        from sympy import And, Or, re, Ne, oo
        R = self.radius_of_convergence
        if R == 0:
            return False
        if R == oo:
            return True
        # The special functions and their approximations, page 44
        e = self.eta
        z_1, z_2 = self.argument

        def param_convergence_conditions(z):
            c1 = And(re(e) < 0, abs(z) <= 1)
            c2 = And(0 <= re(e), re(e) < 1, abs(z) <= 1, Ne(z, 1))
            c3 = And(re(e) >= 1, abs(z) < 1)
            return Or(c1, c2, c3)

        return And(param_convergence_conditions(z_1),
                   param_convergence_conditions(z_2))

    def _latex(self, printer, exp=None):
        if len(self.args) != 5:
            raise ValueError("Args length should be 1")
        return (r'\operatorname{{\Phi_1}}'
                r'{{\left({}\right)}}').format(printer._print(self.args))

    def __repr__(self):
        from sympy.printing import sstr
        return sstr(self, order=None)

    def __str__(self):
        from sympy.printing import sstr
        return sstr(self, order=None)


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

