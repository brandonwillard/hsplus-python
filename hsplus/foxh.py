"""Hypergeometric and Meijer G-functions"""

from __future__ import print_function, division

from sympy.core import S, I, pi, oo, ilcm, Mod
from sympy.core.function import Function, Derivative, ArgumentIndexError
from sympy.core.containers import Tuple
from sympy.core.compatibility import reduce, range
from sympy.core.mul import Mul
from sympy.core.symbol import Dummy

from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
        sinh, cosh, asinh, acosh, atanh, acoth)

from sympy.functions.special import TupleArg, _prep_tuple, TupleParametersBase



class demo_hyper(TupleParametersBase):
    r"""
    The (generalized) hypergeometric function is defined by a series where
    the ratios of successive terms are a rational function of the summation
    index. When convergent, it is continued analytically to the largest
    possible domain.

    The hypergeometric function depends on two vectors of parameters, called
    the numerator parameters :math:`a_p`, and the denominator parameters
    :math:`b_q`. It also has an argument :math:`z`. The series definition is

    .. math ::
        {}_pF_q\left(\begin{matrix} a_1, \dots, a_p \\ b_1, \dots, b_q \end{matrix}
                     \middle| z \right)
        = \sum_{n=0}^\infty \frac{(a_1)_n \dots (a_p)_n}{(b_1)_n \dots (b_q)_n}
                            \frac{z^n}{n!},

    where :math:`(a)_n = (a)(a+1)\dots(a+n-1)` denotes the rising factorial.

    If one of the :math:`b_q` is a non-positive integer then the series is
    undefined unless one of the `a_p` is a larger (i.e. smaller in
    magnitude) non-positive integer. If none of the :math:`b_q` is a
    non-positive integer and one of the :math:`a_p` is a non-positive
    integer, then the series reduces to a polynomial. To simplify the
    following discussion, we assume that none of the :math:`a_p` or
    :math:`b_q` is a non-positive integer. For more details, see the
    references.

    The series converges for all :math:`z` if :math:`p \le q`, and thus
    defines an entire single-valued function in this case. If :math:`p =
    q+1` the series converges for :math:`|z| < 1`, and can be continued
    analytically into a half-plane. If :math:`p > q+1` the series is
    divergent for all :math:`z`.

    Note: The hypergeometric function constructor currently does *not* check
    if the parameters actually yield a well-defined function.

    Examples
    ========

    The parameters :math:`a_p` and :math:`b_q` can be passed as arbitrary
    iterables, for example:

    >>> from sympy.functions import hyper
    >>> from sympy.abc import x, n, a
    >>> hyper((1, 2, 3), [3, 4], x)
    hyper((1, 2, 3), (3, 4), x)

    There is also pretty printing (it looks better using unicode):

    >>> from sympy import pprint
    >>> pprint(hyper((1, 2, 3), [3, 4], x), use_unicode=False)
      _
     |_  /1, 2, 3 |  \
     |   |        | x|
    3  2 \  3, 4  |  /

    The parameters must always be iterables, even if they are vectors of
    length one or zero:

    >>> hyper((1, ), [], x)
    hyper((1,), (), x)

    But of course they may be variables (but if they depend on x then you
    should not expect much implemented functionality):

    >>> hyper((n, a), (n**2,), x)
    hyper((n, a), (n**2,), x)

    The hypergeometric function generalizes many named special functions.
    The function hyperexpand() tries to express a hypergeometric function
    using named special functions.
    For example:

    >>> from sympy import hyperexpand
    >>> hyperexpand(hyper([], [], x))
    exp(x)

    You can also use expand_func:

    >>> from sympy import expand_func
    >>> expand_func(x*hyper([1, 1], [2], -x))
    log(x + 1)

    More examples:

    >>> from sympy import S
    >>> hyperexpand(hyper([], [S(1)/2], -x**2/4))
    cos(x)
    >>> hyperexpand(x*hyper([S(1)/2, S(1)/2], [S(3)/2], x**2))
    asin(x)

    We can also sometimes hyperexpand parametric functions:

    >>> from sympy.abc import a
    >>> hyperexpand(hyper([-a], [], x))
    (-x + 1)**a

    See Also
    ========

    sympy.simplify.hyperexpand
    sympy.functions.special.gamma_functions.gamma
    meijerg

    References
    ==========

    .. [1] Luke, Y. L. (1969), The Special Functions and Their Approximations,
           Volume 1
    .. [2] http://en.wikipedia.org/wiki/Generalized_hypergeometric_function
    """


    def __new__(cls, ap, bq, z):
        # TODO should we check convergence conditions?
        return Function.__new__(cls, _prep_tuple(ap), _prep_tuple(bq), z)

    @classmethod
    def eval(cls, ap, bq, z):
        from sympy import unpolarify
        if len(ap) <= len(bq):
            nz = unpolarify(z)
            if z != nz:
                return hyper(ap, bq, nz)

    def fdiff(self, argindex=3):
        if argindex != 3:
            raise ArgumentIndexError(self, argindex)
        nap = Tuple(*[a + 1 for a in self.ap])
        nbq = Tuple(*[b + 1 for b in self.bq])
        fac = Mul(*self.ap)/Mul(*self.bq)
        return fac*hyper(nap, nbq, self.argument)

    def _eval_expand_func(self, **hints):
        from sympy import gamma, hyperexpand
        if len(self.ap) == 2 and len(self.bq) == 1 and self.argument == 1:
            a, b = self.ap
            c = self.bq[0]
            return gamma(c)*gamma(c - a - b)/gamma(c - a)/gamma(c - b)
        return hyperexpand(self)

    def _eval_rewrite_as_Sum(self, ap, bq, z):
        from sympy.functions import factorial, RisingFactorial, Piecewise
        from sympy import Sum
        n = Dummy("n", integer=True)
        rfap = Tuple(*[RisingFactorial(a, n) for a in ap])
        rfbq = Tuple(*[RisingFactorial(b, n) for b in bq])
        coeff = Mul(*rfap) / Mul(*rfbq)
        return Piecewise((Sum(coeff * z**n / factorial(n), (n, 0, oo)),
                         self.convergence_statement), (self, True))

    @property
    def argument(self):
        """ Argument of the hypergeometric function. """
        return self.args[2]

    @property
    def ap(self):
        """ Numerator parameters of the hypergeometric function. """
        return Tuple(*self.args[0])

    @property
    def bq(self):
        """ Denominator parameters of the hypergeometric function. """
        return Tuple(*self.args[1])

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
        if any(a.is_integer and (a <= 0) == True for a in self.ap + self.bq):
            aints = [a for a in self.ap if a.is_Integer and (a <= 0) == True]
            bints = [a for a in self.bq if a.is_Integer and (a <= 0) == True]
            if len(aints) < len(bints):
                return S(0)
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
                    return S(0)
            if aints or popped:
                # There are still non-positive numerator parameters.
                # This is a polynomial.
                return oo
        if len(self.ap) == len(self.bq) + 1:
            return S(1)
        elif len(self.ap) <= len(self.bq):
            return oo
        else:
            return S(0)

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

    def _eval_simplify(self, ratio, measure):
        from sympy.simplify.hyperexpand import hyperexpand
        return hyperexpand(self)

    def _sage_(self):
        import sage.all as sage
        ap = [arg._sage_() for arg in self.args[0]]
        bq = [arg._sage_() for arg in self.args[1]]
        return sage.hypergeometric(ap, bq, self.argument._sage_())


class foxh(TupleParametersBase):
    r"""
    The Fox H-function is defined by a Mellin-Barnes type integral that
    resembles an inverse Mellin transform. It generalizes the hypergeometric
    functions.

    The Fox H-function depends on four sets of parameters. There are
    "*numerator parameters*"
    :math:`a_1, \dots, a_n; A_1, \dots, A_n` and 
    :math:`a_{n+1}, \dots, a_p; A_{n+1}, \dots A_p`, and there are
    "*denominator parameters*"
    :math:`b_1, \dots, b_m; B_1, \dots, B_m` and 
    :math:`b_{m+1}, \dots, b_q; B_{m+1}, \dots, B_q`.
    Confusingly, it is traditionally denoted as follows (note the position
    of `m`, `n`, `p`, `q`, and how they relate to the lengths of the four
    parameter vectors):

    .. math ::
        H_{p,q}^{m,n} \left(\begin{matrix}
        (a_1, A_1), \dots, (a_n, A_n) & (a_{n+1}, A_{n+1}, \dots, (a_p, A_p) \\
        (b_1, B_1), \dots, (b_m, B_m) & (b_{m+1}, B_{m+1}), \dots, (b_q, B_q)
        \end{matrix} \middle| z \right).

    However, in sympy the four parameter vectors are always available
    separately (see examples), so that there is no need to keep track of the
    decorating sub- and super-scripts on the H symbol.

    The H-function is defined as the following integral:

    .. math ::
         \frac{1}{2 \pi i} \int_L \frac{\prod_{j=1}^m \Gamma(b_j - B_j s)
         \prod_{j=1}^n \Gamma(1 - a_j + A_j s)}{\prod_{j=m+1}^q \Gamma(1- b_j + B_j s)
         \prod_{j=n+1}^p \Gamma(a_j - A_j s)} z^s \mathrm{d}s,

    where :math:`\Gamma(z)` is the gamma function. There are three possible
    contours which we will not describe in detail here (see the references).
    If the integral converges along more than one of them the definitions
    agree. The contours all separate the poles of :math:`\Gamma(1-a_j+s)`
    from the poles of :math:`\Gamma(b_k-s)`, so in particular the G function
    is undefined if :math:`a_j - b_k \in \mathbb{Z}_{>0}` for some
    :math:`j \le n` and :math:`k \le m`.

    The conditions under which one of the contours yields a convergent integral
    are complicated and we do not state them here, see the references.

    Note: Currently the Fox H-function constructor does *not* check any
    convergence conditions.

    Examples
    ========

    You can pass the parameters either as four separate vectors:

    >>> from sympy.functions import foxh
    >>> from sympy.abc import x, a
    >>> from sympy.core.containers import Tuple
    >>> from sympy import pprint
    >>> pprint(foxh(((1,1), (2,1)), ((a, 1), (4,1)), ((5,1),), [], x), use_unicode=False)
       1, 2 /(1,1), (2,1)  (a,1), (4,1) |  \
    |_|     |                           | x|
    | |4, 1 \ (5,1)                     |  /

    or as two nested vectors:

    >>> pprint(foxh([((1,1), (2,1)), ((3,1), (4,1))], ([5], Tuple()), x), use_unicode=False)
       1, 2 /1, 2  3, 4 |  \
    |_|     |           | x|
    | |4, 1 \ 5         |  /
       
    As with the hypergeometric function, the parameters may be passed as
    arbitrary iterables. Vectors of length zero and one also have to be
    passed as iterables. The parameters need not be constants, but if they
    depend on the argument then not much implemented functionality should be
    expected.

    All the subvectors of parameters are available:

    >>> from sympy import pprint
    >>> h = foxh([1], [2], [3], [4], x)
    >>> pprint(h, use_unicode=False)
       1, 1 /1  2 |  \
    |_|     |     | x|
    | |2, 2 \3  4 |  /
    >>> h.an
    (1,)
    >>> h.ap
    (1, 2)
    >>> h.aother
    (2,)
    >>> h.bm
    (3,)
    >>> h.bq
    (3, 4)
    >>> h.bother
    (4,)

    The Fox H-function generalizes the hypergeometric functions.
    In some cases it can be expressed in terms of hypergeometric functions,
    using Slater's theorem. For example:

    >>> from sympy import hyperexpand
    >>> from sympy.abc import a, b, c
    >>> hyperexpand(foxh([(a,1)], [], [(c,1)], [(b,1)], x), allow_hyper=True)
    x**c*gamma(-a + c + 1)*hyper((-a + c + 1,),
                                 (-b + c + 1,), -x)/gamma(-b + c + 1)

    Thus the Fox H-function also subsumes many named functions as special
    cases. You can use expand_func or hyperexpand to (try to) rewrite a
    Meijer G-function in terms of named special functions. For example:

    >>> from sympy import expand_func, S
    >>> expand_func(foxh([[],[]], [[0],[]], -x))
    exp(x)
    >>> hyperexpand(foxh([[],[]], [[S(1)/2],[0]], (x/2)**2))
    sin(x)/sqrt(pi)

    See Also
    ========

    hyper
    sympy.simplify.hyperexpand

    References
    ==========

    .. [1] Mathai, A. M. et al. (2010), The H-function

    """


    def __new__(cls, *args):
        if len(args) == 5:
            args = [(args[0], args[1]), (args[2], args[3]), args[4]]
        if len(args) != 3:
            raise TypeError("args must eiter be as, as', bs, bs', z or "
                            "as, bs, z")

        def tr(p):
            if len(p) != 2:
                raise TypeError("wrong argument")
            return TupleArg(_prep_tuple(p[0]), _prep_tuple(p[1]))

        # TODO should we check convergence conditions?
        return Function.__new__(cls, tr(args[0]), tr(args[1]), args[2])

    def fdiff(self, argindex=3):
        if argindex != 3:
            return self._diff_wrt_parameter(argindex[1])
        if len(self.an) >= 1:
            a = list(self.an)
            a[0] -= 1
            G = meijerg(a, self.aother, self.bm, self.bother, self.argument)
            return 1/self.argument * ((self.an[0] - 1)*self + G)
        elif len(self.bm) >= 1:
            b = list(self.bm)
            b[0] += 1
            G = meijerg(self.an, self.aother, b, self.bother, self.argument)
            return 1/self.argument * (self.bm[0]*self - G)
        else:
            return S.Zero

    def _diff_wrt_parameter(self, idx):
        # Differentiation wrt a parameter can only be done in very special
        # cases. In particular, if we want to differentiate with respect to
        # `a`, all other gamma factors have to reduce to rational functions.
        #
        # Let MT denote mellin transform. Suppose T(-s) is the gamma factor
        # appearing in the definition of G. Then
        #
        #   MT(log(z)G(z)) = d/ds T(s) = d/da T(s) + ...
        #
        # Thus d/da G(z) = log(z)G(z) - ...
        # The ... can be evaluated as a G function under the above conditions,
        # the formula being most easily derived by using
        #
        # d  Gamma(s + n)    Gamma(s + n) / 1    1                1     \
        # -- ------------ =  ------------ | - + ----  + ... + --------- |
        # ds Gamma(s)        Gamma(s)     \ s   s + 1         s + n - 1 /
        #
        # which follows from the difference equation of the digamma function.
        # (There is a similar equation for -n instead of +n).

        # We first figure out how to pair the parameters.
        an = list(self.an)
        ap = list(self.aother)
        bm = list(self.bm)
        bq = list(self.bother)
        if idx < len(an):
            an.pop(idx)
        else:
            idx -= len(an)
            if idx < len(ap):
                ap.pop(idx)
            else:
                idx -= len(ap)
                if idx < len(bm):
                    bm.pop(idx)
                else:
                    bq.pop(idx - len(bm))
        pairs1 = []
        pairs2 = []
        for l1, l2, pairs in [(an, bq, pairs1), (ap, bm, pairs2)]:
            while l1:
                x = l1.pop()
                found = None
                for i, y in enumerate(l2):
                    if not Mod((x - y).simplify(), 1):
                        found = i
                        break
                if found is None:
                    raise NotImplementedError('Derivative not expressible '
                                              'as G-function?')
                y = l2[i]
                l2.pop(i)
                pairs.append((x, y))

        # Now build the result.
        res = log(self.argument)*self

        for a, b in pairs1:
            sign = 1
            n = a - b
            base = b
            if n < 0:
                sign = -1
                n = b - a
                base = a
            for k in range(n):
                res -= sign*meijerg(self.an + (base + k + 1,), self.aother,
                                    self.bm, self.bother + (base + k + 0,),
                                    self.argument)

        for a, b in pairs2:
            sign = 1
            n = b - a
            base = a
            if n < 0:
                sign = -1
                n = a - b
                base = b
            for k in range(n):
                res -= sign*meijerg(self.an, self.aother + (base + k + 1,),
                                    self.bm + (base + k + 0,), self.bother,
                                    self.argument)

        return res

    def get_period(self):
        """
        Return a number P such that G(x*exp(I*P)) == G(x).

        >>> from sympy.functions.special.hyper import meijerg
        >>> from sympy.abc import z
        >>> from sympy import pi, S

        >>> meijerg([1], [], [], [], z).get_period()
        2*pi
        >>> meijerg([pi], [], [], [], z).get_period()
        oo
        >>> meijerg([1, 2], [], [], [], z).get_period()
        oo
        >>> meijerg([1,1], [2], [1, S(1)/2, S(1)/3], [1], z).get_period()
        12*pi
        """
        # This follows from slater's theorem.
        def compute(l):
            # first check that no two differ by an integer
            for i, b in enumerate(l):
                if not b.is_Rational:
                    return oo
                for j in range(i + 1, len(l)):
                    if not Mod((b - l[j]).simplify(), 1):
                        return oo
            return reduce(ilcm, (x.q for x in l), 1)
        beta = compute(self.bm)
        alpha = compute(self.an)
        p, q = len(self.ap), len(self.bq)
        if p == q:
            if beta == oo or alpha == oo:
                return oo
            return 2*pi*ilcm(alpha, beta)
        elif p < q:
            return 2*pi*beta
        else:
            return 2*pi*alpha

    def _eval_expand_func(self, **hints):
        from sympy import hyperexpand
        return hyperexpand(self)

    def _eval_evalf(self, prec):
        # The default code is insufficient for polar arguments.
        # mpmath provides an optional argument "r", which evaluates
        # G(z**(1/r)). I am not sure what its intended use is, but we hijack it
        # here in the following way: to evaluate at a number z of |argument|
        # less than (say) n*pi, we put r=1/n, compute z' = root(z, n)
        # (carefully so as not to loose the branch information), and evaluate
        # G(z'**(1/r)) = G(z'**n) = G(z).
        from sympy.functions import exp_polar, ceiling
        from sympy import Expr
        import mpmath
        z = self.argument
        znum = self.argument._eval_evalf(prec)
        if znum.has(exp_polar):
            znum, branch = znum.as_coeff_mul(exp_polar)
            if len(branch) != 1:
                return
            branch = branch[0].args[0]/I
        else:
            branch = S(0)
        n = ceiling(abs(branch/S.Pi)) + 1
        znum = znum**(S(1)/n)*exp(I*branch / n)

        # Convert all args to mpf or mpc
        try:
            [z, r, ap, bq] = [arg._to_mpmath(prec)
                    for arg in [znum, 1/n, self.args[0], self.args[1]]]
        except ValueError:
            return

        with mpmath.workprec(prec):
            v = mpmath.meijerg(ap, bq, z, r)

        return Expr._from_mpmath(v, prec)

    def integrand(self, s):
        """ Get the defining integrand D(s). """
        from sympy import gamma
        return self.argument**s \
            * Mul(*(gamma(b - s) for b in self.bm)) \
            * Mul(*(gamma(1 - a + s) for a in self.an)) \
            / Mul(*(gamma(1 - b + s) for b in self.bother)) \
            / Mul(*(gamma(a - s) for a in self.aother))

    @property
    def argument(self):
        """ Argument of the Meijer G-function. """
        return self.args[2]

    @property
    def an(self):
        """ First set of numerator parameters. """
        return Tuple(*self.args[0][0])

    @property
    def ap(self):
        """ Combined numerator parameters. """
        return Tuple(*(self.args[0][0] + self.args[0][1]))

    @property
    def aother(self):
        """ Second set of numerator parameters. """
        return Tuple(*self.args[0][1])

    @property
    def bm(self):
        """ First set of denominator parameters. """
        return Tuple(*self.args[1][0])

    @property
    def bq(self):
        """ Combined denominator parameters. """
        return Tuple(*(self.args[1][0] + self.args[1][1]))

    @property
    def bother(self):
        """ Second set of denominator parameters. """
        return Tuple(*self.args[1][1])

    @property
    def _diffargs(self):
        return self.ap + self.bq

    @property
    def nu(self):
        r""" A quantity related to the convergence region of the integral,
            In Mathai et al. (2010) Equation (1.9) this is refered to as
            :math:`\mu`.  
        """
        return sum(self.bq) - sum(self.ap)

    @property
    def delta(self):
        r""" A quantity related to the convergence region of the integral,
            c.f. references. 
            In Mathai et al. (2010) Equation (1.22) this is refered to as
            :math:`c^\star`.  
        """
        return len(self.bm) + len(self.an) - S(len(self.ap) + len(self.bq))/2

    @property
    def mu(self):
        r"""
        See meijerg.nu.
        """
        return self.nu()

    @property
    def c_star(self):
        """ A quantity related to the convergence region of the integral.
            See Mathai et al. (2010) Equation .
        """
        g_m = len(gfunc.bm)
        g_n = len(gfunc.an)
        g_p = len(self.ap)
        g_q = len(self.bq)
        c_star = g_m + g_n - S(g_p + g_q)/2
        return c_star


    @property
    def convergence_statement(self):
        """ Return a condition on z under which the series converges. """
        #from sympy import And, Or, re, Ne, oo
        #R = self.radius_of_convergence
        #if R == 0:
        #    return False
        #if R == oo:
        #    return True
        ## The special functions and their approximations, page 44
        #e = self.eta
        #z = self.argument
        #c1 = And(re(e) < 0, abs(z) <= 1)
        #c2 = And(0 <= re(e), re(e) < 1, abs(z) <= 1, Ne(z, 1))
        #c3 = And(re(e) >= 1, abs(z) < 1)
        #return Or(c1, c2, c3)
        pass

    def _eval_expand_func(self, **hints):
        #from sympy import gamma, hyperexpand
        #if len(self.ap) == 2 and len(self.bq) == 1 and self.argument == 1:
        #    a, b = self.ap
        #    c = self.bq[0]
        #    return gamma(c)*gamma(c - a - b)/gamma(c - a)/gamma(c - b)
        #return hyperexpand(self)
        pass

    def _eval_rewrite_as_Sum(self, ap, bq, z):
        #from sympy.functions import factorial, RisingFactorial, Piecewise
        #from sympy import Sum
        #n = Dummy("n", integer=True)
        #rfap = Tuple(*[RisingFactorial(a, n) for a in ap])
        #rfbq = Tuple(*[RisingFactorial(b, n) for b in bq])
        #coeff = Mul(*rfap) / Mul(*rfbq)
        #return Piecewise((Sum(coeff * z**n / factorial(n), (n, 0, oo)),
        #                 self.convergence_statement), (self, True))
        pass


