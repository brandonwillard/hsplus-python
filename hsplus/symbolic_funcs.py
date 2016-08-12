from collections import namedtuple

import numpy as np
import mpmath as mp
import sympy as sp


def create_symbolic_terms():
    r""" This function creates symbolic forms for the HS and HS+ priors,
    symbolically derives integrands for their posterior marginals in
    both lambda and kappa parameterizations.

    The model is
    .. math:

        y_i &\sim \operatorname{N}(X \beta_i, \sigma^2_i)
        \\
        \beta_i &\sim \operatorname{N}(0, \lambda^2_i)

    Returns
    =======
    A dictionary containing all the symbolic objects.
    """

    lam, tau, sigma = sp.symbols('lambda tau sigma', nonnegative=True,
                                 real=True)
    kap = sp.Symbol('kappa', real=True, nonnegative=True)
    y = sp.Symbol('y', real=True)

    # The HS prior in lambda
    hs_prior_lam = 2/(sp.pi * tau * (1+(lam/tau)**2))

    # The HS+ prior in lambda
    hsp_prior_lam = 4/(sp.pi**2 * tau) * sp.log(lam**2/tau)/((lam/tau)**2 - 1)

    # Compute the jacobian for the kappa transform
    kap_lam = 1/(1 + (tau*lam)**2)

    #sp.solveset(sp.Eq(kap, kap_lam), lam, sp.Interval.open(0, 1))
    #sp.solve([sp.Eq(kap, kap_lam), sp.LessThan(kap, 1), sp.GreaterThan(kap, 0)], lam)
    #sp.refine(sp.solve(sp.Eq(kap, kap_lam), lam),
    #          sp.Q.is_true(sp.LessThan(kap, 1)) & sp.Q.is_true(sp.GreaterThan(kap, 0)))
    #sp.solve([kap - kap_lam, kap <= 1, 0 < kap], lam)

    lam_kap = sp.solve(sp.Eq(kap, kap_lam), lam)
    # A hackish way to get rid of the (apparently unavoidable) negative root.
    lam_kap, = filter(lambda expr: -1 not in expr.as_coeff_mul(), lam_kap)

    kap_jacobian = sp.simplify(sp.diff(lam_kap, kap))
    lam_to_kap = {lam: lam_kap}

    # Marginal posterior (i.e. integrating beta) conditional on tau and lambda
    post_marg_var = sigma**2 * (1+(tau*lam)**2)
    post_marg_lik_part_lam = 1/sp.sqrt(2 * sp.pi * post_marg_var) *\
        sp.exp(-y**2/(2*post_marg_var))

    # Integrand for HS marginal posterior in the lambda parameterization
    hs_post_marg_int_lam = post_marg_lik_part_lam * hs_prior_lam

    # Integrand for HS marginal posterior in the kappa parameterization
    hs_post_marg_int_kap = hs_post_marg_int_lam.subs(lam_to_kap) *\
        kap_jacobian
    hs_post_marg_int_kap = sp.simplify(hs_post_marg_int_kap)

    # Integrand for HS+ marginal posterior in the lambda parameterization
    hsp_post_marg_int_lam = post_marg_lik_part_lam * hsp_prior_lam

    # Integrand for HS+ marginal posterior in the kappa parameterization
    hsp_post_marg_int_kap = hsp_post_marg_int_lam.subs(lam_to_kap) *\
        kap_jacobian
    hsp_post_marg_int_kap = sp.simplify(hsp_post_marg_int_kap)

    hs_prior_kap = hs_prior_lam.subs(lam_to_kap) * kap_jacobian
    hs_prior_kap = sp.simplify(hs_prior_kap)

    hsp_prior_kap = hsp_prior_lam.subs(lam_to_kap) * kap_jacobian
    hsp_prior_kap = sp.simplify(hsp_prior_kap)

    # These "lambdified" functions are useful for numeric evaluations.
    hs_prior_kap_num = sp.lambdify((kap, tau), hs_prior_kap)
    hsp_prior_kap_num = sp.lambdify((kap, tau), hsp_prior_kap)

    return locals()


symbol_dict = create_symbolic_terms()
symbol_obj = namedtuple('hsplus_symbols', symbol_dict.keys())(**symbol_dict)


def horn_phi1_int_sympy(integrand, kap):
    r""" Symbolic integral for :math:`\Phi_1` function.
    """
    aw = sp.Wild("a", exclude=[kap])
    bw = sp.Wild("b", exclude=[kap, aw])
    vw = sp.Wild("v", exclude=[kap, aw, bw])
    sw = sp.Wild("s", exclude=[kap, aw, bw, vw])
    uw = sp.Wild("u", exclude=[kap, aw, bw, vw, sw])
    Ow = sp.Wild("O", exclude=[kap, aw, bw, vw, sw, uw])

    phi1_match_eqn = (-1)**(vw) * Ow * kap**(aw - 1) *\
        (1 - kap)**(bw - 1) * (kap * uw - 1)**(-vw) * sp.exp(-kap * sw)
    phi1_match_eqn = phi1_match_eqn.simplify()

    integrand_trans = sp.powsimp(sp.collect(
        sp.expand_power_base(sp.simplify(integrand)), kap), combine="exp")

    phi1_match_terms = integrand_trans.match(phi1_match_eqn)

    if phi1_match_terms is None:
        return None

    Phi1_sp = sp.Function("Phi1")

    Const = phi1_match_terms.get(Ow)
    a = phi1_match_terms.get(aw)
    b = phi1_match_terms.get(vw)
    c = phi1_match_terms.get(bw) - a
    w = phi1_match_terms.get(uw)
    z = -phi1_match_terms.get(sw)
    phi1_this_term = Const * Phi1_sp(a, b, c, w, z)

    return phi1_this_term


def horn_phi1_sympy(a, b, g, x, y):
    i, j = sp.symbols("i j", cls=sp.Dummy, integer=True, nonnegative=True)
    res = sp.Sum(sp.RisingFactorial(a, i + j) * sp.RisingFactorial(b, j) /
                 (sp.RisingFactorial(g, i + j) * sp.factorial(i) *
                  sp.factorial(j)) * y**j * x**i,
                 (i, sp.S(0), sp.oo), (j, sp.S(0), sp.oo))
    return res
