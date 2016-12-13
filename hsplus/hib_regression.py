# -*- coding: utf-8 -*-
import numpy as np
# from mpmath import mp, fp

from scipy import optimize

from patsy import dmatrices, build_design_matrices

from .hib_stats import SURE_hib, E_kappa


class HIB_fit(object):
    r"""Class containing linear model fit informaiton.

    Inspired by the patsy_ documentation.

    .. _patsy: https://patsy.readthedocs.io/en/latest/library-developers.html
    """

    def __init__(self, formula_like, data={}, tau=None, sigma=1,
                 a=0.5, b=0.5, s=0., tau_method="SURE"):
        r""" Linear regresion with an HIB regression parameter global-local shrinkage
        prior.

        The model has following form
        .. math:
            y \sim N(X \beta, \sigma^2) \\
            \beta_i \sim N(0, \sigma^2 \tau^2 \lambda^2_i)

        The ::math::`\lambda^2_i` are given ::math::`HIB(a, b, s)` distributions,
        for which ::math::`a = b = 1/2` and ::math::`s = 0` gives the Horseshoe
        prior.


        Parameters
        ==========
        formula: Patsy formula
            Regression formula
        data: array or dataframe
            Data usedin regression formula
        tau: array
            Explicit ::math::`\tau` value(s).  If `tau` is a vector, then
            compute the optimal tau under `tau_method`.  If `tau` is
            `None`, find an optimal value using `tau_method`.
        sigma: float
            Observation variance.
        a: float
            Hypergeometric inverted-beta model parameter
        b: float
            Hypergeometric inverted-beta model parameter
        s: float
            Hypergeometric inverted-beta model parameter
        tau_method: str, or None
            A string signifying the method used to estimate the
            global shrinkage parameter, ::math::`\tau`.  Currently the only
            option is `"SURE"`, which minimizes Stein's unbiased risk estimate
            (SURE).

        Returns
        =======
        beta: array
            Estimated regression parameters
        """
        self.a = a
        self.b = b
        self.s = s
        self.tau0 = np.atleast_1d(tau)
        self.sigma = np.atleast_1d(sigma)
        self.tau_method = tau_method

        y, X = dmatrices(formula_like, data, 1)
        self.nobs = X.shape[0]
        self.beta_mean, self.beta_var, self.tau_opt = self.fit(y, X)
        self._y_design_info = y.design_info
        self._X_design_info = X.design_info

    def fit(self, y, X):
        U_X, d_X, Vt_X = np.linalg.svd(X, full_matrices=False)

        D_inv = np.diag(1./d_X)
        D_inv[np.abs(d_X) < 1e-5] = 0

        alpha_hat = np.linalg.multi_dot([D_inv, U_X.T, np.ravel(y)])

        tau_opt, SURE_opt = self.find_tau(alpha_hat, d_X)

        beta_mean, beta_var = self.estimate_beta(alpha_hat, tau_opt, d_X, Vt_X)

        return (beta_mean, beta_var, tau_opt)

    def tau_opt_SURE(self, alpha_hat, d_X):
        tau_opt_res = optimize.minimize(
            lambda t_: np.sum(SURE_hib(alpha_hat * d_X,
                              self.sigma,
                              t_ * d_X,
                              self.a, self.b, self.s)),
            1,
            method='SLSQP',
            bounds=((1e-7, None),),
            options={'disp': True}
        )
        return (tau_opt_res.x, None)

    opt_method_dict = {"SURE": (tau_opt_SURE, SURE_hib)}

    def find_tau(self, alpha_hat, d_X):

        opt_methods = self.opt_method_dict.get(
            self.tau_method.upper(), None)

        if opt_methods is None:
            from warnings import warn
            warn(("tau_method does not match a valid method. "
                  "Using SURE"))
            opt_methods = self.opt_method_dict['SURE']

        if self.tau0 is None:
            res = opt_methods[0](alpha_hat, d_X)
        else:
            obj_vals = np.sum(opt_methods[1](
                alpha_hat * d_X,
                self.sigma,
                self.tau0 * d_X,
                self.a, self.b, self.s))

            obj_vals = np.atleast_1d(obj_vals)
            arg_min = np.argmin(obj_vals)

            res = (self.tau0[arg_min], obj_vals[arg_min])

        return res

    def estimate_beta(self, alpha_hat, tau, d_X, Vt_X):

        V_X = Vt_X.T

        kappa_m_mp = E_kappa(alpha_hat * d_X,
                             self.sigma,
                             tau * d_X,
                             a=self.a, b=self.b, s=self.s,
                             n=1)

        kappa_m = np.fromiter((float(a_) for a_ in kappa_m_mp),
                              dtype=np.float)

        kappa_m_m1 = (1. - kappa_m)
        alpha_m = kappa_m_m1 * alpha_hat

        beta_mean = np.ravel(V_X.dot(alpha_m))
        beta_var = np.linalg.multi_dot([V_X,
                                        np.diag(kappa_m_m1 *
                                                self.sigma**2),
                                        V_X.T])

        return (beta_mean, beta_var)

    def __repr__(self):
        summary = ("Normal errors regression with HIB({}, {}, {}) prior"
                   "and sigma={}, tau={}\n"
                   "\tModel: {} ~ {}\n"
                   "\tRegression (beta) coefficients:\n"
                   .format(self.a, self.b, self.s,
                           self.sigma, self.tau_opt,
                           self._y_design_info.describe(),
                           self._X_design_info.describe()))
        for name, value in zip(self._X_design_info.column_names,
                               self.beta_mean):
            summary += "\t\t{}:  {:.3g}\n".format(name, value)
        return summary

    def predict(self, new_data):
        (X_new,) = build_design_matrices([self._X_design_info],
                                         new_data)
        return np.dot(X_new, self.betas)
