import numpy as np
from scipy.special import expi
from scipy.special import gamma
from scipy.special import gammainc
from scipy.stats import genextreme

from extreme_tree.equal_distributions import empirical_cdf


def _log_score(target, params):
    mu, sigma, xi = params
    score = -genextreme.logpdf(target, loc=mu, scale=sigma, c=-xi)
    score = np.nan_to_num(score, copy=False, nan=0, posinf=0)
    return score.sum()


def _kolmogorov_smirnov(target, params):
    mu, sigma, xi = params
    empirical_values, empirical_quantiles = empirical_cdf(target)
    theoretical_quantiles = genextreme.cdf(empirical_values, loc=mu, scale=sigma, c=-xi)
    statistic = np.max(abs(empirical_quantiles - theoretical_quantiles))
    return statistic


def _crps_score(y, params):
    mu, sigma, xi = params

    xi_zero = np.isclose(xi, 0, atol=1e-3)
    one_over_xi = np.divide(1, xi, where=~xi_zero)

    cdf = genextreme.cdf(y, loc=mu, scale=sigma, c=-xi)
    logcdf = genextreme.logcdf(y, loc=mu, scale=sigma, c=-xi)

    gamma_term = gamma(1 - xi)
    gammainc_term = gamma_term * gammainc(1 - xi, -logcdf)

    first_part = (mu - y - sigma * one_over_xi) * (1 - 2 * cdf)
    second_part = sigma * one_over_xi * (2 ** xi * gamma_term - 2 * gammainc_term)
    score_nonzero = first_part - second_part

    score_zero = mu - y + sigma * (np.euler_gamma - np.log(2)) - 2 * sigma * expi(logcdf)

    score = np.positive(score_zero, where=xi_zero, out=score_nonzero)
    return score.sum()


def _pwm_estimate(target):
    # Probability weighted moment estimate
    # See https://doi.org/10.1080/00401706.1985.10488049
    target = np.ravel(target)

    x = np.sort(target)
    n = len(target)
    j = np.arange(1, n + 1)

    if n <= 2:
        raise ValueError('Increase minimum partition size to at least 3.')

    num0 = x
    num1 = num0 * (j - 1)
    num2 = num1 * (j - 2)

    den0 = n
    den1 = den0 * (n - 1)
    den2 = den1 * (n - 2)

    b0 = np.sum(num0) / den0
    b1 = np.sum(num1) / den1
    b2 = np.sum(num2) / den2

    with np.errstate(divide='ignore', invalid='ignore'):
        c = (2 * b1 - b0) / (3 * b2 - b0) - np.log(2) / np.log(3)

    xi = -7.8590 * c - 2.9554 * c ** 2
    gamma_term = gamma(1 - xi)
    sigma = (b0 - 2 * b1) * xi / gamma_term / (1 - 2 ** xi)
    mu = b0 - sigma / xi * (gamma_term - 1)

    sigma = np.nan_to_num(sigma, nan=0).clip(min=1e-3)
    return mu, sigma, xi


_supported_impurity_metrics = {
    'log_score': _log_score,
    'crps': _crps_score
}


class GenExtreme:
    __slots__ = (
        '_impurity_func'
    )

    def __init__(self, impurity_metric='log_score'):
        self._impurity_func = _supported_impurity_metrics[impurity_metric]

    @staticmethod
    def estimate(target):
        mu, sigma, xi = _pwm_estimate(target)
        params = np.stack([mu, sigma, xi], axis=0)[..., np.newaxis]
        return params

    @staticmethod
    def score_func(parent, left, right):
        parent_score = _kolmogorov_smirnov(parent, GenExtreme.estimate(parent))
        left_score = _kolmogorov_smirnov(left, GenExtreme.estimate(left))
        right_score = _kolmogorov_smirnov(right, GenExtreme.estimate(right))

        return parent_score - left_score - right_score
