import numpy as np
from scipy.stats import genpareto


def _log_score(target, params):
    mu, sigma, xi = params
    score = -genpareto.logpdf(target, scale=sigma, c=-xi)
    score = np.nan_to_num(score, copy=False, nan=0, posinf=0)
    return score.sum()


def _pwm_estimate(target):
    # Probability weighted moment estimate
    # See https://doi.org/10.2307/1269343
    target = np.ravel(target)

    x = np.sort(target)
    n = len(target)
    j = 1 + np.arange(n)

    if not n >= 2:
        raise ValueError('Increase minimum partition size to at least 2.')

    num0 = x
    num1 = num0 * (n - j)

    den0 = n
    den1 = den0 * (n - 1)

    a0 = np.sum(num0) / den0
    a1 = np.sum(num1) / den1

    sigma = 2 * a0 * a1 / (a0 - 2 * a1)
    xi = 2 - a0 / (a0 - 2 * a1)

    return sigma, xi


_supported_impurity_metrics = {
    'log_score': _log_score,
}


class GenPareto:
    __slots__ = (
        '_impurity_func'
    )

    def __init__(self, impurity_metric='log_score'):
        self._impurity_func = _supported_impurity_metrics[impurity_metric]

    @staticmethod
    def estimate(target):
        sigma, xi = _pwm_estimate(target)
        params = np.stack([sigma, xi], axis=0)[..., np.newaxis]
        return params

    def impurity(self, target, params):
        return self._impurity_func(target, params)
