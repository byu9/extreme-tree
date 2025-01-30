from dataclasses import dataclass

import numpy as np
from scipy.special import gamma
from scipy.stats import genextreme


@dataclass(init=False)
class _DistParams:
    # Distribution parameters
    # Location - mu
    # Scale - sigma
    # Shape - xi
    mu: ...
    sigma: ...
    xi: ...


class GenExtreme:
    @staticmethod
    def compute_estimate(target):
        params = _DistParams()
        params.mu, params.sigma, params.xi = _pwm_estimate(target)
        return params

    @staticmethod
    def compute_score(params: _DistParams, target):
        return _log_score(mu=params.mu, sigma=params.sigma, xi=params.xi, y=target).sum()

    @staticmethod
    def forward_prop(leaves):
        mu_hat = sum(leaf.pi * leaf.params.mu for leaf in leaves)
        sigma_hat = sum(leaf.pi * leaf.params.sigma for leaf in leaves)
        xi_hat = sum(leaf.pi * leaf.params.xi for leaf in leaves)
        prediction = genextreme(loc=mu_hat, scale=sigma_hat, c=-xi_hat)
        return prediction


def _log_score(mu, sigma, xi, y):
    logpdf = genextreme.logpdf(y, loc=mu, scale=sigma, c=-xi)

    logpdf_outside_support = -1000
    np.nan_to_num(logpdf, nan=logpdf_outside_support, neginf=logpdf_outside_support, copy=False)

    return -logpdf


def _pwm_estimate(target):
    # Probability weighted moment estimate
    # See https://doi.org/10.1080/00401706.1985.10488049
    x = np.sort(target)
    n = len(target)
    j = np.arange(1, n + 1)

    if n <= 2:
        raise ValueError('Increase minimum partition size to at least 2')

    num0 = x
    num1 = num0 * (j - 1)
    num2 = num1 * (j - 2)

    den0 = n
    den1 = den0 * (n - 1)
    den2 = den1 * (n - 2)

    b0 = np.sum(num0, axis=-1, keepdims=True) / den0
    b1 = np.sum(num1, axis=-1, keepdims=True) / den1
    b2 = np.sum(num2, axis=-1, keepdims=True) / den2

    c = (2 * b1 - b0) / (3 * b2 - b0) - np.log(2) / np.log(3)

    xi = -7.8590 * c - 2.9554 * c ** 2
    gamma_term = gamma(1 - xi)
    sigma = (b0 - 2 * b1) * xi / gamma_term / (1 - 2 ** xi)
    mu = b0 - sigma / xi * (gamma_term - 1)

    return mu, sigma, xi
