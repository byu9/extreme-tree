import numpy as np
from scipy.special import gamma
from scipy.stats import genextreme


class GenExtreme:
    __slots__ = ()

    @staticmethod
    def estimate(target):
        mu, sigma, xi = _pwm_estimate(target)
        return np.stack([mu, sigma, xi], axis=0)

    @staticmethod
    def convert_to_scipy(params):
        mu, sigma, xi = params
        return genextreme(loc=mu, scale=sigma, c=-xi)


def _pwm_estimate(target):
    # Probability weighted moment estimate
    # See https://doi.org/10.1080/00401706.1985.10488049
    x = np.sort(target, axis=-1)
    n = target.shape[-1]
    j = np.arange(1, n + 1)

    if n <= 2:
        raise ValueError('Increase minimum partition size to at least 3.')

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
