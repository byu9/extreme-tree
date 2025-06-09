#!/usr/bin/env python3
import numpy as np
from scipy.special import digamma
from scipy.special import gamma
from scipy.stats import norm
import pandas as pd


def gev_fisher(mu, sigma, xi):
    xi_zero = np.isclose(xi, 0)
    one_over_xi = np.divide(1, xi, where=~xi_zero)
    one_over_xi_sq = one_over_xi ** 2

    p = (1 + xi) ** 2 * gamma(1 + 2 * xi)
    q = gamma(2 + xi) * (digamma(1 + xi) + 1 + one_over_xi)
    r = gamma(2 + xi)

    fisher11 = p / sigma ** 2
    fisher12 = (r - p) * one_over_xi / sigma ** 2
    fisher13 = (q - p * one_over_xi) * one_over_xi / sigma
    fisher22 = (1 - 2 * r + p) * one_over_xi_sq / sigma ** 2
    fisher23 = one_over_xi_sq / sigma * (one_over_xi * (1 - r + p) + 1 - np.euler_gamma - q)
    fisher33 = one_over_xi_sq * (
            np.pi ** 2 / 6
            + (1 - np.euler_gamma + one_over_xi) ** 2
            - 2 * q * one_over_xi
            + p * one_over_xi_sq
    )
    fisher = (fisher11, fisher12, fisher13, fisher22, fisher23, fisher23, fisher33)
    target_shape = np.broadcast(*fisher).shape

    fisher11 = np.broadcast_to(fisher11, target_shape)
    fisher12 = np.broadcast_to(fisher12, target_shape)
    fisher13 = np.broadcast_to(fisher13, target_shape)
    fisher22 = np.broadcast_to(fisher22, target_shape)
    fisher23 = np.broadcast_to(fisher23, target_shape)
    fisher33 = np.broadcast_to(fisher33, target_shape)

    fisher = np.asarray([
        [fisher11, fisher12, fisher13],
        [fisher12, fisher22, fisher23],
        [fisher13, fisher23, fisher33]
    ])

    fisher = np.positive(np.nan, where=xi_zero, out=fisher)
    fisher = np.moveaxis(fisher, [0, 1], [-2, -1])
    return fisher


def write_cramer_rao_bounds(mu, sigma, xi):
    fisher = gev_fisher(mu, sigma, xi)
    inv_fisher = np.linalg.inv(fisher)

    mu_dist = norm(loc=mu, scale=inv_fisher[..., 0, 0].reshape(mu.shape))
    sigma_dist = norm(loc=sigma, scale=inv_fisher[..., 1, 1].reshape(sigma.shape))
    xi_dist = norm(loc=xi, scale=inv_fisher[..., 2, 2].reshape(xi.shape))

    bounds = pd.DataFrame({
        'mu_lo': mu_dist.ppf(0.05).ravel(),
        'mu_hi': mu_dist.ppf(0.95).ravel(),
        'sigma_lo': sigma_dist.ppf(0.05).ravel(),
        'sigma_hi': sigma_dist.ppf(0.95).ravel(),
        'xi_lo': xi_dist.ppf(0.05).ravel(),
        'xi_hi': xi_dist.ppf(0.95).ravel(),
    })

    bounds.to_csv('093-run_mvue_on_synthetic_bounds.csv', index_label='index')


def read_true_parameters():
    dataframe = pd.read_csv('datasets/synthetic/true_parameters.csv', index_col='index')
    return dataframe


def main():
    true_parameters = read_true_parameters()

    mu = true_parameters['mu']
    sigma = true_parameters['sigma']
    xi = true_parameters['xi']

    write_cramer_rao_bounds(mu, sigma, xi)


if __name__ == '__main__':
    main()
