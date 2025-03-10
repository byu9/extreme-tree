#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.special import gamma
from scipy.stats import genextreme
from scipy.stats import norm

from extreme_tree import ExtremeForest

np.random.seed(123)


def _create_scenario():
    x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
    mu = 10 * np.sin(x) + 2 * np.cos(3 * x)
    sigma = abs(np.cos(x)) + 0.1
    xi = 0.8 * np.cos(x)

    return x, mu, sigma, xi


def _gev_inverse_fisher_information(mu, sigma, xi):
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

    (
        fisher11, fisher12, fisher13, fisher22,
        fisher23, fisher23, fisher33
    ) = np.broadcast_arrays(fisher11, fisher12, fisher13, fisher22,
                            fisher23, fisher23, fisher33)

    fisher = np.asarray([
        [fisher11, fisher12, fisher13],
        [fisher12, fisher22, fisher23],
        [fisher13, fisher23, fisher33]
    ])

    fisher = np.positive(np.nan, where=xi_zero, out=fisher)
    fisher = np.moveaxis(fisher, [0, 1], [-2, -1])
    inv_fisher = np.linalg.inv(fisher)

    return inv_fisher


def _save_dist(x, mu, sigma, xi, save_as):
    dist = genextreme(loc=mu, scale=sigma, c=-xi)
    dist_data = pd.DataFrame({
        'x': x.ravel(),
        'mu': mu.ravel(),
        'sigma': sigma.ravel(),
        'xi': xi.ravel(),
        'mean': dist.mean().ravel(),
        'lo': dist.ppf(0.1).ravel(),
        'hi': dist.ppf(0.9).ravel()
    })
    dist_data.to_csv(save_as, index_label='Index')


def _save_inv_fisher(x, mu, sigma, xi, save_as):
    inv_fisher = _gev_inverse_fisher_information(mu, sigma, xi)
    mu_dist = norm(loc=mu, scale=inv_fisher[..., 0, 0].reshape(mu.shape))
    sigma_dist = norm(loc=sigma, scale=inv_fisher[..., 1, 1].reshape(sigma.shape))
    xi_dist = norm(loc=xi, scale=inv_fisher[..., 2, 2].reshape(xi.shape))

    inv_fisher_data = pd.DataFrame({
        'x': x.ravel(),
        'mu_lo': mu_dist.ppf(0.1).ravel(),
        'mu_hi': mu_dist.ppf(0.9).ravel(),
        'sigma_lo': sigma_dist.ppf(0.1).ravel(),
        'sigma_hi': sigma_dist.ppf(0.9).ravel(),
        'xi_lo': xi_dist.ppf(0.1).ravel(),
        'xi_hi': xi_dist.ppf(0.9).ravel(),
    })

    inv_fisher_data.to_csv(save_as, index_label='Index')
    return inv_fisher_data


def _create_dist(mu, sigma, xi):
    return genextreme(loc=mu, scale=sigma, c=-xi)


def _main():
    x, mu, sigma, xi = _create_scenario()
    dist = _create_dist(mu, sigma, xi)
    y = dist.rvs()
    pd.DataFrame({'y': y.ravel(), 'x': x.ravel()}).to_csv('example_target.csv', index_label='Index')

    inv_fisher_data = _save_inv_fisher(x, mu, sigma, xi, save_as='example_inv_fisher.csv')

    model = ExtremeForest(ensemble_size=20, min_score=0.005, min_partition_size=50)
    model.fit(x, y)
    pred_mu, pred_sigma, pred_xi = model.predict(x)

    _save_dist(x, mu, sigma, xi, save_as='example_dist.csv')
    _save_dist(x, pred_mu, pred_sigma, pred_xi, save_as='example_pred.csv')

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(x, mu, label=r'$\mu$')
    ax1.plot(x, pred_mu, label=r'$\hat\mu$')
    ax1.fill_between(
        inv_fisher_data['x'], inv_fisher_data['mu_lo'], inv_fisher_data['mu_hi'],
        label='CRB', alpha=0.5, color='black'
    )
    ax1.legend()

    ax2.plot(x, sigma, label=r'$\sigma$')
    ax2.plot(x, pred_sigma, label=r'$\hat\sigma$')
    ax2.fill_between(
        inv_fisher_data['x'], inv_fisher_data['sigma_lo'], inv_fisher_data['sigma_hi'],
        label='CRB', alpha=0.5, color='black'
    )
    ax2.legend()

    ax3.plot(x, xi, label=r'$\xi$')
    ax3.plot(x, pred_xi, label=r'$\hat\xi$')
    ax3.fill_between(
        inv_fisher_data['x'], inv_fisher_data['xi_lo'], inv_fisher_data['xi_hi'],
        label='CRB', alpha=0.5, color='black'
    )
    ax3.legend()

    pred_dist = _create_dist(pred_mu, pred_sigma, pred_xi)

    plt.figure()
    plt.plot(x, pred_dist.mean(), label=r'E[GEV]')
    plt.plot(x, pred_dist.ppf(0.9), label='0.9')
    plt.plot(x, pred_dist.ppf(0.1), label='0.1')
    plt.plot(x, y, label='target')
    plt.legend()

    plt.show()


_main()
