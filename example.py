#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.special import gamma
from scipy.special import zeta
from scipy.stats import genextreme
from scipy.stats import norm

from extreme_tree import ExtremeForest

np.random.seed(123)


def _create_scenario():
    x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
    mu = np.sin(x) + 0.2 * np.cos(3 * x)
    sigma = 1 - x / 5 / np.pi
    xi = 0.2 * np.cos(x)

    return x, mu, sigma, xi


def _gev_inverse_fisher_information(mu, sigma, xi):
    xi_zero = np.isclose(xi, 0)

    one_over_xi = np.divide(1, xi, where=~xi_zero)
    one_over_xi_sq = one_over_xi ** 2

    p = (1 + xi) ** 2 * gamma(1 + 2 * xi)
    q = gamma(2 + xi) * (digamma(1 + xi) + 1 + one_over_xi)
    r = gamma(2 + xi)

    fisher11 = p / sigma ** 2
    fisher12 = (r - p) * one_over_xi / sigma
    fisher13 = (q - p * one_over_xi) * one_over_xi / sigma
    fisher22 = (1 - 2 * r + p) * one_over_xi_sq
    fisher23 = one_over_xi_sq * (one_over_xi * (1 - r + p) + 1 - np.euler_gamma - q)
    fisher33 = one_over_xi_sq * (
            np.pi ** 2 / 6
            + (1 - np.euler_gamma + one_over_xi) ** 2
            - 2 * q * one_over_xi
            + p * one_over_xi_sq
    )

    # zero cases
    fisher12 = np.divide(np.euler_gamma - 1, sigma, where=xi_zero, out=fisher12)
    fisher13 = np.divide(np.euler_gamma - np.euler_gamma ** 2 / 2 - np.pi ** 2 / 12, sigma,
                         where=xi_zero, out=fisher13)
    fisher22[xi_zero] = np.pi ** 2 / 6 + (np.euler_gamma - 1) ** 2
    fisher23[xi_zero] = (
            np.euler_gamma ** 2 * 3 / 2
            - zeta(3)
            - np.euler_gamma * np.pi ** 2 / 4
            - np.euler_gamma
            - np.euler_gamma ** 3 / 2
            + np.pi ** 2 / 4
    )
    fisher33[xi_zero] = (
            2 * zeta(3) * (np.euler_gamma - 1)
            - np.pi ** 2 / 2 * np.euler_gamma
            + np.euler_gamma ** 2
            - np.euler_gamma ** 3
            + np.euler_gamma ** 4 / 4
            + np.pi ** 2 / 6
            + 3 * np.pi ** 4 / 80
            + np.euler_gamma ** 2 * np.pi ** 2 / 4
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

    fisher = np.moveaxis(fisher, [0, 1], [-2, -1])

    inv_fisher = np.linalg.inv(fisher)

    return inv_fisher


def _save_dist(x, mu, sigma, xi, save_as):
    dist_data = pd.DataFrame({
        'x': x.ravel(),
        'mu': mu.ravel(),
        'sigma': sigma.ravel(),
        'xi': xi.ravel()
    })
    dist_data.to_csv(save_as, index_label='Index')


def _save_inv_fisher(mu, sigma, xi, save_as):
    inv_fisher = _gev_inverse_fisher_information(mu, sigma, xi)
    inv_fisher_data = pd.DataFrame({
        'mu': inv_fisher[..., 0, 0].ravel(),
        'sigma': inv_fisher[..., 1, 1].ravel(),
        'xi': inv_fisher[..., 2, 2].ravel(),
    })

    inv_fisher_data.to_csv(save_as, index_label='Index')
    return inv_fisher_data


def _create_dist(mu, sigma, xi):
    return genextreme(loc=mu, scale=sigma, c=-xi)


def _main():
    x, mu, sigma, xi = _create_scenario()
    dist = _create_dist(mu, sigma, xi)
    y = dist.rvs()

    inv_fisher_data = _save_inv_fisher(mu, sigma, xi, save_as='example_inv_fisher.csv')
    mu_dist = norm(loc=mu, scale=inv_fisher_data['mu'].to_numpy().reshape(y.shape))
    sigma_dist = norm(loc=sigma, scale=inv_fisher_data['sigma'].to_numpy().reshape(y.shape))
    xi_dist = norm(loc=xi, scale=inv_fisher_data['xi'].to_numpy().reshape(y.shape))

    model = ExtremeForest(ensemble_size=20, min_score=0.005, min_partition_size=50)
    model.fit(x, y)
    pred_mu, pred_sigma, pred_xi = model.predict(x)

    _save_dist(x, mu, sigma, xi, save_as='example_dist.csv')
    _save_dist(x, pred_mu, pred_sigma, pred_xi, save_as='example_pred.csv')
    pd.Series(y.ravel(), name='y').to_csv('example_target.csv')

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(x, mu, label=r'$\mu$')
    ax1.plot(x, pred_mu, label=r'$\hat\mu$')
    ax1.fill_between(
        x.ravel(), mu_dist.ppf(0.05).ravel(), mu_dist.ppf(0.95).ravel(),
        label='CRB', alpha=0.5, color='black'
    )
    ax1.legend()

    ax2.plot(x, sigma, label=r'$\sigma$')
    ax2.plot(x, pred_sigma, label=r'$\hat\sigma$')
    ax2.fill_between(
        x.ravel(), sigma_dist.ppf(0.05).ravel(), sigma_dist.ppf(0.95).ravel(),
        label='CRB', alpha=0.5, color='black'
    )
    ax2.legend()

    ax3.plot(x, xi, label=r'$\xi$')
    ax3.plot(x, pred_xi, label=r'$\hat\xi$')
    ax3.fill_between(
        x.ravel(), xi_dist.ppf(0.05).ravel(), xi_dist.ppf(0.95).ravel(),
        label='CRB', alpha=0.5, color='black'
    )
    ax3.legend()

    pred_dist = _create_dist(pred_mu, pred_sigma, pred_xi)

    plt.figure()
    plt.plot(x, pred_dist.mean(), label=r'E[GEV]')
    plt.plot(x, pred_dist.ppf(0.95), label='0.95')
    plt.plot(x, pred_dist.ppf(0.05), label='0.05')
    plt.plot(x, y, label='target')
    plt.legend()

    plt.show()


_main()
