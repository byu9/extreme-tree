#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genextreme

from extreme_tree import ExtremeForest

np.random.seed(123)


def _create_scenario():
    x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
    mu = np.sin(x) + 0.2 * np.cos(3 * x)
    sigma = 1 - x / 5 / np.pi
    xi = 0.2 * np.cos(x)

    return x, mu, sigma, xi


def _save_dist(x, mu, sigma, xi, save_as):
    dist_data = pd.DataFrame({
        'x': x.ravel(),
        'mu': mu.ravel(),
        'sigma': sigma.ravel(),
        'xi': xi.ravel()
    })
    dist_data.to_csv(save_as, index_label='Index')


def _create_dist(mu, sigma, xi):
    return genextreme(loc=mu, scale=sigma, c=-xi)


def _main():
    x, mu, sigma, xi = _create_scenario()
    dist = _create_dist(mu, sigma, xi)
    y = dist.rvs()

    model = ExtremeForest(ensemble_size=20, min_score=0.005)
    model.fit(x, y)
    pred_mu, pred_sigma, pred_xi = model.predict(x)

    _save_dist(x, mu, sigma, xi, save_as='example_dist.csv')
    _save_dist(x, pred_mu, pred_sigma, pred_xi, save_as='example_pred.csv')
    pd.Series(y.ravel(), name='y').to_csv('example_target.csv')

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(x, mu, label=r'$\mu$')
    ax1.plot(x, pred_mu, label=r'$\hat\mu$')
    ax1.legend()

    ax2.plot(x, sigma, label=r'$\sigma$')
    ax2.plot(x, pred_sigma, label=r'$\hat\sigma$')
    ax2.legend()

    ax3.plot(x, xi, label=r'$\xi$')
    ax3.plot(x, pred_xi, label=r'$\hat\xi$')
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
