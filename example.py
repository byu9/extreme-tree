#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from scipy.special import digamma
from scipy.special import gamma
from scipy.stats import genextreme
from scipy.stats import norm

from extreme_tree import ExtremeForest

np.random.seed(123)


def _create_scenario():
    x = np.linspace(0, np.pi, 1000).reshape(-1, 1)
    mu = np.cos(1.23 * x) + 0.3 * np.cos(4.56 * x)
    xi = 0.3 * np.cos(5.67 * x)
    sigma = 1 + 0.1 * np.cos(6.78 * x)
    dist = _param_to_dist(mu, sigma, xi)
    y = dist.rvs()
    return x, y, dist


def _param_to_dist(mu, sigma, xi):
    return genextreme(loc=mu, scale=sigma, c=-xi)


def _param_from_dist(dist):
    return dist.kwds['loc'], dist.kwds['scale'], -dist.kwds['c']


def _save_target(x, y, save_as):
    pd.DataFrame({'y': y.ravel(), 'x': x.ravel()}).to_csv(save_as, index_label='Index')


def _save_dist(x, dist, save_as):
    mu, sigma, xi = _param_from_dist(dist)
    dist_data = pd.DataFrame({
        'x': x.ravel(),
        'mu': mu.ravel(),
        'sigma': sigma.ravel(),
        'xi': xi.ravel(),
        'mean': dist.mean().ravel(),
        'lo': dist.ppf(0.05).ravel(),
        'hi': dist.ppf(0.95).ravel()
    })
    dist_data.to_csv(save_as, index_label='Index')


def _dist_to_quantiles(dist, quantiles):
    predictions = pd.DataFrame({
        q: dist.ppf(q).ravel()
        for q in quantiles
    })
    return predictions


def _quantile_regression(x, y, quantiles):
    model = RandomForestQuantileRegressor(default_quantiles=quantiles)
    model.fit(x, y.ravel())
    predictions = model.predict(x)
    predictions = pd.DataFrame(predictions, columns=quantiles)
    return predictions


def _save_quantile_regression(x, quantiles, save_as):
    quantiles = quantiles.copy()
    quantiles.index = x.ravel()
    quantiles.to_csv(save_as, index_label='x')


def _compare_quantiles(x, y, dist, pred_dist, save_as):
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    dist_quantiles = _dist_to_quantiles(dist, quantiles=quantiles)
    pred_dist_quantiles = _dist_to_quantiles(pred_dist, quantiles=quantiles)
    qr_quantiles = _quantile_regression(x, y, quantiles=quantiles)

    pred_dist_error = (pred_dist_quantiles - dist_quantiles).abs().mean()
    qr_error = (qr_quantiles - dist_quantiles).abs().mean()

    report = pd.DataFrame({
        'pred_dist': pred_dist_error,
        'qr': qr_error
    }, index=['pred_dist', 'qr'])

    report.to_csv(save_as, index_label='Index')


def _plot_dist_against_target(x, y, dist):
    plt.figure()
    plt.plot(x, dist.mean(), label=r'E[GEV]')
    plt.plot(x, dist.ppf(0.95), label='0.95')
    plt.plot(x, dist.ppf(0.05), label='0.05')
    plt.plot(x, y, label='target')
    plt.legend()


def _get_fisher(dist):
    _, sigma, xi = _param_from_dist(dist)

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


def _get_param_bounds(x, dist):
    mu, sigma, xi = _param_from_dist(dist)
    fisher = _get_fisher(dist)
    inv_fisher = np.linalg.inv(fisher)

    mu_dist = norm(loc=mu, scale=inv_fisher[..., 0, 0].reshape(mu.shape))
    sigma_dist = norm(loc=sigma, scale=inv_fisher[..., 1, 1].reshape(sigma.shape))
    xi_dist = norm(loc=xi, scale=inv_fisher[..., 2, 2].reshape(xi.shape))

    report = pd.DataFrame({
        'x': x.ravel(),
        'mu_lo': mu_dist.ppf(0.1).ravel(),
        'mu_hi': mu_dist.ppf(0.9).ravel(),
        'sigma_lo': sigma_dist.ppf(0.1).ravel(),
        'sigma_hi': sigma_dist.ppf(0.9).ravel(),
        'xi_lo': xi_dist.ppf(0.1).ravel(),
        'xi_hi': xi_dist.ppf(0.9).ravel(),
    })
    return report


def _save_param_bounds(x, dist, save_as):
    report = _get_param_bounds(x, dist)
    report.to_csv(save_as, index_label='Index')


def _plot_dist_comparison(x, dist, pred_dist):
    bounds = _get_param_bounds(x, dist)
    mu, sigma, xi = _param_from_dist(dist)
    pred_mu, pred_sigma, pred_xi = _param_from_dist(pred_dist)

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    fill_params = {'label': 'CRB', 'alpha': 0.5, 'color': 'black'}

    ax1.plot(x.ravel(), mu.ravel(), label=r'$\mu$')
    ax1.plot(x.ravel(), pred_mu.ravel(), label=r'$\hat\mu$')
    ax1.fill_between(
        x.ravel(),
        bounds['mu_lo'].to_numpy().ravel(),
        bounds['mu_hi'].to_numpy().ravel(),
        **fill_params
    )

    ax2.plot(x.ravel(), sigma.ravel(), label=r'$\sigma$')
    ax2.plot(x.ravel(), pred_sigma.ravel(), label=r'$\hat\sigma$')
    ax2.fill_between(
        x.ravel(),
        bounds['sigma_lo'].to_numpy().ravel(),
        bounds['sigma_hi'].to_numpy().ravel(),
        **fill_params
    )

    ax3.plot(x.ravel(), xi.ravel(), label=r'$\xi$')
    ax3.plot(x.ravel(), pred_xi.ravel(), label=r'$\hat\xi$')
    ax3.fill_between(
        x.ravel(),
        bounds['xi_lo'].to_numpy().ravel(),
        bounds['xi_hi'].to_numpy().ravel(),
        **fill_params
    )

    ax1.legend()
    ax2.legend()
    ax3.legend()


def _fit_extreme_forest(x, y):
    model = ExtremeForest(ensemble_size=50, min_score=0.0001, min_partition_size=20)
    model.fit(x, y)
    pred_mu, pred_sigma, pred_xi = model.predict(x)
    return _param_to_dist(pred_mu, pred_sigma, pred_xi)


def _plot_quantile_comparison(dist, pred_dist, qr_quantiles):
    quantiles = qr_quantiles.columns
    dist_quantiles = _dist_to_quantiles(dist, quantiles=quantiles)
    pred_dist_quantiles = _dist_to_quantiles(pred_dist, quantiles=quantiles)

    for quantile in quantiles:
        plt.figure()
        plt.plot(dist_quantiles[quantile], label='dist')
        plt.plot(pred_dist_quantiles[quantile], label='pred_dist')
        plt.plot(qr_quantiles[quantile], label='qr')
        plt.legend()
        plt.title(f'{quantile=:}')


def _save_quantile_comparison_report(dist, pred_dist, qr_quantiles, save_as, clip_lower=None):
    quantiles = qr_quantiles.columns
    dist_quantiles = _dist_to_quantiles(dist, quantiles=quantiles)
    pred_dist_quantiles = _dist_to_quantiles(pred_dist, quantiles=quantiles)

    pred_dist_err = (dist_quantiles - pred_dist_quantiles).clip(lower=clip_lower).abs().mean()
    qr_err = (dist_quantiles - qr_quantiles).clip(lower=clip_lower).abs().mean()

    report = pd.DataFrame({'pred_dist_err': pred_dist_err, 'qr_err': qr_err})
    report.to_csv(save_as, index_label='quantile')


def _save_quantiles(x, dist, quantiles, save_as):
    dist_quantiles = _dist_to_quantiles(dist, quantiles=quantiles)
    report = pd.DataFrame(dist_quantiles)
    report['x'] = x.ravel()
    report.to_csv(save_as, index_label='Index')


def _main():
    x, y, dist = _create_scenario()
    pred_dist = _fit_extreme_forest(x, y)

    _plot_dist_against_target(x, y, dist=pred_dist)
    _plot_dist_comparison(x, dist=dist, pred_dist=pred_dist)

    quantiles = [0.1, 0.5, 0.9, 0.99, 0.999, 0.999999]
    qr_quantiles = _quantile_regression(x, y, quantiles=quantiles)
    _save_quantile_regression(x, qr_quantiles, save_as='example_qr_quantiles.csv')
    _plot_quantile_comparison(dist=dist, pred_dist=pred_dist, qr_quantiles=qr_quantiles)

    _save_target(x, y, save_as='example_target.csv')
    _save_dist(x, dist, save_as='example_dist.csv')
    _save_param_bounds(x, dist=dist, save_as='example_param_bounds.csv')
    _save_dist(x, pred_dist, save_as='example_pred_dist.csv')
    _save_quantile_comparison_report(
        dist=dist, pred_dist=pred_dist, qr_quantiles=qr_quantiles, clip_lower=None,
        save_as='example_quantiles.csv'
    )
    _save_quantile_comparison_report(
        dist=dist, pred_dist=pred_dist, qr_quantiles=qr_quantiles, clip_lower=0,
        save_as='example_quantiles_underforecasted.csv'
    )
    _save_quantiles(x, dist=dist, quantiles=quantiles, save_as='example_dist_quantiles.csv')
    _save_quantiles(x, dist=pred_dist, quantiles=quantiles,
                    save_as='example_pred_dist_quantiles.csv')

    plt.show()


_main()
