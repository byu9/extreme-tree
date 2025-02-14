#!/usr/bin/env python3
import os
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm

from extreme_tree import ExtremeForest

os.environ["TQDM_DISABLE"] = "True"
np.random.seed(123)


def _create_block_maxima(n_samples, block_size=50):
    x = np.arange(n_samples)
    z = norm(loc=0, scale=1).rvs(size=(n_samples, block_size))
    y = np.max(z, axis=-1)
    return x.reshape(-1, 1), y.reshape(-1, 1)


def _fit_predict_once(ensemble_size, resample_ratio, max_n_splits):
    x, y = _create_block_maxima(n_samples=1000)
    model = ExtremeForest(ensemble_size=ensemble_size, resample_ratio=resample_ratio,
                          max_n_splits=max_n_splits)
    model.fit(x, y)
    mu, sigma, xi = model.predict(x)

    report = pd.DataFrame({
        'ensemble_size': ensemble_size,
        'resample_ratio': resample_ratio,
        'max_n_splits': max_n_splits,

        'mu_mean': mu.mean(),
        'mu_std': mu.std(),
        'mu_max': mu.max(),
        'mu_min': mu.min(),
        'mu_upper': np.quantile(mu, q=0.75),
        'mu_lower': np.quantile(mu, q=0.25),
        'mu_median': np.quantile(mu, q=0.5),

        'sigma_mean': sigma.mean(),
        'sigma_std': sigma.std(),
        'sigma_max': sigma.max(),
        'sigma_min': sigma.min(),
        'sigma_upper': np.quantile(sigma, q=0.75),
        'sigma_lower': np.quantile(sigma, q=0.25),
        'sigma_median': np.quantile(sigma, q=0.5),

        'xi_mean': xi.mean(),
        'xi_std': xi.std(),
        'xi_max': xi.max(),
        'xi_min': xi.min(),
        'xi_upper': np.quantile(xi, q=0.75),
        'xi_lower': np.quantile(xi, q=0.25),
        'xi_median': np.quantile(xi, q=0.5),
    }, index=[1])

    return report


def _fit_predict_all():
    ensemble_sizes = range(1, 10)
    resample_ratios = [0.1, 0.5, 0.9]
    max_n_splits = [1, 4, 8, 12, 20]

    param_all_runs = product(ensemble_sizes, resample_ratios, max_n_splits)

    results = list()
    for param in param_all_runs:
        results.append(_fit_predict_once(*param))

    report = pd.concat(results, axis='index')

    return report


def _main():
    report = _fit_predict_all()
    report.to_csv('example2.csv')


if __name__ == '__main__':
    _main()
