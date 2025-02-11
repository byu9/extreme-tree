#!/usr/bin/env python3
import os
from itertools import product
from multiprocessing import Pool

import numpy as np
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

    print_items = ','.join([
        f'{ensemble_size=:}',
        f'{resample_ratio =:}',
        f'{max_n_splits=:}',

        f'mu_mean={mu.mean()}',
        f'sigma_mean={sigma.mean()}',
        f'xi_mean={xi.mean()}',

        f'mu_std={mu.std()}',
        f'sigma_std={sigma.std()}',
        f'xi_std={xi.std()}',

        f'mu_max={mu.max()}',
        f'sigma_max={sigma.max()}',
        f'xi_max={xi.max()}',

        f'mu_min={mu.min()}',
        f'sigma_min={sigma.min()}',
        f'xi_min={xi.min()}',

        f'mu_median={np.quantile(mu, q=0.5)}',
        f'sigma_median={np.quantile(sigma, q=0.5)}',
        f'xi_median={np.quantile(xi, 0.5)}',

        f'mu_upper={np.quantile(mu, 0.75)}',
        f'sigma_upper={np.quantile(sigma, 0.75)}',
        f'xi_upper={np.quantile(xi, 0.75)}',

        f'mu_lower={np.quantile(mu, 0.25)}',
        f'sigma_lower={np.quantile(sigma, 0.25)}',
        f'xi_lower={np.quantile(xi, 0.25)}',
    ])

    return print_items


def _fit_predict_all():
    ensemble_sizes = [1, 10, 100, 200, 500, 800, 1000]
    resample_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    max_splits = [1, 3, 6, 10, 12, 16, 20]

    with Pool() as pool:
        param_all_runs = product(ensemble_sizes, resample_ratios, max_splits)
        print_items = pool.starmap(_fit_predict_once, param_all_runs)

    print('\n'.join(print_items))


def _main():
    _fit_predict_all()


if __name__ == '__main__':
    _main()
