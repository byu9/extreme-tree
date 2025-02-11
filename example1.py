#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from extreme_tree import ExtremeForest

np.random.seed(123)


def _create_block_maxima(n_samples, block_size=50):
    x = np.arange(n_samples)
    z = norm(loc=0, scale=1).rvs(size=(n_samples, block_size))
    y = np.max(z, axis=-1)
    return x.reshape(-1, 1), y.reshape(-1, 1)


def _fit_predict_once():
    x, y = _create_block_maxima(n_samples=200)
    model = ExtremeForest(ensemble_size=60, resample_ratio=0.9, alpha=0.01)
    model.fit(x, y)
    mu, sigma, xi = model.predict(x)
    return x, y, mu, sigma, xi


def _main():
    x, y, mu, sigma, xi = _fit_predict_once()
    plt.plot(x, y, label='y')
    plt.plot(x, mu, label='mu')
    plt.plot(x, sigma, label='sigma')
    plt.plot(x, xi, label='xi')
    plt.legend()
    plt.show()


_main()
