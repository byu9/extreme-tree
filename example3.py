#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genextreme

from extreme_tree import ExtremeForest

np.random.seed(123)


def _create_block_maxima():
    cluster_sizes = [300, 300, 300]

    y1 = genextreme(loc=0, scale=1, c=0).rvs(cluster_sizes[0])
    y2 = genextreme(loc=0, scale=2, c=0).rvs(cluster_sizes[1])
    y3 = genextreme(loc=0, scale=1, c=0.2).rvs(cluster_sizes[2])

    y = np.concat([y1, y2, y3]).reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)

    return x, y


def _fit_predict_once():
    x, y = _create_block_maxima()
    model = ExtremeForest(ensemble_size=10, resample_ratio=0.8, min_impurity_drop_ratio=0.8)
    model.fit(x, y)
    mu, sigma, xi = model.predict(x)
    return x, y, mu, sigma, xi


def _main():
    x, y, mu, sigma, xi = _fit_predict_once()
    plt.plot(x, mu, label='mu')
    plt.plot(x, sigma, label='sigma')
    plt.plot(x, xi, label='xi')
    plt.legend()
    plt.show()


_main()
