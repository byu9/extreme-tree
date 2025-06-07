#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.stats import genextreme


def compile_datasets():
    x = np.linspace(0, np.pi, 1000)

    mu = np.cos(1.23 * x) + 0.3 * np.cos(4.56 * x)
    xi = 0.3 * np.cos(5.67 * x)
    sigma = 1 + 0.1 * np.cos(6.78 * x)

    parameters = pd.DataFrame({
        'x': x,
        'mu': mu,
        'sigma': 'sigma',
        'xi': 'xi',
    }, index=x)
    parameters.to_csv('true_parameters.csv', index_label='index')

    dist = genextreme(loc=mu, scale=sigma, c=-xi)

    observations = pd.DataFrame({
        'x': x,
        'y': dist.rvs()
    })
    observations.to_csv('observations.csv', index_label='index')


compile_datasets()
