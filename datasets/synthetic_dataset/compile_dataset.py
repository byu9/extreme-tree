import numpy as np
import pandas as pd
from scipy.stats import norm


def _save_feature(dataset, save_as):
    dataset.drop(columns=['Target']).to_csv(save_as, index_label='Index')


def _save_target(dataset, save_as):
    dataset[['Target']].to_csv(save_as, index_label='Index')


def _compile_dataset():
    n_samples = 2000
    n_blocks = 40

    x = np.linspace(0, 1, n_samples)
    mu = np.sin(2 * np.pi * x)
    sigma = 0.1 + (x - 0.6) ** 2
    z = norm.rvs(loc=mu, scale=sigma, random_state=123)

    z_blocks = z.reshape(n_blocks, -1)
    x_blocks = x.reshape(n_blocks, -1)
    max_indices = z_blocks.argmax(axis=-1, keepdims=True)

    x_at_max = np.take_along_axis(x_blocks, max_indices, axis=-1).ravel()
    z_at_max = np.take_along_axis(z_blocks, max_indices, axis=-1).ravel()

    all_z = pd.DataFrame({'x': x, 'Target': z}, index=x)
    all_z.to_csv('all_z.csv', index_label='Index')

    dataset = pd.DataFrame({'x': x_at_max, 'Target': z_at_max}, index=x_at_max)
    dataset.to_csv('max_z.csv', index_label='Index')

    _save_feature(dataset, '../train/feature-synthetic.csv')
    _save_target(dataset, '../train/target-synthetic.csv')
    _save_feature(dataset, '../test/feature-synthetic.csv')
    _save_target(dataset, '../test/target-synthetic.csv')


_compile_dataset()
