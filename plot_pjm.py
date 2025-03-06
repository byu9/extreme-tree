#!/usr/bin/env python3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

from extreme_tree.equal_distributions import empirical_cdf
from pjm import read_generation
from pjm import read_prediction_extended
from pjm import read_testing


def calc_residual_mae(residuals):
    mae = np.nanmean(abs(residuals))
    print(f'{mae=:}')


def calc_residual_umae(residuals):
    umae = np.nanmean(residuals[residuals < 0])
    print(f'{umae=:}')


def plot_prediction(prediction, target):
    plt.figure()

    plt.step(target.index, target, label='target')
    plt.step(prediction.index, prediction['mean'], label='prediction')

    upper = prediction['hi']
    lower = prediction['lo']
    plt.fill_between(prediction.index, lower, upper, label='interval', color='black', alpha=0.2)

    plt.legend()


def plot_value_at_risk(prediction, generation, target):
    generation = generation.reindex(target.index)

    plt.figure()
    plt.step(generation.index, generation, label='generation')
    plt.step(target.index, target, label='target')
    plt.step(prediction.index, prediction['VaR'], label='VaR')
    plt.legend()


def plot_histogram(sample, save_as):
    height, edges = np.histogram(sample, bins=30)
    centers = (edges[1:] + edges[:-1]) / 2

    histogram_data = pd.DataFrame({'height': height, 'center': centers})
    histogram_data.set_index('center', inplace=True, drop=True)
    histogram_data.to_csv(save_as)

    plt.figure()
    plt.stem(centers, height)
    plt.title('Histogram')


def plot_quantile_quantile(sample, save_as):
    sample_dist = norm(loc=sample.mean(), scale=sample.std())
    quantiles, probabilities = empirical_cdf(sample)
    theoretical_quantiles = sample_dist.ppf(probabilities)

    report = pd.DataFrame({'theoretical': theoretical_quantiles, 'actual': quantiles})
    report.to_csv(save_as, index_label='Index')

    plt.figure()
    plt.scatter(theoretical_quantiles, quantiles)
    plt.axline((0, 0), (1, 1))
    plt.xlabel('theoretical')
    plt.ylabel('actual')
    plt.title('Quantile-Quantile')


def main():
    _, target = read_testing()
    prediction = read_prediction_extended()
    generation = read_generation()

    residuals = prediction['mean'] - target

    calc_residual_mae(residuals)
    calc_residual_umae(residuals)

    plot_prediction(prediction, target)
    plot_value_at_risk(prediction, generation, target)
    plot_histogram(residuals, save_as='pjm_residual_histogram.csv')
    plot_quantile_quantile(residuals, save_as='pjm_residual_qq.csv')

    plt.show()


if __name__ == '__main__':
    main()
