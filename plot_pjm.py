#!/usr/bin/env python3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

from extreme_tree.equal_distributions import empirical_cdf
from pjm import read_generation
from pjm import read_peak_dataset
from pjm import read_prediction_extended


def calc_residual_mae(residuals, label):
    mae = np.nanmean(abs(residuals))
    print(f'{label} --- {mae=:}')


def calc_residual_umae(residuals, label):
    umae = np.nanmean(residuals[residuals < 0])
    print(f'{label} --- {umae=:}')


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

    underforecasted = np.count_nonzero(prediction['VaR'] <= target)
    print(f'{underforecasted=:}')


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
    _, test_target = read_peak_dataset('datasets/pjm/testing.csv')
    _, train_target = read_peak_dataset('datasets/pjm/training.csv')
    _, validation_target = read_peak_dataset('datasets/pjm/validating.csv')

    test_prediction = read_prediction_extended('pjm_test_prediction.csv')
    train_prediction = read_prediction_extended('pjm_train_prediction.csv')
    validation_prediction = read_prediction_extended('pjm_validation_prediction.csv')

    generation = read_generation()

    test_residuals = test_prediction['mean'] - test_target
    train_residuals = train_prediction['mean'] - train_target
    validation_residuals = validation_prediction['mean'] - validation_target

    calc_residual_mae(train_residuals, label='train')
    calc_residual_umae(train_residuals, label='train')

    calc_residual_mae(test_residuals, label='test')
    calc_residual_umae(test_residuals, label='test')

    calc_residual_mae(validation_residuals, label='validation')
    calc_residual_umae(validation_residuals, label='validation')

    plot_prediction(test_prediction, test_target)
    plot_value_at_risk(test_prediction, generation, test_target)

    plot_histogram(train_residuals, save_as='pjm_residual_histogram.csv')
    plot_quantile_quantile(train_residuals, save_as='pjm_residual_qq.csv')

    plt.show()


if __name__ == '__main__':
    main()
