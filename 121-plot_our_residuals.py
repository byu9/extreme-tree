#!/usr/bin/env python3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

from extreme_tree.equal_distributions import empirical_cdf


def plot_histogram(sample, save_data_as):
    height, edges = np.histogram(sample, bins=30)
    centers = (edges[1:] + edges[:-1]) / 2

    histogram_data = pd.DataFrame({'height': height, 'center': centers})
    histogram_data.set_index('center', inplace=True, drop=True)
    histogram_data.to_csv(save_data_as, index_label='index')

    plt.step(centers, height)
    plt.title('Histogram')


def plot_quantile_quantile(sample, save_data_as):
    sample_dist = norm(loc=sample.mean(), scale=sample.std())
    quantiles, probabilities = empirical_cdf(sample)
    theoretical_quantiles = sample_dist.ppf(probabilities)

    report = pd.DataFrame({'theoretical': theoretical_quantiles, 'actual': quantiles})
    report.to_csv(save_data_as, index_label='index')

    plt.scatter(quantiles, theoretical_quantiles)
    plt.axline((0, 0), (1, 1))
    plt.xlabel('actual')
    plt.ylabel('theoretical')
    plt.title('Quantile-Quantile')


def read_test_target():
    dataset = pd.read_csv('datasets/pjm/peak_testing.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('US/Eastern')
    target = dataset['Load MW']
    return target


def read_our_prediction():
    dataset = pd.read_csv('190-run_ours_on_pjm_testing_peaks.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('US/Eastern')
    prediction = dataset['mean']
    return prediction


def main():
    test_target = read_test_target()
    our_prediction = read_our_prediction()

    residuals = our_prediction - test_target

    plt.figure()
    plot_quantile_quantile(residuals, save_data_as='193-plot_our_residuals_quantile_quantile.csv')

    plt.figure()
    plot_histogram(residuals, save_data_as='193-plot_out_residuals_histogram.csv')

    plt.show()


if __name__ == '__main__':
    main()
