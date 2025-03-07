#!/usr/bin/env python3
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt


def read_correlation_study():
    dataset = pd.read_csv('datasets/pjm/correlation_study.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True)
    dataset = dataset['MW']
    return dataset


def plot_acf_pacf(sample, save_as, n_lags=200):
    acf_vals = sm.tsa.acf(sample, nlags=n_lags)
    pacf_vals = sm.tsa.pacf(sample, nlags=n_lags)

    report = pd.DataFrame({'acf': acf_vals, 'pacf': pacf_vals}).reset_index(names='lags')
    report.sort_values(by='pacf', ascending=False, inplace=True)
    report = report[report['lags'] >= 36]
    report.to_csv(save_as, index=False)

    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.stem(acf_vals)
    ax1.set_title('ACF')

    ax2.stem(pacf_vals)
    ax2.set_title('PACF')


def main():
    target = read_correlation_study()
    plot_acf_pacf(target, 'pjm_acf_pacf.csv')
    plt.show()


if __name__ == '__main__':
    main()
