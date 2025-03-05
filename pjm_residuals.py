#!/usr/bin/env python3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import genextreme
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from extreme_tree.equal_distributions import empirical_cdf

n_lags = 30
alpha = 0.05


def fit_line(x, y):
    slope, intercept = np.polyfit(x, y, deg=1)
    return slope, intercept


def read_test_target():
    dataset = pd.read_csv('datasets/pjm/forecasting.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True)
    dataset = dataset.resample('1h').ffill()
    test_mask = (dataset.index >= '2024')
    target = dataset[test_mask]['MW']
    return target


def read_prediction():
    prediction = pd.read_csv('pjm_prediction.csv', index_col=0)
    prediction.index = pd.to_datetime(prediction.index, utc=True)
    predict_dist = genextreme(loc=prediction['mu'], scale=prediction['sigma'], c=-prediction['xi'])
    return predict_dist


def eval_residual_qq(residuals):
    dist = norm(loc=residuals.mean(), scale=residuals.std())
    cdf_v, cdf_p = empirical_cdf(residuals)

    qq = pd.DataFrame({'theoretical': dist.cdf(cdf_v), 'actual': cdf_p})
    qq.to_csv('pjm-qq.csv', index_label='Index')
    qq_slope, qq_intercept = fit_line(qq['theoretical'], qq['actual'])
    print(f'{qq_slope=:}, {qq_intercept=:}')

    plt.figure()
    plt.scatter(qq['theoretical'], qq['actual'])
    plt.title('Residual QQ')


def eval_residual_hist(residuals):
    hist_vals, hist_edges = np.histogram(residuals, bins=30)
    hist = pd.DataFrame({
        'height': hist_vals,
        'bin_center': (hist_edges[1:] + hist_edges[:-1]) / 2
    })
    hist.to_csv('pjm-hist.csv', index_label='Index')

    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title('Residual Histogram')


def make_acf_pacf_plots(residual):
    plot_acf(residual, lags=n_lags, alpha=alpha)
    plt.grid()
    plot_pacf(residual, lags=n_lags, alpha=alpha)
    plt.grid()


def save_acf_pacf_plot(residual):
    acf_vals, acf_ci = sm.tsa.acf(residual, nlags=n_lags, alpha=alpha)
    pacf_vals, pacf_ci = sm.tsa.pacf(residual, nlags=n_lags, alpha=alpha)

    plot_data = pd.DataFrame({
        'acf': acf_vals,
        'pacf': pacf_vals,
        'ci_acf_upper': acf_ci[:, 0],
        'ci_acf_lower': acf_ci[:, 1],
        'ci_pacf_upper': pacf_ci[:, 0],
        'ci_pacf_lower': pacf_ci[:, 1],
    })
    plot_data.to_csv('pjm_residual_acf_pacf.csv', index_label='Index')


def main():
    prediction = read_prediction()
    target = read_test_target()
    residuals = prediction.mean() - target

    mae = abs(residuals).mean()
    underforecasted = residuals[residuals < 0].mean()

    print(f'{mae=:}')
    print(f'{underforecasted=:}')

    plt.figure()
    plt.step(target.index, prediction.mean(), label='predict')
    plt.step(target.index, target, label='target')
    plt.legend()

    eval_residual_qq(residuals)
    eval_residual_hist(residuals)

    make_acf_pacf_plots(residuals)
    save_acf_pacf_plot(residuals)

    plt.show()


main()
