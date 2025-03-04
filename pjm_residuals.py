import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import genextreme
from scipy.stats import norm

from extreme_tree.equal_distributions import empirical_cdf


def fit_line(x, y):
    slope, intercept = np.polyfit(x, y, deg=1)
    return slope, intercept


def read_test_target():
    dataset = pd.read_csv('datasets/pjm/forecasting.csv', index_col=0, parse_dates=True)
    test_mask = (dataset.index >= '2024')
    target = dataset[test_mask]['MW']
    return target


def read_prediction():
    prediction = pd.read_csv('pjm_prediction.csv', index_col=0)
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


def main():
    prediction = read_prediction()
    target = read_test_target()
    residuals = prediction.mean() - target

    mae = abs(residuals).mean()
    print(f'{mae=:}')

    plt.figure()
    plt.plot(target.index, prediction.mean(), label='predict')
    plt.plot(target.index, target, label='target')
    plt.legend()

    eval_residual_qq(residuals)
    eval_residual_hist(residuals)
    plt.show()


main()
