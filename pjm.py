#!/usr/bin/env python3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import genextreme
from scipy.stats import norm

from extreme_tree import ExtremeForest
from extreme_tree.equal_distributions import empirical_cdf


def load_dataset():
    dataset = pd.read_csv('datasets/pjm/forecasting.csv', index_col=0, parse_dates=True)
    feature = dataset.drop(columns='MW')
    target = dataset[['MW']]

    train_mask = dataset.index < '2024'
    test_mask = (dataset.index >= '2024')

    train = (feature[train_mask], target[train_mask])
    test = (feature[test_mask], target[test_mask])

    return train, test


def train_model(train):
    train_feature, train_target = train
    model = ExtremeForest(ensemble_size=30, resample_ratio=0.8, max_n_splits=20)
    model.fit(train_feature, train_target)
    return model


def plot_predict_and_target(predict_dist, target):
    plt.figure()
    plt.plot(target.index, predict_dist.mean(), label='predict')
    plt.plot(target.index, target, label='target')
    plt.legend()


def test_model(model, test):
    test_feature, test_target = test
    mu, sigma, xi = model.predict(test_feature)

    prediction = pd.DataFrame({
        'mu': mu.ravel(),
        'sigma': sigma.ravel(),
        'xi': xi.ravel()
    }, index=test_feature.index)
    prediction.to_csv('pjm_prediction.csv')

    predict_dist = genextreme(loc=mu, scale=sigma, c=-xi)
    plot_predict_and_target(predict_dist, test_target)

    residuals = pd.DataFrame({
        'residual': predict_dist.mean().ravel() - test_target.to_numpy().ravel()
    }, index=test_target.index)

    print(f'predict-mae={abs(residuals).mean()}')
    return residuals


def fit_line(x, y):
    slope, intercept = np.polyfit(x, y, deg=1)
    return slope, intercept


def eval_residual_qq(residuals):
    dist = norm(loc=residuals.mean(), scale=residuals.std())
    cdf_v, cdf_p = empirical_cdf(residuals)

    qq = pd.DataFrame({'theoretical': dist.cdf(cdf_v), 'actual': cdf_p})
    qq.plot.scatter(x='theoretical', y='actual', title='Residual QQ')
    qq.to_csv('pjm-qq.csv', index_label='Index')

    qq_slope, qq_intercept = fit_line(qq['theoretical'], qq['actual'])
    print(f'{qq_slope=:}, {qq_intercept=:}')


def eval_residual_hist(residuals):
    hist_vals, hist_edges = np.histogram(residuals, bins=10)
    hist = pd.DataFrame({
        'height': hist_vals,
        'bin_center': (hist_edges[1:] + hist_edges[:-1]) / 2
    })
    hist.to_csv('pjm-hist.csv', index_label='Index')


def main():
    train, test = load_dataset()

    model = train_model(train)
    residuals = test_model(model, test)

    eval_residual_qq(residuals)
    eval_residual_hist(residuals)

    plt.show()


main()
