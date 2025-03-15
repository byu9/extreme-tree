#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.special import expi
from scipy.special import gamma
from scipy.special import gammainc
from scipy.stats import genextreme

from extreme_tree import ExtremeForest


def logarithmic_integral(x):
    return expi(np.log(x))


def lower_incomplete_gamma(s, x):
    return gamma(s) * gammainc(s, x)


def conditional_value_at_risk(mu, sigma, xi, alpha):
    xi_zero = np.isclose(xi, 0)
    one_over_xi = np.divide(1, xi, where=~xi_zero)
    log_alpha = np.log(alpha)
    li_alpha = logarithmic_integral(alpha)

    term = one_over_xi * (lower_incomplete_gamma(1 - xi, -log_alpha) - (1 - alpha))
    term_zero = np.euler_gamma - li_alpha + alpha * np.log(-log_alpha)
    term = np.positive(term_zero, where=xi_zero, out=term)

    return mu + sigma / (1 - alpha) * term


def read_peak_dataset(filename):
    dataset = pd.read_csv(filename, index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('US/Eastern')
    feature = dataset.drop(columns='Load MW')
    target = dataset['Load MW']
    return feature, target


def read_generation():
    dataset = pd.read_csv('datasets/pjm/generation.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('US/Eastern')
    dataset = dataset['MW']
    return dataset


def read_prediction(filename):
    prediction = pd.read_csv(filename, index_col=0)
    prediction.index = pd.to_datetime(prediction.index, utc=True).tz_convert('US/Eastern')
    return prediction


def write_prediction(filename, model, feature):
    mu, sigma, xi = model.predict(feature)
    predict_dist = genextreme(loc=mu.ravel(), scale=sigma.ravel(), c=-xi.ravel())
    eta = 0.1 / 365
    alpha = 1 - eta

    val_at_risk = predict_dist.isf(eta)
    cond_val_at_risk = conditional_value_at_risk(mu.ravel(), sigma.ravel(), xi.ravel(), alpha=alpha)

    prediction_dict = {
        'mu': mu.ravel(),
        'sigma': sigma.ravel(),
        'xi': xi.ravel(),
        'mean': predict_dist.mean(),
        'lo': predict_dist.ppf(0.05),
        'hi': predict_dist.ppf(0.95),
        'VaR': val_at_risk,
        'EUE': (cond_val_at_risk - val_at_risk) * eta
    }
    prediction = pd.DataFrame(prediction_dict, index=feature.index)
    prediction.to_csv(filename)


def main():
    train_feature, train_target = read_peak_dataset('datasets/pjm/training.csv')
    test_feature, test_target = read_peak_dataset('datasets/pjm/testing.csv')
    validation_feature, validation_target = read_peak_dataset('datasets/pjm/validation.csv')

    model = ExtremeForest(ensemble_size=50, min_score=0.05, min_partition_size=30)
    model.fit(train_feature, train_target)

    write_prediction('training_prediction.csv', model, train_feature)
    write_prediction('testing_prediction.csv', model, test_feature)
    write_prediction('validation_prediction.csv', model, validation_feature)


if __name__ == '__main__':
    main()
