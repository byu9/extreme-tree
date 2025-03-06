#!/usr/bin/env python3
import pandas as pd
from scipy.stats import genextreme

from extreme_tree import ExtremeForest


def read_training():
    dataset = pd.read_csv('datasets/pjm/training.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True)
    feature = dataset.drop(columns='MW')
    target = dataset['MW']
    return feature, target


def read_testing():
    dataset = pd.read_csv('datasets/pjm/testing.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True)
    feature = dataset.drop(columns='MW')
    target = dataset['MW']
    return feature, target


def read_generation():
    dataset = pd.read_csv('datasets/pjm/generation.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True)
    dataset = dataset['MW']
    return dataset


def read_prediction():
    prediction = pd.read_csv('pjm_prediction.csv', index_col=0)
    prediction.index = pd.to_datetime(prediction.index, utc=True)
    return prediction


def read_prediction_extended():
    prediction = read_prediction()
    predict_dist = genextreme(loc=prediction['mu'], scale=prediction['sigma'], c=-prediction['xi'])

    extended_data = {
        'mean': predict_dist.mean().ravel(),
        'lo': predict_dist.ppf(0.05).ravel(),
        'hi': predict_dist.ppf(0.95).ravel(),
        'VaR': predict_dist.isf(11.415e-6).ravel(),
    }
    extended = pd.DataFrame(extended_data, index=prediction.index)
    return extended


def write_prediction(index, mu, sigma, xi):
    prediction_dict = {'mu': mu.ravel(), 'sigma': sigma.ravel(), 'xi': xi.ravel()}
    prediction = pd.DataFrame(prediction_dict, index=index)
    prediction.to_csv('pjm_prediction.csv')


def main():
    train_feature, train_target = read_training()
    test_feature, test_target = read_testing()

    model = ExtremeForest(ensemble_size=30, resample_ratio=0.9, min_impurity_drop_ratio=0.4)
    model.fit(train_feature, train_target)
    mu, sigma, xi = model.predict(test_feature)
    write_prediction(test_feature.index, mu, sigma, xi)


if __name__ == '__main__':
    main()
