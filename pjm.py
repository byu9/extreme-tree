#!/usr/bin/env python3
import pandas as pd
from scipy.stats import genextreme

from extreme_tree import ExtremeForest


def read_peak_dataset(filename):
    dataset = pd.read_csv(filename, index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('US/Eastern')
    feature = dataset.drop(columns='Error MW')
    target = dataset['Error MW']
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


def read_prediction_extended(filename):
    prediction = read_prediction(filename)
    predict_dist = genextreme(loc=prediction['mu'], scale=prediction['sigma'], c=-prediction['xi'])

    extended_data = {
        'mean': predict_dist.mean().ravel(),
        'lo': predict_dist.ppf(0.05).ravel(),
        'hi': predict_dist.ppf(0.95).ravel(),
        'VaR': predict_dist.isf(11.415e-6).ravel(),
    }
    extended = pd.DataFrame(extended_data, index=prediction.index)
    return extended


def write_prediction(filename, model, feature):
    mu, sigma, xi = model.predict(feature)
    prediction_dict = {'mu': mu.ravel(), 'sigma': sigma.ravel(), 'xi': xi.ravel()}
    prediction = pd.DataFrame(prediction_dict, index=feature.index)
    prediction.to_csv(filename)


def main():
    train_feature, train_target = read_peak_dataset('datasets/pjm/training.csv')
    test_feature, test_target = read_peak_dataset('datasets/pjm/testing.csv')
    validation_feature, validation_target = read_peak_dataset('datasets/pjm/validation.csv')

    model = ExtremeForest(ensemble_size=100, resample_ratio=0.9, max_n_splits=40, min_impurity_drop_ratio=0.01)
    model.fit(train_feature, train_target)

    write_prediction('training_prediction.csv', model, train_feature)
    write_prediction('testing_prediction.csv', model, test_feature)
    write_prediction('validation_prediction.csv', model, validation_feature)


if __name__ == '__main__':
    main()
