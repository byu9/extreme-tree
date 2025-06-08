#!/usr/bin/env python3
from quantile_forest import RandomForestQuantileRegressor
import pandas as pd


def read_dataset():
    dataset = pd.read_csv('datasets/synthetic/observations.csv')
    feature = dataset['x'].to_frame()
    target = dataset['y']

    return feature, target


def fit_predict_model(feature, target):
    quantiles = [0.1, 0.5, 0.9, 0.99, 0.999, 0.999999]
    model = RandomForestQuantileRegressor(default_quantiles=quantiles)
    model.fit(feature, target)
    prediction_data = model.predict(feature)

    prediction = pd.DataFrame(prediction_data, columns=quantiles)
    return prediction


def main():
    feature, target = read_dataset()
    prediction = fit_predict_model(feature, target)
    prediction.to_csv('091-run_competitor1_on_synthetic_quantiles.csv', index_label='index')


if __name__ == '__main__':
    main()
