#!/usr/bin/env python3
import pandas as pd
from extreme_tree import ExtremeForest
from scipy.stats import genextreme


def read_dataset():
    dataset = pd.read_csv('datasets/synthetic/observations.csv')
    feature = dataset['x'].to_frame()
    target = dataset['y']

    return feature, target


def write_prediction(filename, model, feature):
    prediction = model.predict(feature)
    prediction_dict = {
        'VaR': prediction,
    }
    prediction = pd.DataFrame(prediction_dict, index=feature.index)
    prediction.to_csv(filename)


def fit_predict_model(feature, target):
    model = ExtremeForest(ensemble_size=50, min_score=0.0001, min_partition_size=20)
    model.fit(feature, target)
    mu_hat, sigma_hat, xi_hat = model.predict(feature)
    return mu_hat, sigma_hat, xi_hat


def write_quantiles_to_file(mu, sigma, xi):
    quantiles = [0.1, 0.5, 0.9, 0.99, 0.999, 0.999999]

    pred_dist = genextreme(loc=mu, scale=sigma, c=-xi)
    dataframe = pd.DataFrame(pred_dist.ppf(quantiles), columns=quantiles)
    dataframe.to_csv('090-run_ours_on_synthetic_quantiles.csv', index_label='index')


def write_parameters_to_file(mu, sigma, xi):
    dataframe = pd.DataFrame({'mu': mu.ravel(), 'sigma': sigma.ravel(), 'xi': xi.ravel()})
    dataframe.to_csv('090-run_ours_on_synthetic_parameters.csv', index_label='index')


def main():
    feature, target = read_dataset()
    mu_hat, sigma_hat, xi_hat = fit_predict_model(feature, target)

    write_parameters_to_file(mu_hat, sigma_hat, xi_hat)
    write_quantiles_to_file(mu_hat, sigma_hat, xi_hat)


if __name__ == '__main__':
    main()
