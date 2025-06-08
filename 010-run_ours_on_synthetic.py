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
    pred_dist = genextreme(loc=mu_hat, scale=sigma_hat, c=-xi_hat)

    quantiles = [0.1, 0.5, 0.9, 0.99, 0.999, 0.999999]
    prediction = pd.DataFrame(pred_dist.ppf(quantiles), columns=quantiles)

    return prediction


def main():
    feature, target = read_dataset()
    prediction = fit_predict_model(feature, target)
    prediction.to_csv('020-run_ours_on_synthetic.csv', index_label='index')


if __name__ == '__main__':
    main()
