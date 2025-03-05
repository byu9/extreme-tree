#!/usr/bin/env python3
import pandas as pd

from extreme_tree import ExtremeForest


def load_dataset():
    dataset = pd.read_csv('datasets/pjm/forecasting.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True)
    dataset = dataset.resample('1h').ffill()

    feature = dataset.drop(columns='MW')
    target = dataset[['MW']]

    train_mask = dataset.index < '2024'
    test_mask = (dataset.index >= '2024')

    train = (feature[train_mask], target[train_mask])
    test = (feature[test_mask], target[test_mask])

    return train, test


def main():
    (train_feature, train_target), (test_feature, test_target) = load_dataset()

    model = ExtremeForest(ensemble_size=10, resample_ratio=0.9, min_partition_size=10)
    model.fit(train_feature, train_target)
    mu, sigma, xi = model.predict(test_feature)

    prediction = pd.DataFrame({
        'mu': mu.ravel(),
        'sigma': sigma.ravel(),
        'xi': xi.ravel()
    }, index=test_feature.index)

    prediction.to_csv('pjm_prediction.csv')


main()
