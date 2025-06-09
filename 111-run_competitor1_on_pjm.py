#!/usr/bin/env python3
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor


def read_dataset(filename):
    dataset = pd.read_csv(filename, index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('US/Eastern')
    feature = dataset.drop(columns='Load MW')
    target = dataset['Load MW']
    return feature, target


def write_prediction(filename, model, feature):
    prediction = model.predict(feature)
    prediction_dict = {
        'VaR': prediction,
    }
    prediction = pd.DataFrame(prediction_dict, index=feature.index)
    prediction.to_csv(filename)


def main():
    train_feature, train_target = read_dataset('datasets/pjm/whole_training.csv')
    test_feature, test_target = read_dataset('datasets/pjm/whole_testing.csv')

    model = RandomForestQuantileRegressor(default_quantiles=1 - 0.1 / 365)
    model.fit(train_feature, train_target)

    write_prediction('191-run_competitor1_on_pjm_training.csv', model, train_feature)
    write_prediction('191-run_competitor1_on_pjm_testing.csv', model, test_feature)


if __name__ == '__main__':
    main()
