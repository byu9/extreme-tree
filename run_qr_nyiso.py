#!/usr/bin/env python3
import pandas as pd
from sklearn.linear_model import QuantileRegressor

zones = [
    'NYC',
    'CAPITL',
    'CENTRL',
    'DUNWOD',
    'GENESE',
    'HUD_VL',
    'LONGIL',
    'MHK_VL',
    'NORTH',
    'WEST'
]

quantiles = {
    'q=0.1': 0.1,
    'q=0.2': 0.2,
    'q=0.3': 0.3,
    'q=0.4': 0.4,
    'q=0.5': 0.5,
    'q=0.6': 0.6,
    'q=0.7': 0.7,
    'q=0.8': 0.8,
    'q=0.9': 0.9,
}

for zone in zones:
    train = pd.read_csv(f'nyiso/compiled_datasets/train-{zone}.csv', index_col=0)
    test = pd.read_csv(f'nyiso/compiled_datasets/test-{zone}.csv', index_col=0)

    train_feature = train.drop(columns='Load')
    test_feature = test.drop(columns='Load')

    train_target = train['Load']
    test_target = test['Load']

    predictions = pd.DataFrame(index=test.index)

    for label, quantile in quantiles.items():
        print(f'Fitting {zone}: {label}')
        model = QuantileRegressor(quantile=quantile)
        model.fit(train_feature, train_target)
        predict = model.predict(test_feature)
        predictions[label] = predict

    predictions.to_csv(f'qr_nyiso/{zone}.csv')
