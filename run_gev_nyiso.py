#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extreme_tree import ExtremeTree

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


# Visualize fitment
def plot_heatmap(dist, vlo, vhi, hlo, hhi, resolution=1000):
    vertical = np.linspace(vlo, vhi, resolution).reshape(-1, 1)
    density = dist.pdf(vertical)
    plt.imshow(density, extent=(hlo, hhi, vlo, vhi), aspect='auto', origin='lower', cmap='hot')
    plt.colorbar(label='Probability Density')


for zone in zones:
    train = pd.read_csv(f'nyiso/compiled_datasets/train-{zone}.csv', index_col=0)
    test = pd.read_csv(f'nyiso/compiled_datasets/test-{zone}.csv', index_col=0)

    train.index = pd.to_datetime(train.index, utc=True)
    test.index = pd.to_datetime(test.index, utc=True)

    train_feature = train.drop(columns='Load')
    test_feature = test.drop(columns='Load')

    train_target = train['Load']
    test_target = test['Load']

    print(f'Fitting {zone}')

    model = ExtremeTree(max_split=20, min_samples=20)
    model.fit(train_feature, train_target, feature_names=train_feature.columns)
    predict = model.predict(test_feature)

    predictions = pd.DataFrame(index=test.index)

    for label, quantile in quantiles.items():
        predictions[label] = predict.ppf(quantile)

    predictions.to_csv(f'gev_nyiso/{zone}.csv')

    plt.clf()
    plot_heatmap(predict, hlo=test_target.index.min(), hhi=test_target.index.max(),
                 vlo=min(test_target.min(), predict.ppf(0.1).min()),
                 vhi=max(test_target.max(), predict.ppf(0.9).max()))
    plt.scatter(test.index, test_target, label='Actual hourly maxima', color='white', s=10)

    plt.grid(color='white', alpha=0.4)
    plt.xlabel(r'UTC Time')
    plt.ylabel(r'Hourly Maximum of Load')
    plt.xticks(rotation=45)
    plt.title('Hourly Maximum of Load')
    plt.legend()
    plt.savefig(f'gev_nyiso/{zone}.png', dpi=600)
