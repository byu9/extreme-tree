import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from load_zones import zone_filenames

plt.figure()

for zone in zone_filenames.values():
    load_data = pd.read_csv(f'combined_load/{zone}.csv', index_col=0)
    load_data.index = pd.to_datetime(load_data.index, utc=True).tz_convert('America/New_York')
    load_data = load_data.resample('1h').mean()

    arma_dataset = pd.DataFrame(index=load_data.index)
    arma_dataset['value'] = load_data['Load']
    arma_dataset['value24'] = load_data['Load'].shift(24)
    arma_dataset['value25'] = load_data['Load'].shift(25)
    arma_dataset['value168'] = load_data['Load'].shift(168)
    arma_dataset.dropna(inplace=True, axis='rows')
    arma_features = arma_dataset.drop(columns='value')
    arma_target = arma_dataset['value']

    arma_model = LinearRegression()
    arma_model.fit(arma_features, arma_dataset['value'])
    arma_predict = arma_model.predict(arma_features)
    arma_residuals = arma_predict - arma_target

    weather_data = pd.read_csv(f'combined_weather/{zone}.csv', index_col=0)
    weather_data.index = pd.to_datetime(weather_data.index, utc=True).tz_convert('America/New_York')
    weather_data = weather_data.resample('1h').ffill()

    residuals_dataset = pd.DataFrame(index=arma_residuals.index)
    residuals_dataset = pd.merge(residuals_dataset, weather_data, how='left', left_index=True, right_index=True)
    residuals_dataset['residual'] = arma_residuals
    residuals_dataset['mo'] = arma_residuals.index.month
    residuals_dataset['hr'] = arma_residuals.index.hour
    residuals_dataset['dofw'] = arma_residuals.index.dayofweek

    max_indices = residuals_dataset.resample('1d')['residual'].idxmax()
    residuals_dataset = residuals_dataset.loc[max_indices]
    residuals_dataset.dropna(inplace=True, axis='rows')
    residuals_dataset.sort_index(inplace=True)

    train = residuals_dataset[residuals_dataset.index < '2023']
    test = residuals_dataset[residuals_dataset.index >= '2023']

    train.to_csv(f'compiled_datasets/train-{zone}.csv')
    test.to_csv(f'compiled_datasets/test-{zone}.csv')

    plt.clf()
    residuals_dataset['residual'].plot(title=f'{zone}')
    plt.savefig(f'compiled_datasets/{zone}.png')
