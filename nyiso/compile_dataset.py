import pandas as pd
import matplotlib.pyplot as plt
from load_zones import zone_filenames

plt.figure()

for zone in zone_filenames.values():
    load_data = pd.read_csv(f'combined_load/{zone}.csv', index_col=0)
    load_data.index = pd.to_datetime(load_data.index, utc=True).tz_convert('America/New_York')

    weather_data = pd.read_csv(f'combined_weather/{zone}.csv', index_col=0)
    weather_data.index = pd.to_datetime(weather_data.index, utc=True).tz_convert('America/New_York')
    weather_data = weather_data.reindex(load_data.index, method='ffill')

    compiled_data = pd.merge(load_data, weather_data, how='left', left_index=True, right_index=True)
    compiled_data = compiled_data.resample('5min').nearest()
    n_samples_per_hour = 12

    compiled_data['Load24'] = load_data['Load'].shift(24 * n_samples_per_hour)
    compiled_data['Load25'] = load_data['Load'].shift(25 * n_samples_per_hour)
    compiled_data['Load168'] = load_data['Load'].shift(168 * n_samples_per_hour)

    compiled_data['mo'] = compiled_data.index.month
    compiled_data['hr'] = compiled_data.index.hour
    compiled_data['dofw'] = compiled_data.index.dayofweek

    max_indices = compiled_data.resample('1h')['Load'].idxmax()
    compiled_data = compiled_data.loc[max_indices]

    compiled_data.dropna(inplace=True, axis='rows')

    plt.clf()
    compiled_data['Load'].plot(title=f'{zone}')
    plt.savefig(f'compiled_datasets/{zone}.png')

    train = compiled_data[compiled_data.index < '2023']
    test = compiled_data[compiled_data.index >= '2023']

    train.to_csv(f'compiled_datasets/train-{zone}.csv')
    test.to_csv(f'compiled_datasets/test-{zone}.csv')
