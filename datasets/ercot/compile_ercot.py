import pandas as pd


def read_generation(filename):
    dataset = pd.read_csv(
        filename,
        header=3,
        usecols=[1, 5],
        names=['Central Time', 'MW'],
        index_col=0
    )
    dataset.index = pd.to_datetime(dataset.index)
    return dataset


def read_load(filename):
    dataset = pd.read_csv(
        filename,
        header=3,
        usecols=[1, 5],
        names=['Central Time', 'MW'],
        index_col=0
    )
    dataset.index = pd.to_datetime(dataset.index)
    return dataset


def read_temperature(filename):
    dataset = pd.read_csv(
        filename,
        header=3,
        usecols=[1, 14],
        names=['Central Time', 'Degrees F'],
        index_col=0
    )
    dataset.index = pd.to_datetime(dataset.index)
    return dataset


def compile_fragments(filenames, read_func):
    fragments = [
        read_func(filename)
        for filename in filenames
    ]
    dataset = pd.concat(fragments, axis='index')
    return dataset


def compile_datasets():
    generation_files = [
        f'../raw/ercot_gen_all_15min_{year}Q{quarter}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
        for quarter in [1, 2, 3, 4]
    ]

    load_files = [
        f'../raw/ercot_load-temp_hr_{year}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
    ]

    generation = compile_fragments(generation_files, read_func=read_generation)
    load = compile_fragments(load_files, read_func=read_load)
    temperature = compile_fragments(load_files, read_func=read_temperature)
    interval = '1d'

    generation_interval = generation.resample(interval).mean().ffill()
    temperature_interval = temperature.resample(interval).mean().ffill()
    load_interval = load.resample(interval).mean().ffill()

    load_interval_max = load.resample(interval).max().ffill().add_suffix('_max')
    load_interval_min = load.resample(interval).min().ffill().add_suffix('_min')

    load_shifted = load_interval.shift([1, 2, 7])

    forecasting_interval = load_interval_max.join([load_shifted, temperature_interval], sort=True)
    forecasting_interval.dropna(axis='index', inplace=True)

    forecasting_interval['Hour'] = forecasting_interval.index.hour
    forecasting_interval['Day'] = forecasting_interval.index.day
    forecasting_interval['DoW'] = forecasting_interval.index.dayofweek
    forecasting_interval['Month'] = forecasting_interval.index.month

    forecasting_interval.to_csv('forecasting.csv')
    generation_interval.to_csv('generation.csv')


compile_datasets()
#
# from statsmodels.graphics.tsaplots import plot_pacf
#
# plot_pacf(load, lags=200)
# plt.show()
