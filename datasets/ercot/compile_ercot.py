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

    generation_daily = generation.resample('1d').mean()
    temperature_daily = temperature.resample('1d').mean()
    load_daily = load.resample('1d').mean()

    load_daily_max = load.resample('1d').max().add_suffix('_max')
    load_daily_min = load.resample('1d').min().add_suffix('_min')

    load_shifted = load_daily.shift([1, 2, 7])

    forecasting = load_daily_max.join([load_shifted, temperature_daily], sort=True)
    forecasting.dropna(axis='index', inplace=True)

    forecasting['Hour'] = forecasting.index.hour
    forecasting['Day'] = forecasting.index.day
    forecasting['DoW'] = forecasting.index.dayofweek
    forecasting['Month'] = forecasting.index.month

    forecasting.to_csv('forecasting.csv')
    generation_daily.to_csv('generation.csv')


compile_datasets()
