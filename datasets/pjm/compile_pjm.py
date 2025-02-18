import pandas as pd


def to_datetime(arr):
    localized = pd.to_datetime(arr, format='%m/%d/%Y %I:%M:%S %p').tz_localize('UTC')
    converted = localized.tz_convert('US/Eastern')
    return converted


def parse_temperature(value_str):
    value, kind = value_str.split('-')
    value = float(value)

    return value, kind


def read_generation(filename):
    dataset = pd.read_csv(
        filename,
        header=1,
        usecols=[0, 3],
        names=['Time', 'MW'],
        index_col=0
    )
    dataset.index = to_datetime(dataset.index)
    return dataset


def read_load(filename):
    dataset = pd.read_csv(
        filename,
        header=1,
        usecols=[0, 4, 5],
        names=['Time', 'Zone', 'MW'],
        index_col=0
    )
    dataset.index = to_datetime(dataset.index)
    return dataset


def read_temperature(filename):
    dataset = pd.read_csv(
        filename,
        header=1,
        usecols=[0, 4, 5],
        names=['Time', 'Zone', 'Value Kind'],
        index_col=0
    )
    dataset.index = to_datetime(dataset.index)
    dataset['Degrees F'], dataset['Kind'] = zip(*dataset.pop('Value Kind').apply(parse_temperature))
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
        f'raw/day_gen_capacity-{year}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
    ]

    load_files = [
        f'raw/hrl_load_prelim-{year}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
    ]

    temperature_files = [
        f'raw/da_tempset-{year}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
    ]

    load = compile_fragments(load_files, read_func=read_load)
    load = load.drop(columns='Zone').groupby('Time').sum()

    temperature = compile_fragments(temperature_files, read_func=read_temperature)
    temperature = temperature.drop(columns=['Zone', 'Kind']).groupby(['Time']).mean()

    generation = compile_fragments(generation_files, read_func=read_generation)
    generation = generation.resample('1d').mean()
    generation.index = generation.index.tz_localize(None)
    generation.to_csv('generation.csv')

    daily_max = load.resample('1d').max()
    shifted_max = daily_max.shift([1, 2, 7])

    forecasting = daily_max.join([shifted_max, temperature], sort=True)
    forecasting.dropna(axis='index', inplace=True)
    forecasting['Day'] = forecasting.index.day
    forecasting['DoW'] = forecasting.index.dayofweek
    forecasting['Month'] = forecasting.index.month
    forecasting.index = forecasting.index.tz_localize(None)
    forecasting.to_csv('forecasting.csv')


compile_datasets()
