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


def compile_generation():
    generation_files = [
        f'raw/day_gen_capacity-{year}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
    ]
    generation = compile_fragments(generation_files, read_func=read_generation)
    generation = generation.resample('1h').ffill()
    generation.index = generation.index.tz_localize(None)
    return generation


def compile_temperature():
    temperature_files = [
        f'raw/da_tempset-{year}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
    ]
    temperature = compile_fragments(temperature_files, read_func=read_temperature)
    temperature = temperature.drop(columns=['Zone', 'Kind']).groupby(['Time']).mean()
    temperature = temperature.resample('1h').ffill()
    return temperature


def compile_load():
    load_files = [
        f'raw/hrl_load_estimated-{year}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
    ]

    load = compile_fragments(load_files, read_func=read_load)
    load = load.drop(columns='Zone').groupby('Time').sum()
    load = load.resample('1h').ffill()
    return load


def compile_datasets():
    float_format = '%.1f'

    temperature = compile_temperature()

    generation = compile_generation()
    generation = generation.resample('1d').max()
    generation.to_csv('generation.csv', float_format=float_format)

    load = compile_load()
    load.to_csv('load.csv', float_format=float_format)

    load_daily_max = load.resample('1d').max()
    shifted_max = load_daily_max.shift([2, 3, 6, 8, 13, 22])

    forecasting = load_daily_max.join([shifted_max, temperature], sort=True).ffill()
    forecasting.dropna(inplace=True, axis='index')
    forecasting['Day'] = forecasting.index.day
    forecasting['DoW'] = forecasting.index.dayofweek
    forecasting['Month'] = forecasting.index.month
    forecasting.index = forecasting.index.tz_localize(None)
    forecasting.to_csv('forecasting.csv', float_format=float_format)


compile_datasets()
