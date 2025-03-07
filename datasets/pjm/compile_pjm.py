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
    generation.dropna(inplace=True, axis='index')
    return generation


def compile_temperature():
    temperature_files = [
        f'raw/da_tempset-{year}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
    ]
    temperature = compile_fragments(temperature_files, read_func=read_temperature)
    temperature = temperature.drop(columns=['Zone', 'Kind']).groupby(['Time']).mean()
    temperature = temperature.resample('1d').ffill()
    return temperature


def compile_load():
    load_files = [
        f'raw/hrl_load_estimated-{year}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
    ]

    load = compile_fragments(load_files, read_func=read_load)
    load = pd.pivot(load, columns='Zone', values='MW')
    load = load.resample('1h').ffill()
    load = load.sum(axis='columns', skipna=False).to_frame(name='MW')
    load.dropna(inplace=True, axis='index')
    return load


def build_dataset(load, lagged_load, temperature):
    temperature = temperature.reindex(load.index)
    lagged_load = lagged_load.reindex(load.index)
    dataset = pd.concat([load, lagged_load, temperature], axis='columns')
    dataset['Day'] = dataset.index.day
    dataset['DoW'] = dataset.index.dayofweek
    dataset['Month'] = dataset.index.month
    dataset['Hour'] = dataset.index.hour
    dataset.dropna(inplace=True, axis='index')
    return dataset


def compile_datasets():
    float_format = '%.0f'
    generation = compile_generation()

    load = compile_load()
    lagged_load = load.shift([1, 2, 3], freq='h')

    peak_times = pd.Index(load.resample('1d')['MW'].idxmax(), name='Peak Time')
    peak_load = load.loc[peak_times]
    lagged_peak = lagged_load.reindex(peak_times)

    temperature = compile_temperature()
    temperature = temperature.resample('1h').ffill()

    peak_dataset = build_dataset(load=peak_load, lagged_load=lagged_peak, temperature=temperature)
    testing = build_dataset(load=load, lagged_load=lagged_load, temperature=temperature)

    training = peak_dataset[peak_dataset.index < '2024']
    validating = peak_dataset[peak_dataset.index >= '2024']
    testing = testing[testing.index >= '2024']
    correlation_study = load[load.index < '2024']

    training.to_csv('training.csv', float_format=float_format)
    validating.to_csv('validating.csv', float_format=float_format)
    testing.to_csv('testing.csv', float_format=float_format)

    generation.to_csv('generation.csv', float_format=float_format)
    correlation_study.to_csv('correlation_study.csv', float_format=float_format)


compile_datasets()
