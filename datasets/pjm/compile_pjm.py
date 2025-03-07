import pandas as pd


def to_datetime(arr):
    localized = pd.to_datetime(arr, format='%m/%d/%Y %I:%M:%S %p').dt.tz_localize('UTC')
    converted = localized.dt.tz_convert('US/Eastern')
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
        names=['Time', 'MW']
    )
    dataset['Time'] = to_datetime(dataset['Time'])
    dataset.set_index('Time', inplace=True, drop=True)
    return dataset


def read_load(filename):
    dataset = pd.read_csv(
        filename,
        header=1,
        usecols=[0, 4, 5],
        names=['Time', 'Zone', 'MW']
    )
    dataset['Time'] = to_datetime(dataset['Time'])
    dataset.set_index('Time', inplace=True, drop=True)
    return dataset


def read_temperature(filename):
    dataset = pd.read_csv(
        filename,
        header=1,
        usecols=[0, 4, 5],
        names=['Time', 'Zone', 'Value Kind']
    )
    dataset['Degrees F'], dataset['Kind'] = zip(*dataset.pop('Value Kind').apply(parse_temperature))
    dataset['Time'] = to_datetime(dataset['Time'])
    dataset.set_index('Time', inplace=True, drop=True)
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
    return generation


def compile_temperature():
    temperature_files = [
        f'raw/da_tempset-{year}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
    ]
    temperature = compile_fragments(temperature_files, read_func=read_temperature)
    temperature = temperature.reset_index().set_index(['Time', 'Zone'])
    temperature = temperature.pivot(columns='Kind', values='Degrees F')
    temperature = temperature.mean(axis='columns').to_frame(name='Degrees F')
    temperature = temperature.reset_index().set_index('Time')
    temperature = temperature.pivot(columns='Zone', values='Degrees F')

    return temperature


def compile_load():
    load_files = [
        f'raw/hrl_load_estimated-{year}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
    ]

    load = compile_fragments(load_files, read_func=read_load)
    load = pd.pivot(load, columns='Zone', values='MW')
    return load


def read_load_forecast(filename):
    dataset = pd.read_csv(
        filename,
        header=1,
        usecols=[0, 2, 4, 5],
        names=['Forecast Time', 'Time', 'Zone', 'MW']
    )
    dataset['Forecast Time'] = to_datetime(dataset['Forecast Time'])
    dataset['Time'] = to_datetime(dataset['Time'])
    return dataset


def compile_load_forecast():
    files = [
        f'raw/load_frcstd_hist-{year}.csv'
        for year in [2020, 2021, 2022, 2023, 2024]
    ]

    forecast = compile_fragments(files, read_load_forecast)
    forecast = pd.pivot(forecast, index=['Time', 'Forecast Time'], columns='Zone', values='MW')

    # Keep only the first forecast (more than 36 hours ahead)
    forecast = forecast.reset_index().sort_values(['Time', 'Forecast Time']).groupby('Time').first()
    forecast.drop(columns='Forecast Time', inplace=True)

    return forecast


def compile_datasets():
    float_format = '%.0f'

    load = compile_load().sum(axis='columns', skipna=False).rename('Load MW')
    forecast = compile_load_forecast()['RTO'].rename('Forecast MW')
    forecast_error = (load - forecast).to_frame(name='Error MW')

    temperature = compile_temperature().ffill().mean(axis='columns').to_frame('Degrees F')
    temperature = temperature.reindex(forecast_error.index).ffill()

    dataset = pd.concat([forecast_error, temperature], axis='columns')
    dataset['Day'] = dataset.index.day
    dataset['DoW'] = dataset.index.dayofweek
    dataset['Month'] = dataset.index.month
    dataset['Hour'] = dataset.index.hour
    dataset = dataset.resample('1h').ffill().dropna(axis='index')

    peak_time = pd.Index(dataset.resample('1d')['Error MW'].idxmax(), name='Peak Time')
    peak_dataset = dataset.loc[peak_time]

    testing = dataset[dataset.index >= '2024']
    peak_training = peak_dataset[peak_dataset.index < '2024']
    peak_testing = peak_dataset[peak_dataset.index >= '2024']

    testing.to_csv('testing.csv', float_format=float_format)
    peak_training.to_csv('training.csv', float_format=float_format)
    peak_testing.to_csv('validation.csv', float_format=float_format)

    generation = compile_generation()
    generation.to_csv('generation.csv', float_format=float_format)


compile_datasets()
