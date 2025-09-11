#!/usr/bin/env python3
import pandas as pd
from matplotlib import pyplot as plt


def write_csv_local_time(dataset, filename):
    dataset = dataset.copy()
    dataset.index = dataset.index.tz_convert('US/Eastern').tz_localize(None)
    dataset.to_csv(filename)


def read_testing():
    dataset = pd.read_csv('datasets/pjm/whole_testing.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('US/Eastern')
    target = dataset['Load MW']
    return target


def read_our_prediction():
    dataset = pd.read_csv('190-run_ours_on_pjm_testing.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('US/Eastern')
    prediction = dataset['VaR']
    return prediction


def read_competitor1_prediction():
    dataset = pd.read_csv('191-run_competitor1_on_pjm_testing.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('US/Eastern')
    prediction = dataset['VaR']
    return prediction


def read_generation():
    dataset = pd.read_csv('datasets/pjm/generation.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True).tz_convert('US/Eastern')
    target = dataset['MW']
    return target


def plot_comparison():
    target = read_testing()
    ours = read_our_prediction()
    competitor1 = read_competitor1_prediction()
    pjm = read_generation()

    comparison = pd.DataFrame({
        'Ours': ours,
        'Competitor1': competitor1,
        'PJM': pjm,
        'Demand': target
    })
    comparison.plot(title='Committed Capacity', grid=True)
    write_csv_local_time(comparison, '194-committed-capacity.csv')


def plot_comparison_cumulative():
    target = read_testing().cumsum()
    ours = read_our_prediction().cumsum()
    competitor1 = read_competitor1_prediction().cumsum()
    pjm = read_generation().cumsum()

    comparison = pd.DataFrame({
        'Ours': ours,
        'Competitor1': competitor1,
        'PJM': pjm,
        'Demand': target
    })

    comparison.plot(title='Cumulative Committed Capacity', grid=True)
    write_csv_local_time(comparison, '195-cumulative-committed.csv')


def plot_under_commitment():
    target = read_testing()
    ours = read_our_prediction() - target
    competitor1 = read_competitor1_prediction() - target
    pjm = read_generation() - target

    ours = ours.clip(upper=0).cumsum()
    competitor1 = competitor1.clip(upper=0).cumsum()
    pjm = pjm.clip(upper=0).cumsum()

    comparison = pd.DataFrame({
        'Ours': ours,
        'Competitor1': competitor1,
        'PJM': pjm,
    })

    comparison.plot(title='Cumulative Under-Committed Capacity', grid=True)
    write_csv_local_time(comparison, '196-cumulative_under_committed.csv')


def main():
    plot_comparison()
    plot_comparison_cumulative()
    plot_under_commitment()
    plt.show()

    pass


if __name__ == '__main__':
    main()
