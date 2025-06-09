#!/usr/bin/env python3
import pandas as pd
import pandas as pd
from matplotlib import pyplot as plt


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


def main():
    target = read_testing()
    ours = read_our_prediction()
    competitor1 = read_competitor1_prediction()

    plt.plot(target, label='Demand HIA')
    plt.plot(ours, label='Ours')
    plt.plot(competitor1, label='Competitor1')


    plt.show()

    pass


if __name__ == '__main__':
    main()
