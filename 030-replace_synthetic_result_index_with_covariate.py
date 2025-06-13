#!/usr/bin/env python3
# The script is used to facilitate data preparation for plotting in LaTeX
from glob import glob

import pandas as pd


def read_covariate():
    dataframe = pd.read_csv('datasets/synthetic/true_parameters.csv', index_col='index')
    return dataframe['x']


def read_result(filename):
    dataframe = pd.read_csv(filename, index_col='index')
    return dataframe


def main():
    for filename in glob('09*.csv'):
        covariate = read_covariate()

        result = read_result(filename)
        result['x'] = covariate

        result.reset_index(drop=True).set_index('x').to_csv(filename)


if __name__ == '__main__':
    main()
