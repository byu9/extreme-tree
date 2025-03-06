#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt

from pjm import read_generation
from pjm import read_prediction_extended
from pjm import read_testing


def calc_residual_mae(residuals):
    mae = np.nanmean(abs(residuals))
    print(f'{mae=:}')


def calc_residual_umae(residuals):
    umae = np.nanmean(residuals[residuals < 0])
    print(f'{umae=:}')


def plot_prediction(prediction, target):
    plt.figure()

    plt.step(target.index, target, label='target')
    plt.step(prediction.index, prediction['mean'], label='prediction')

    upper = prediction['hi']
    lower = prediction['lo']
    plt.fill_between(prediction.index, lower, upper, label='interval', color='black', alpha=0.2)

    plt.legend()


def plot_value_at_risk(prediction, generation, target):
    generation = generation.reindex(target.index)

    plt.figure()
    plt.step(generation.index, generation, label='generation')
    plt.step(target.index, target, label='target')
    plt.step(prediction.index, prediction['VaR'], label='VaR')
    plt.legend()


def main():
    _, target = read_testing()
    prediction = read_prediction_extended()
    generation = read_generation()

    residuals = prediction['mean'] - target

    calc_residual_mae(residuals)
    calc_residual_umae(residuals)

    plot_prediction(prediction, target)
    plot_value_at_risk(prediction, generation, target)

    plt.show()


if __name__ == '__main__':
    main()
