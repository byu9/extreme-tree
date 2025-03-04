import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import genextreme


def fit_line(x, y):
    slope, intercept = np.polyfit(x, y, deg=1)
    return slope, intercept


def read_test_target():
    dataset = pd.read_csv('datasets/pjm/forecasting.csv', index_col=0, parse_dates=True)
    test_mask = (dataset.index >= '2024')
    target = dataset[test_mask]['MW']
    return target


def read_prediction():
    prediction = pd.read_csv('pjm_prediction.csv', index_col=0)
    predict_dist = genextreme(loc=prediction['mu'], scale=prediction['sigma'], c=-prediction['xi'])
    return predict_dist


def read_generation():
    dataset = pd.read_csv('datasets/pjm/generation.csv', index_col=0, parse_dates=True)
    test_mask = (dataset.index >= '2024')
    dataset = dataset[test_mask]['MW']
    return dataset


def compute_value_at_risk(predict_dist, risk=11.415e-6):
    value_at_risk = predict_dist.ppf(1 - risk)
    return value_at_risk


def main():
    predict_dist = read_prediction()
    value_at_risk = compute_value_at_risk(predict_dist)
    target = read_test_target()
    generation = read_generation()

    plt.figure()
    plt.plot(target, label='Peak Target')
    plt.plot(generation, label='PJM Generation')
    plt.plot(target.index, value_at_risk, label='VaR')
    plt.legend()
    plt.show()


main()
