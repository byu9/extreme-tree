import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

n_lags = 200
alpha = 0.05


def load_target():
    dataset = pd.read_csv('datasets/pjm/forecasting.csv', index_col=0, parse_dates=True)
    dataset = dataset[dataset.index < '2024']
    target = dataset.resample('1h')['MW'].mean().ffill().to_numpy()
    return target


def make_plots(target):
    plot_acf(target, lags=n_lags, alpha=alpha)
    plt.grid()
    plot_pacf(target, lags=n_lags, alpha=alpha)
    plt.grid()


def save_plot_data(target):
    acf_vals, acf_ci = sm.tsa.acf(target, nlags=n_lags, alpha=alpha)
    pacf_vals, pacf_ci = sm.tsa.pacf(target, nlags=n_lags, alpha=alpha)

    plot_data = pd.DataFrame({
        'acf': acf_vals,
        'pacf': pacf_vals,
        'ci_acf_upper': acf_ci[:, 0],
        'ci_acf_lower': acf_ci[:, 1],
        'ci_pacf_upper': pacf_ci[:, 0],
        'ci_pacf_lower': pacf_ci[:, 1],
    })
    plot_data.to_csv('pjm_acf_pacf.csv', index_label='Index')


def main():
    target = load_target()
    save_plot_data(target)
    make_plots(target)
    plt.show()


main()
