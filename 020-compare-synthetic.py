#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd


def plot_parameters(filename, axes, label):
    dataframe = pd.read_csv(filename, index_col='index')

    axes[0].plot(dataframe['mu'], label=label)
    axes[0].set_title(r'Parameter $\hat{\mu}$')

    axes[1].plot(dataframe['sigma'], label=label)
    axes[1].set_title(r'Parameter $\hat{\sigma}$')

    axes[2].plot(dataframe['xi'], label=label)
    axes[2].set_title(r'Parameter $\hat{\xi}$')


def plot_quantiles(filename, axes, label):
    dataframe = pd.read_csv(filename, index_col='index')

    for index, quantile in enumerate(dataframe.columns):
        axes[index].plot(dataframe[quantile], label=label)
        axes[index].set_title(f'Quantile {quantile}')


def set_axes_style(axes):
    for ax in axes:
        ax.grid()
        ax.legend()


fig1 = plt.figure(figsize=(12, 5))
axes = fig1.subplots(3, 1)
plot_parameters('datasets/synthetic/true_parameters.csv', axes=axes, label='true_parameters')
plot_parameters('090-run_ours_on_synthetic_parameters.csv', axes=axes, label='ours')
plot_parameters('092-run_competitor2_on_synthetic_parameters.csv', axes=axes, label='competitor2')
plt.suptitle('Parameter Estimates')
set_axes_style(axes)

fig2 = plt.figure(figsize=(12, 5))
axes = fig2.subplots(6, 1)
plot_quantiles('datasets/synthetic/true_quantiles.csv', axes=axes, label='true_quantiles')
plot_quantiles('090-run_ours_on_synthetic_quantiles.csv', axes=axes, label='ours')
plot_quantiles('091-run_competitor1_on_synthetic_quantiles.csv', axes=axes, label='competitor1')
plot_quantiles('092-run_competitor2_on_synthetic_quantiles.csv', axes=axes, label='competitor2')
set_axes_style(axes)

plt.show()
