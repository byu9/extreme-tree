#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

quantiles = np.linspace(0.1, 0.9, 9).reshape(1, -1)


def _pinball_loss(prediction, target):
    scores = np.full_like(prediction, fill_value=np.nan)
    np.multiply(target - prediction, quantiles, where=target >= prediction, out=scores)
    np.multiply(prediction - target, 1 - quantiles, where=target < prediction, out=scores)

    scores = scores.mean(axis=0)
    return scores


def _crps_loss(prediction, target):
    scores = np.full_like(prediction, fill_value=np.nan)
    np.square(quantiles, where=target >= prediction, out=scores)
    np.square(1 - quantiles, where=target < prediction, out=scores)

    scores = scores.sum(axis=-1).mean(axis=0)
    return scores


def _parse_args():
    parser = ArgumentParser(
        allow_abbrev=False,
        description='Validates intraday predictions against actual observations.'
    )
    parser.add_argument(
        '--pairs', type=Path, required=True,
        help='CSV file listing prediction/target file pairs'
    )
    parser.add_argument(
        '--scoreboard', type=Path, required=True,
        help='file to write the validation scores to'
    )
    args = parser.parse_args()
    return args


def _load_pairs_table(filename):
    pairs = pd.read_csv(filename, index_col='Label')
    return pairs.to_dict(orient='index')


def _load_target(filename):
    target_df = pd.read_csv(filename, index_col='Index')
    return target_df


def _load_prediction(filename):
    prediction_df = pd.read_csv(filename, index_col='Index')
    return prediction_df


def _run_main():
    args = _parse_args()
    pairs = _load_pairs_table(args.pairs)

    score_dict = dict()
    for label, paths in pairs.items():
        prediction = _load_prediction(paths['Prediction']).to_numpy()
        target = _load_target(paths['Target']).to_numpy()
        score_dict[label] = _crps_loss(prediction, target)

    scoreboard = pd.DataFrame.from_dict(score_dict, orient='index')
    scoreboard.loc['(mean)'] = scoreboard.mean(axis='index')
    scoreboard.to_csv(args.scoreboard, index_label='Label', float_format='%.2f')


if __name__ == '__main__':
    _run_main()
