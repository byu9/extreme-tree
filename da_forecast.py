#!/usr/bin/env python3
import pickle
from abc import ABCMeta
from abc import abstractmethod
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from extreme_tree import ExtremeTree


class DAForecastModel(metaclass=ABCMeta):
    _QUANTILES = {
        f'Target{index}': quantile
        for index, quantile in enumerate(np.linspace(0.1, 0.9, 9))
    }

    @abstractmethod
    def fit(self, feature, target):
        raise NotImplementedError

    @abstractmethod
    def predict(self, feature):
        raise NotImplementedError

    def save_model(self, filename):
        with open(filename, 'wb') as model_file:
            pickle.dump(self, file=model_file)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as model_file:
            model = pickle.load(model_file)

        if not isinstance(model, DAForecastModel):
            raise RuntimeError(f'"{model}" is not a valid forecast model.')

        return model

    @staticmethod
    def save_prediction(predictions, index, filename):
        prediction = pd.DataFrame(index=index, data=predictions)
        prediction.to_csv(filename, index_label='Index')


class DAExtremeTree(DAForecastModel):
    def __init__(self):
        super().__init__()
        self._model = None

    def _create_prediction_dict(self, predict_dist):
        predict_dist = {
            target: predict_dist.ppf(quantile)
            for target, quantile in self._QUANTILES.items()
        }
        return predict_dist

    def fit(self, feature, target):
        self._model = ExtremeTree(max_split=20, min_samples=5)
        self._model.fit(feature.to_numpy(), target.to_numpy().ravel())
        predict_dist = self._model.predict(feature.to_numpy())
        prediction = self._create_prediction_dict(predict_dist)
        return prediction

    def predict(self, feature):
        predict_dist = self._model.predict(feature.to_numpy())
        prediction = self._create_prediction_dict(predict_dist)
        return prediction


class DAQuantileRegressor(DAForecastModel):

    def __init__(self):
        super().__init__()
        self._models = {
            target: QuantileRegressor(quantile=quantile)
            for target, quantile in self._QUANTILES.items()
        }

    def predict(self, feature):
        predict_dist = {
            target: model.predict(feature.to_numpy())
            for target, model in self._models.items()
        }
        return predict_dist

    def fit(self, feature, target):
        for model in self._models.values():
            model.fit(feature.to_numpy(), target.to_numpy().ravel())

        return self.predict(feature)


_supported_models = {
    'et': DAExtremeTree,
    'qr': DAQuantileRegressor,
}


def _parse_args():
    parser = ArgumentParser(
        allow_abbrev=False,
        description='Learns a day-ahead forecast model or uses a learned model to make a forecast.'
    )

    parser.add_argument(
        '--learn', type=str, default=None,
        choices=_supported_models.keys(),
        help='put the program in learn mode and learn with specified model'
    )

    parser.add_argument(
        '--model', type=Path, required=True,
        help='in learn mode, the file path to write the model to; '
             'in forecast mode, the file path to load the model from'
    )

    parser.add_argument(
        '--feature', type=Path, required=True,
        help='the file path to load historical covariates from'
    )

    parser.add_argument(
        '--target', type=Path,
        help='the file path to load historical observations from'
    )

    parser.add_argument(
        '--prediction', type=Path, default=None,
        help='the file path to write predictions to'
    )

    args = parser.parse_args()

    if args.learn is not None:
        if args.learn not in _supported_models:
            raise ValueError(f'Model "{args.learn}" is not supported.')

        if args.prediction is None:
            raise ValueError(f'Prediction file must be specified in forecast mode.')

    return args


def _load_feature(filename):
    feature = pd.read_csv(filename, index_col='Index').sort_index()
    return feature


def _load_target(filename):
    target = pd.read_csv(filename, index_col='Index').sort_index()
    return target


def _run_learn_task(args):
    feature = _load_feature(args.feature)
    target = _load_target(args.target)

    if not feature.index.equals(target.index):
        raise ValueError(f'Feature and target contains different indices.')

    model = _supported_models[args.learn]()
    predictions = model.fit(feature, target)
    model.save_model(args.model)

    if args.prediction is not None:
        model.save_prediction(predictions, index=feature.index, filename=args.prediction)


def _run_predict_task(args):
    feature = _load_feature(args.feature)
    model = DAForecastModel.load_model(args.model)
    predictions = model.predict(feature)
    model.save_prediction(predictions, index=feature.index, filename=args.prediction)


def _run_main():
    args = _parse_args()

    if args.learn:
        _run_learn_task(args)

    else:
        _run_predict_task(args)


if __name__ == '__main__':
    _run_main()
