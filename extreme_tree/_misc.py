import numpy as np
import pandas as pd


def validate_feature(feature):
    if isinstance(feature, pd.DataFrame):
        feature = feature.to_numpy()

    else:
        feature = np.atleast_2d(feature)

    if not feature.ndim == 2:
        raise ValueError('Feature must be 2D having (n_samples, n_features).')

    return feature


def validate_target(target):
    if isinstance(target, pd.DataFrame):
        target = target.to_numpy()

    else:
        target = np.atleast_2d(target)

    if not target.ndim == 2:
        raise ValueError('Target must be 2D having (n_samples, 1).')

    return target


def validate_feature_target(feature, target):
    feature = validate_feature(feature)
    target = validate_target(target)

    if len(feature) != len(target):
        raise ValueError('Feature and target contains different number of samples.')

    return feature, target
