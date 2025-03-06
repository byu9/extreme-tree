import numpy as np
import pandas as pd


def ensure_size_at_least(sample, min_size=2):
    if sample.size < min_size:
        raise ValueError(f'Sample size must be at least {min_size}.')


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
        target = np.ravel(target)

    return target


def validate_feature_target(feature, target):
    feature = validate_feature(feature)
    target = validate_target(target)

    if len(feature) != len(target):
        raise ValueError('Feature and target contains different number of samples.')

    return feature, target
