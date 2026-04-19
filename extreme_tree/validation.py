import numpy as np


def ensure_size_at_least(sample, min_size=2):
    if sample.size < min_size:
        raise ValueError(f'Sample size must be at least {min_size}.')


def validate_feature(feature):
    # Cast pandas dataframe to numpy without introducing pandas dependency
    func_to_numpy = getattr(feature, 'to_numpy', None)
    if callable(func_to_numpy):
        feature = func_to_numpy()

    feature = np.atleast_2d(feature)
    if not feature.ndim == 2:
        raise ValueError('Feature must be 2D having (n_samples, n_features).')

    return feature


def validate_target(target):
    # Cast pandas dataframe to numpy without introducing pandas dependency
    func_to_numpy = getattr(target, 'to_numpy', None)
    if callable(func_to_numpy):
        target = func_to_numpy()

    target = np.ravel(target)

    return target


def validate_feature_target(feature, target):
    feature = validate_feature(feature)
    target = validate_target(target)

    if len(feature) != len(target):
        raise ValueError('Feature and target contains different number of samples.')

    return feature, target
