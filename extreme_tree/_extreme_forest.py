import numpy as np
from tqdm.auto import tqdm

from ._extreme_tree import ExtremeTree
from ._misc import validate_feature
from ._misc import validate_feature_target


class ExtremeForest:
    __slots__ = (
        '_ensemble',
        '_resample_ratio',
        '_random_gen'
    )

    def __init__(self, ensemble_size: int, resample_ratio: float | None, **weak_args):
        self._ensemble = tqdm([
            ExtremeTree(**weak_args)
            for _ in range(ensemble_size)
        ])
        self._ensemble.set_description('Ensemble')
        self._resample_ratio = resample_ratio
        self._random_gen = np.random.default_rng()

    def fit(self, feature, target):
        feature, target = validate_feature_target(feature, target)
        sample_indices = np.arange(len(target))
        resample_size = int(self._resample_ratio * len(target))

        for weak_learner in self._ensemble:
            selected_indices = self._random_gen.choice(sample_indices, size=resample_size)
            selected_feature = feature[selected_indices]
            selected_target = target[selected_indices]
            weak_learner.fit(selected_feature, selected_target)

    def predict(self, feature):
        feature = validate_feature(feature)

        weak_predictions = list()
        for weak_learner in self._ensemble:
            weak_predictions.append(weak_learner.predict(feature, return_dist=False))

        return sum(weak_predictions) / len(weak_predictions)
