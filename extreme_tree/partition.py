import numpy as np
from tqdm.auto import tqdm

from extreme_tree.equal_distributions import anderson_darling
from extreme_tree.equal_distributions import empirical_p_values


class Partition:
    __slots__ = (
        # Assigned training partition
        'feature',
        'target',

        # The best splitting criteria
        'feature_id',
        'threshold',
        'p_value',

        # Distribution parameters
        'params',
    )

    def __init__(self, feature, target):
        _, n_feature_samples = feature.shape
        _, n_target_samples = target.shape

        if n_feature_samples != n_target_samples:
            raise ValueError(f'Feature and target must have the same number of samples.')

        self.feature = feature
        self.target = target

        self.feature_id = None
        self.threshold = None
        self.p_value = None
        self.params = None

    def split_partition(self, feature_id, threshold):
        split_mask = self.feature[feature_id] <= threshold
        left_part = Partition(self.feature[:, split_mask], self.target[:, split_mask])
        right_part = Partition(self.feature[:, ~split_mask], self.target[:, ~split_mask])
        return left_part, right_part

    def _split_candidates(self, min_partition_size):
        n_features, n_samples = self.feature.shape
        feature_ids = tqdm(range(n_features), desc='Feature', leave=False)
        for feature_id in feature_ids:
            sort_indices = self.feature[feature_id].argsort()
            sort_feature = self.feature[:, sort_indices]
            sort_target = self.target[:, sort_indices]

            unique_values = np.unique(sort_feature[feature_id])
            midpoints = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in midpoints:
                split_index = np.searchsorted(sort_feature[feature_id], threshold, side='right')

                if min_partition_size <= split_index <= n_samples - min_partition_size:
                    left_target = sort_target[:, :split_index]
                    right_target = sort_target[:, split_index:]

                    statistic = anderson_darling(left_target, right_target)
                    yield feature_id, threshold, statistic

    def evolve(self, min_partition_size):
        split_statistics = self._split_candidates(min_partition_size)
        candidates = list(zip(*split_statistics))

        if candidates:
            feature_ids, thresholds, statistics = candidates
            p_values = 1 - empirical_p_values(statistics, population=statistics)
            best_split = p_values.argmin()

            self.feature_id = feature_ids[best_split]
            self.threshold = thresholds[best_split]
            self.p_value = p_values[best_split]

            return self.split_partition(feature_id=self.feature_id, threshold=self.threshold)

        else:
            self.p_value = np.inf
