import numpy as np
from tqdm.auto import tqdm

from .binary_trees import BinaryTree
from .equal_distributions import anderson_darling
from .equal_distributions import kolmogorov_smirnov
from .genextreme import GenExtreme
from .partition import Partition
from .validation import validate_feature
from .validation import validate_feature_target

_supported_distributions = {
    'gev': GenExtreme,
}

_supported_statistic = {
    'ks': kolmogorov_smirnov,
    'anderson': anderson_darling,
}


class ExtremeTree:
    __slots__ = (
        '_min_partition_size',
        '_alpha',
        '_max_n_splits',
        '_distribution',
        '_statistic',
        '_tree',
    )

    def __init__(
            self,
            distribution='gev',
            max_n_splits=10,
            min_partition_size=10,
            alpha=0.05,
            statistic='anderson',
    ):

        self._min_partition_size = min_partition_size
        self._alpha = alpha
        self._max_n_splits = max_n_splits
        self._distribution = _supported_distributions[distribution]()
        self._statistic = _supported_statistic[statistic]
        self._tree = None

    def _ensure_fitted(self):
        if self._tree is None:
            raise RuntimeError('Model is not fitted.')

    def _build_tree(self, feature, target):
        root_node = Partition(feature, target)
        root_node.evolve(
            min_partition_size=self._min_partition_size,
            statistic_func=self._statistic
        )

        self._tree = BinaryTree()
        self._tree.add_node(root_node)

        for _ in tqdm(range(self._max_n_splits), desc='Round', leave=False):
            candidate_leaves = [
                leaf for leaf in self._tree.leaves
                if leaf.statistic is not None
            ]

            if not candidate_leaves:
                break

            statistics = np.array([leaf.statistic for leaf in candidate_leaves])
            best_leaf = candidate_leaves[statistics.argmax()]

            left_child, right_child = best_leaf.split()

            left_child.evolve(
                min_partition_size=self._min_partition_size,
                statistic_func=self._statistic
            )

            right_child.evolve(
                min_partition_size=self._min_partition_size,
                statistic_func=self._statistic
            )

            self._tree.add_node(left_child, parent=best_leaf, is_left=True)
            self._tree.add_node(right_child, parent=best_leaf, is_left=False)

        for leaf in self._tree.leaves:
            leaf.params = self._distribution.estimate(leaf.target)

    def _forward_prop(self, feature):
        n_features, n_samples = feature.shape
        self._tree.root.pi = np.ones(shape=n_samples)

        for node in self._tree.topological_ordering():
            if node not in self._tree.leaves:
                feature_val = feature[node.feature_id]
                left_child = self._tree.left_child_of(node)
                right_child = self._tree.right_child_of(node)

                left_child.pi = node.pi * (feature_val <= node.threshold)
                right_child.pi = node.pi * (feature_val > node.threshold)

        pi = np.stack([leaf.pi for leaf in self._tree.leaves], axis=-1)
        params = np.stack([leaf.params for leaf in self._tree.leaves], axis=-1)
        prediction = np.sum(pi * params, axis=-1)

        return prediction

    def fit(self, feature, target):
        feature, target = validate_feature_target(feature, target)
        feature = feature.transpose()
        target = target.transpose()
        self._build_tree(feature, target)

    def predict(self, feature, convert_to_scipy: bool = False):
        self._ensure_fitted()
        feature = validate_feature(feature)
        feature = feature.transpose()
        predict = self._forward_prop(feature)
        predict = np.moveaxis(predict, [1, 2], [-1, -2])

        if convert_to_scipy:
            predict = self._distribution.convert_to_scipy(predict)

        return predict
