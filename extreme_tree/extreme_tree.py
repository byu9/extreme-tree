from operator import attrgetter

import numpy as np
from tqdm.auto import tqdm

from .binary_trees import BinaryTree
from .genextreme import GenExtreme
from .validation import validate_feature
from .validation import validate_feature_target


class _Partition:
    __slots__ = (
        'feature',
        'target',
    )

    def __init__(self, feature, target):
        self.feature = feature
        self.target = target

    def split(self, feature_id, threshold):
        split_mask = self.feature[feature_id] <= threshold
        left_part = _Partition(self.feature[:, split_mask], self.target[split_mask])
        right_part = _Partition(self.feature[:, ~split_mask], self.target[~split_mask])
        return left_part, right_part

    def splitting_rules(self, min_partition_size):
        n_features, n_samples = self.feature.shape

        for feature_id in tqdm(range(n_features), desc='Feature', leave=False):
            sort_indices = self.feature[feature_id].argsort()
            sort_feature = self.feature[:, sort_indices]
            sort_target = self.target[sort_indices]

            unique_values = np.unique(sort_feature[feature_id])
            midpoints = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in tqdm(midpoints, desc='Threshold', leave=False):
                split_index = np.searchsorted(sort_feature[feature_id], threshold, side='right')

                if min_partition_size <= split_index <= n_samples - min_partition_size:
                    left_target = sort_target[:split_index]
                    right_target = sort_target[split_index:]

                    yield feature_id, threshold, left_target, right_target


class _TreeNode:
    __slots__ = (
        'partition',

        'feature_id',
        'threshold',
        'impurity_drop',

        'prediction',
        'impurity',
        'pi',
    )

    def __init__(self, partition):
        self.partition = partition
        self.pi = None

        self.impurity_drop = -np.inf
        self.feature_id = None
        self.threshold = None
        self.prediction = None
        self.impurity = None

    def find_optimal_rule(self, min_partition_size, distribution):
        self.prediction = distribution.estimate(self.partition.target)
        self.impurity = distribution.impurity(self.partition.target, params=self.prediction)
        rules = self.partition.splitting_rules(min_partition_size=min_partition_size)

        for feature_id, threshold, left_target, right_target in rules:

            left_params = distribution.estimate(left_target)
            right_params = distribution.estimate(right_target)

            left_impurity = distribution.impurity(left_target, params=left_params)
            right_impurity = distribution.impurity(right_target, params=right_params)
            impurity_drop = self.impurity - left_impurity - right_impurity

            if impurity_drop > self.impurity_drop:
                self.impurity_drop = impurity_drop
                self.feature_id = feature_id
                self.threshold = threshold


class ExtremeTree:
    __slots__ = (
        '_min_partition_size',
        '_min_impurity_drop_ratio',
        '_max_n_splits',
        '_distribution',
        '_tree',
    )

    def __init__(self, distribution=GenExtreme(), max_n_splits=20, min_partition_size=5,
                 min_impurity_drop_ratio=0.0):
        self._min_partition_size = min_partition_size
        self._max_n_splits = max_n_splits
        self._min_impurity_drop_ratio = min_impurity_drop_ratio
        self._distribution = distribution
        self._tree = None

    def _ensure_fitted(self):
        if self._tree is None:
            raise RuntimeError('Model is not fitted.')

    def _build_tree(self, feature, target):
        self._tree = BinaryTree()

        root_node = _TreeNode(_Partition(feature, target))
        self._tree.add_node(root_node)

        root_node.find_optimal_rule(
            min_partition_size=self._min_partition_size,
            distribution=self._distribution
        )

        min_impurity_drop = root_node.impurity_drop * self._min_impurity_drop_ratio

        for _ in tqdm(range(self._max_n_splits), desc='Split', leave=False):
            leaf = max(self._tree.leaves, key=attrgetter('impurity_drop'), default=None)

            if leaf is None:
                break

            if not leaf.impurity_drop > min_impurity_drop:
                break

            left_part, right_part = leaf.partition.split(leaf.feature_id, leaf.threshold)

            left_child = _TreeNode(left_part)
            right_child = _TreeNode(right_part)

            left_child.find_optimal_rule(
                min_partition_size=self._min_partition_size,
                distribution=self._distribution
            )
            right_child.find_optimal_rule(
                min_partition_size=self._min_partition_size,
                distribution=self._distribution
            )

            self._tree.add_node(left_child, parent=leaf, is_left=True)
            self._tree.add_node(right_child, parent=leaf, is_left=False)

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
        predictions = np.stack([leaf.prediction for leaf in self._tree.leaves], axis=-1)
        prediction = np.nansum(pi * predictions, axis=-1, keepdims=True)

        return prediction

    def fit(self, feature, target):
        feature, target = validate_feature_target(feature, target)
        feature = feature.transpose()
        target = target.ravel()
        self._build_tree(feature, target)

    def predict(self, feature):
        self._ensure_fitted()
        feature = validate_feature(feature)
        feature = feature.transpose()
        predict = self._forward_prop(feature)
        return predict
