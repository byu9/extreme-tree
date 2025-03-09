from operator import attrgetter

import numpy as np
from tqdm.auto import tqdm

from .binary_trees import BinaryTree
from .genextreme import GenExtreme
from .validation import validate_feature
from .validation import validate_feature_target


class _TreeNode:
    __slots__ = (
        'feature',
        'target',

        'feature_id',
        'threshold',
        'score',

        'prediction',
        'pi',
    )

    def __init__(self, feature, target):
        self.feature = feature
        self.target = target

        self.pi = None

        self.score = None
        self.feature_id = None
        self.threshold = None
        self.prediction = None

    def _splitting_rules(self, min_partition_size, score_func):
        n_features, n_samples = self.feature.shape

        for feature_id in tqdm(range(n_features), desc='Feature', leave=False):
            sort_indices = self.feature[feature_id].argsort()
            feature_vals = self.feature[feature_id, sort_indices]
            target_vals = self.target[sort_indices]

            unique_values = np.unique(feature_vals)
            midpoints = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in tqdm(midpoints, desc='Threshold', leave=False):
                split_index = np.searchsorted(feature_vals, threshold, side='right')

                if min_partition_size <= split_index <= n_samples - min_partition_size:
                    left_target_vals = target_vals[:split_index]
                    right_target_vals = target_vals[split_index:]
                    score = score_func(target_vals, left_target_vals, right_target_vals)
                    yield score, feature_id, threshold

    def find_optimal_rule(self, min_score, **kwargs):
        rules = list(self._splitting_rules(**kwargs))

        if rules:
            scores, feature_ids, thresholds = zip(*rules)
            best_index = max(range(len(scores)), key=scores.__getitem__)

            if scores[best_index] > min_score:
                self.score = scores[best_index]
                self.feature_id = feature_ids[best_index]
                self.threshold = thresholds[best_index]

    def split(self):
        split_mask = self.feature[self.feature_id] <= self.threshold
        left = _TreeNode(self.feature[:, split_mask], self.target[split_mask])
        right = _TreeNode(self.feature[:, ~split_mask], self.target[~split_mask])
        return left, right


class ExtremeTree:
    __slots__ = (
        '_max_n_splits',
        '_distribution',
        '_find_rule_params',
        '_tree',
    )

    def __init__(self, distribution=GenExtreme(), max_n_splits=40, min_partition_size=5,
                 min_score=-np.inf):
        self._max_n_splits = max_n_splits
        self._distribution = distribution
        self._tree = None
        self._find_rule_params = {
            'min_partition_size': min_partition_size,
            'score_func': distribution.score_func,
            'min_score': min_score
        }

    def _ensure_fitted(self):
        if self._tree is None:
            raise RuntimeError('Model is not fitted.')

    def _build_tree(self, feature, target):
        root_node = _TreeNode(feature, target)
        root_node.find_optimal_rule(**self._find_rule_params)
        self._tree = BinaryTree()
        self._tree.add_node(root_node)

        for _ in tqdm(range(self._max_n_splits), desc='Split', leave=False):

            candidate_leaves = [leaf for leaf in self._tree.leaves if leaf.score is not None]
            if not candidate_leaves:
                break

            leaf = max(candidate_leaves, key=attrgetter('score'))
            left, right = leaf.split()
            left.find_optimal_rule(**self._find_rule_params)
            right.find_optimal_rule(**self._find_rule_params)

            self._tree.add_node(left, parent=leaf, is_left=True)
            self._tree.add_node(right, parent=leaf, is_left=False)

        for leaf in self._tree.leaves:
            leaf.prediction = self._distribution.estimate(leaf.target)

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
