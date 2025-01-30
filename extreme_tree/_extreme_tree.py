from dataclasses import dataclass
from operator import attrgetter

import numpy as np
from tqdm.auto import tqdm

from ._binary_trees import BinaryTree
from ._binary_trees import BinaryTreeNode
from ._genextreme import GenExtreme
from ._misc import validate_feature
from ._misc import validate_feature_target

_supported_dists = {
    'GenExtreme': GenExtreme,
}


@dataclass(init=False, eq=False, slots=True)
class _TreeNode(BinaryTreeNode):
    # Assigned training partition
    feature: ...
    target: ...

    # Distribution parameters
    params: ...

    # This is used to determine from which node to grow the tree. The node with the
    # maximum score drop will be selected.
    score_drop: float

    # Child partitions, if the node can split
    left_feature: ...
    left_target: ...
    right_feature: ...
    right_target: ...

    # Splitting criteria
    feature_id: int
    threshold: ...

    # Forward prop related
    # Firing strength - pi
    pi: ...


class ExtremeTree:
    __slots__ = (
        '_min_part_size',
        '_min_score_drop',
        '_max_n_splits',
        '_dist',
        '_tree',
    )

    def __init__(self,
                 dist='GenExtreme',
                 max_n_splits: int = 10,
                 min_part_size: int = 10,
                 min_score_drop: float = 0.0):

        self._min_part_size = min_part_size
        self._min_score_drop = min_score_drop
        self._max_n_splits = max_n_splits
        self._dist = _supported_dists[dist]()
        self._tree = None

    def _ensure_fitted(self):
        if self._tree is None:
            raise RuntimeError('Model is not fitted.')

    def _evolve_node(self, node: _TreeNode):
        node.params = self._dist.compute_estimate(node.target)
        best_score_drop = self._min_score_drop

        parent_score = self._dist.compute_score(node.params, node.target)

        n_features, n_samples = node.feature.shape
        feature_ids = tqdm(range(n_features), leave=False)
        feature_ids.set_description('Feature')

        for feature_id in feature_ids:
            sort_indices = node.feature[feature_id].argsort()
            feature = node.feature[:, sort_indices]
            target = node.target[:, sort_indices]

            unique_values = np.unique(feature[feature_id])
            midpoints = (unique_values[:-1] + unique_values[1:]) / 2
            thresholds = tqdm(midpoints, leave=False)
            thresholds.set_description('Threshold')

            for threshold in thresholds:
                split_index = np.searchsorted(feature[feature_id], threshold, side='right')

                if self._min_part_size <= split_index <= n_samples - self._min_part_size:
                    left_feature = feature[:, :split_index]
                    right_feature = feature[:, split_index:]

                    left_target = target[:, :split_index]
                    right_target = target[:, split_index:]

                    left_params = self._dist.compute_estimate(left_target)
                    right_params = self._dist.compute_estimate(right_target)

                    left_score = self._dist.compute_score(left_params, left_target)
                    right_score = self._dist.compute_score(right_params, right_target)

                    score_drop = parent_score - left_score - right_score

                    if score_drop > best_score_drop:
                        best_score_drop = score_drop

                        node.score_drop = score_drop
                        node.feature_id = feature_id
                        node.threshold = threshold

                        node.left_feature = left_feature
                        node.right_feature = right_feature

                        node.left_target = left_target
                        node.right_target = right_target

    def _build_tree(self, feature, target):
        self._tree = BinaryTree()

        root_node = _TreeNode()
        root_node.feature = feature
        root_node.target = target
        self._evolve_node(root_node)
        self._tree.add_node(root_node)

        splits = tqdm(range(self._max_n_splits), leave=False)
        splits.set_description('Split')

        for _ in splits:
            candidate_leaves = [leaf for leaf in self._tree.leaves if hasattr(leaf, 'score_drop')]
            if not candidate_leaves:
                break

            best_leaf = max(candidate_leaves, key=attrgetter('score_drop'))

            left_child = _TreeNode()
            left_child.feature = best_leaf.left_feature
            left_child.target = best_leaf.left_target

            right_child = _TreeNode()
            right_child.feature = best_leaf.right_feature
            right_child.target = best_leaf.right_target

            self._evolve_node(left_child)
            self._evolve_node(right_child)

            self._tree.add_node(left_child, parent=best_leaf, is_left=True)
            self._tree.add_node(right_child, parent=best_leaf, is_left=False)

    def _forward_prop(self, feature):
        self._tree.root.pi = 1.0

        for node in self._tree.topological_ordering():
            if node not in self._tree.leaves:
                feature_val = feature[node.feature_id]
                left_child = self._tree.left_child_of(node)
                right_child = self._tree.right_child_of(node)

                left_child.pi = node.pi * (feature_val <= node.threshold)
                right_child.pi = node.pi * (feature_val > node.threshold)

        prediction = self._dist.forward_prop(self._tree.leaves)
        return prediction

    def fit(self, feature, target):
        feature, target = validate_feature_target(feature, target)
        feature = feature.transpose()
        target = target.transpose()
        self._build_tree(feature, target)

    def predict(self, feature):
        self._ensure_fitted()
        feature = validate_feature(feature)
        feature = feature.transpose()
        predict = self._forward_prop(feature)
        return predict
