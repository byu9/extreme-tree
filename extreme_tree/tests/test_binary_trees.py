import unittest

from ..binary_trees import BinaryTree
from ..binary_trees import BinaryTreeNode


class TestBinaryTree(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tree = BinaryTree()

        self.root = BinaryTreeNode()
        self.left = BinaryTreeNode()
        self.right = BinaryTreeNode()
        self.left_left = BinaryTreeNode()

        self.tree.add_node(self.root)
        self.tree.add_node(self.left, parent=self.root, is_left=True)
        self.tree.add_node(self.right, parent=self.root, is_left=False)
        self.tree.add_node(self.left_left, parent=self.left, is_left=True)

    def test_root(self):
        self.assertEqual(self.tree.root, self.root)

    def test_nodes(self):
        expected = [self.root, self.left, self.right, self.left_left]
        self.assertEqual(self.tree.nodes, expected)

    def test_leaves(self):
        expected = [self.right, self.left_left]
        self.assertEqual(self.tree.leaves, expected)

    def test_non_leaves(self):
        expected = [self.root, self.left]
        self.assertEqual(self.tree.non_leaves, expected)

    def test_contains(self):
        self.assertIn(self.root, self.tree)
        self.assertIn(self.left, self.tree)
        self.assertIn(self.right, self.tree)
        self.assertIn(self.left_left, self.tree)

    def test_contains_foreign_node(self):
        node = BinaryTreeNode()
        self.assertNotIn(node, self.tree)

    def test_len(self):
        self.assertEqual(len(self.tree), 4)

    def test_parent_of(self):
        self.assertIs(self.tree.parent_of(self.root), None)
        self.assertIs(self.tree.parent_of(self.left), self.root)
        self.assertIs(self.tree.parent_of(self.right), self.root)
        self.assertIs(self.tree.parent_of(self.left_left), self.left)

    def test_left_child_of(self):
        self.assertIs(self.tree.left_child_of(self.root), self.left)
        self.assertIs(self.tree.left_child_of(self.left), self.left_left)
        self.assertIs(self.tree.left_child_of(self.right), None)
        self.assertIs(self.tree.left_child_of(self.left_left), None)

    def test_right_child_of(self):
        self.assertIs(self.tree.right_child_of(self.root), self.right)
        self.assertIs(self.tree.right_child_of(self.left), None)
        self.assertIs(self.tree.right_child_of(self.right), None)
        self.assertIs(self.tree.right_child_of(self.left_left), None)

    def test_ancestors(self):
        self.assertEqual(set(self.tree.ancestors_of(self.root)), set())
        self.assertEqual(set(self.tree.ancestors_of(self.left)), {self.root})
        self.assertEqual(set(self.tree.ancestors_of(self.right)), {self.root})
        self.assertEqual(set(self.tree.ancestors_of(self.left_left)),
                         {self.left, self.root})

    def test_descendants(self):
        self.assertEqual(set(self.tree.descendants_of(self.root)),
                         {self.left, self.right, self.left_left})

        self.assertEqual(set(self.tree.descendants_of(self.left)),
                         {self.left_left})

        self.assertEqual(set(self.tree.descendants_of(self.right)), set())
        self.assertEqual(set(self.tree.descendants_of(self.left_left)), set())

    def test_topological_ordering(self):
        expected = [self.root, self.left, self.right, self.left_left]
        self.assertEqual(list(self.tree.topological_ordering()), expected)

    def test_add_node_twice(self):
        with self.assertRaises(ValueError):
            self.tree.add_node(self.left, parent=self.root, is_left=True)

    def test_add_root_twice(self):
        with self.assertRaises(ValueError):
            self.tree.add_node(self.root)

    def test_convert_to_boolean(self):
        self.assertTrue(self.tree)

    def test_parent_of_foreign_node(self):
        node = BinaryTreeNode()

        with self.assertRaises(LookupError):
            self.tree.parent_of(node)

    def test_left_child_of_foreign_node(self):
        node = BinaryTreeNode()

        with self.assertRaises(LookupError):
            self.tree.left_child_of(node)

    def test_right_child_of_foreign_node(self):
        node = BinaryTreeNode()

        with self.assertRaises(LookupError):
            self.tree.right_child_of(node)

    def test_ancestors_of_foreign_node(self):
        node = BinaryTreeNode()

        with self.assertRaises(LookupError):
            list(self.tree.ancestors_of(node))

    def test_descendants_of_foreign_node(self):
        node = BinaryTreeNode()

        with self.assertRaises(LookupError):
            list(self.tree.descendants_of(node))

    def test_add_root_exists(self):
        another_root = BinaryTreeNode()

        with self.assertRaises(ValueError):
            self.tree.add_node(another_root, parent=None)

    def test_add_left_child_exists(self):
        another_left_child = BinaryTreeNode()

        with self.assertRaises(ValueError):
            self.tree.add_node(another_left_child, parent=self.root,
                               is_left=True)

    def test_add_right_child_exists(self):
        another_right_child = BinaryTreeNode()

        with self.assertRaises(ValueError):
            self.tree.add_node(another_right_child, parent=self.root,
                               is_left=False)


class TestEmptyBinaryTree(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = BinaryTree()

    def test_convert_to_boolean(self):
        self.assertFalse(self.tree)

    def test_topological_ordering(self):
        """Traversal of empty tree should not raise exceptions."""
        for _ in self.tree.topological_ordering():
            pass

    def test_add_with_foreign_parent(self):
        parent = BinaryTreeNode()
        node = BinaryTreeNode()

        with self.assertRaises(ValueError):
            self.tree.add_node(node, parent=parent, is_left=True)
