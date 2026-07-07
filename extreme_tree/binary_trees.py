from collections import deque


class BinaryTreeNode:
    __slots__ = ()


class BinaryTree:
    __slots__ = (
        "_root",
        "_nodes",
        "_leaves",
        "_non_leaves",
        "_parents",
        "_left_children",
        "_right_children",
    )

    def __init__(self):
        self._root: BinaryTreeNode | None = None
        self._nodes: list[BinaryTreeNode] = list()
        self._leaves: list[BinaryTreeNode] = list()
        self._non_leaves: list[BinaryTreeNode] = list()
        self._parents: dict[BinaryTreeNode, BinaryTreeNode | None] = dict()
        self._left_children: dict[BinaryTreeNode, BinaryTreeNode | None] = dict()
        self._right_children: dict[BinaryTreeNode, BinaryTreeNode | None] = dict()

    def add_node(
        self,
        node: BinaryTreeNode,
        parent: BinaryTreeNode | None = None,
        is_left: bool = True,
    ):

        if node in self._nodes:
            raise ValueError(f"{self} contains {node}.")

        if parent is None:
            if self._root is not None:
                raise ValueError(f"Root exists.")

            self._root = node

        else:
            if parent not in self._nodes:
                raise ValueError(f"Parent {parent} is not in {self}.")

            if parent in self._leaves:
                self._leaves.remove(parent)
                self._non_leaves.append(parent)

            if is_left:
                if self._left_children[parent] is not None:
                    raise ValueError(f"Left child of parent {parent} exists.")

                self._left_children[parent] = node

            else:
                if self._right_children[parent] is not None:
                    raise ValueError(f"Right child of parent {parent} exists.")

                self._right_children[parent] = node

        self._nodes.append(node)
        self._leaves.append(node)
        self._parents[node] = parent
        self._left_children[node] = None
        self._right_children[node] = None

    @property
    def root(self):
        return self._root

    @property
    def nodes(self):
        return list(self._nodes)

    @property
    def leaves(self):
        return list(self._leaves)

    @property
    def non_leaves(self):
        return list(self._non_leaves)

    def __contains__(self, node):
        return node in self._nodes

    def __len__(self):
        return len(self.nodes)

    def _must_contain(self, node: BinaryTreeNode):
        if node not in self:
            raise LookupError(f"{node} is not in {self}.")

    def parent_of(self, node: BinaryTreeNode):
        self._must_contain(node)
        return self._parents[node]

    def left_child_of(self, node: BinaryTreeNode):
        self._must_contain(node)
        return self._left_children[node]

    def right_child_of(self, node: BinaryTreeNode):
        self._must_contain(node)
        return self._right_children[node]

    def ancestors_of(self, node: BinaryTreeNode):
        self._must_contain(node)
        ancestor = self.parent_of(node)

        while ancestor is not None:
            yield ancestor
            ancestor = self.parent_of(ancestor)

    def descendants_of(self, node: BinaryTreeNode):
        """
        Returns descendants in level-order traversal: root, left, right, ...
        """
        self._must_contain(node)

        descendants = deque()
        descendants.append(self.left_child_of(node))
        descendants.append(self.right_child_of(node))

        while descendants:
            descendant = descendants.popleft()

            if descendant is not None:
                yield descendant
                descendants.append(self.left_child_of(descendant))
                descendants.append(self.right_child_of(descendant))

    def topological_ordering(self):
        if self.root is not None:
            yield self.root
            yield from self.descendants_of(self.root)
