class WeightedUnionFind(object):

    """
      Weighted union find algorithm,
    put the smaller tree into the larger tree, lower the tree size.
    >>> wuf = WeightedUnionFind(10)
    >>> connections = [(4, 3), (3, 8), (6, 5), (9, 4),
    ... (2, 1), (8, 9), (5, 0), (7, 2), (6, 1), (1, 0), (6, 7)]
    >>> for i, j in connections:
    ...     wuf.union(i, j)
    ...
    >>> wuf.connected(1, 4)
    False
    >>> wuf.connected(8, 4)
    True
    >>> wuf.connected(1, 5)
    True
    >>> wuf.connected(1, 7)
    True
    """
    def __init__(self, size: int) -> None:
        self._id = [i for i in range(size)];
        self._weight = [1] * size
        self._count = size

    def count(self):
        return self._count

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def find(self, node):
        root = node
        while root != self._id[root]:
            root = self._id[root]

        ## 1.5.12 practice
        ## path compression take the ccurret node and all the nodes above to root

        while root != node:
            immediate_parent = self._id[node]
            self._id[node] = root
            node = immediate_parent

        return root

    def union(self, p, q):
        p_root = self.find(p)
        q_root = self.find(q)
        if p_root == q_root:
            return
        
        if self._weight[p_root] < self._weight[q_root]:
            self._id[p_root] = self._id[q_root]
            self._weight[q_root] += self._weight[p_root]
        else:
            self._id[q_root] = self._id[p_root]
            self._weight[p_root] += self._weight[q_root]
        self._count -= 1
   