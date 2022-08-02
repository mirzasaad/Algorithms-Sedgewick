import doctest

class Node():
    def __init__(self, value, parent, size=1):
        self._value = value
        self._parent = parent
        self._size = size
    
    @property
    def value(self):
        return self._value

    @value.setter
    def parent(self, p):
        self._value = p

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, p):
        self._parent = p

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, s):
        assert s > 0
        self._size = s

    def __repr__(self):
        return '(Value={}, Parent={}, Weight={})'.format(self._value, self._parent, self._size)

class QuickUnionFind(object):

    """
      Union find implementation, the algorithm is a little bit
    like tree algorithm but not the same.
    >>> uf = QuickUnionFind(10)
    >>> connections = [(4, 3), (3, 8), (6, 5), (9, 4), (2, 1),
    ... (8, 9), (5, 0), (7, 2), (6, 1), (1, 0), (6, 7)]
    >>> for i, j in connections:
    ...     uf.union(i, j)
    ...
    >>> uf.connected(1, 4)
    False
    >>> uf.connected(8, 4)
    True
    >>> uf.connected(1, 5)
    True
    >>> uf.connected(1, 7)
    True
    >>> uf.find(4)
    8
    >>> uf.find(8)
    8
    """
    def __init__(self, size) -> None:
        self._id = [i for i in range(size)];
        self._count = size;

    def count(self) -> int:
        return self._count

    
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

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def union(self, p, q):
        p_root = self.find(p)
        q_root = self.find(q)
        if p_root == q_root:
            return
        self._id[p_root] = q_root
        self._count -= 1

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
    
# 1.5.14 practice
class HeightedUnionFind(object):

    """
      Heighted union find algorithm,
    put the shorter tree into taller tree,
    the tree's height won't be taller than log(n).
    >>> huf = HeightedUnionFind(10)
    >>> connections = [(9, 0), (3, 4), (5, 8), (7, 2), (2, 1), (5, 7), (0, 3), (4, 2)]
    >>> for i, j in connections:
    ...     huf.union(i, j)
    ...
    >>> huf.connected(9, 3)
    True
    >>> huf.connected(0, 1)
    True
    >>> huf.connected(9, 8)
    True
    """
    def __init__(self, size) -> None:
        self._id = [i for i in range(size)]
        self._count = size
        self._height = [1] * size

    def count(self):
        return self._count

    def count(self):
        return self._count

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def find(self, node):
        root = node
        while  root != self._id[root]:
            root = self._id[root]
        
        return root
    
    def union(self, p, q):
        p_root = self.find(p)
        q_root = self.find(q)

        if p_root == q_root:
            return

        if self._height[p_root] < self._height[q_root]:
            self._id[p_root] = q_root
        elif self._height[q_root] < self._height[p_root]:
            self._id[q_root] = p_root
        else:
            self._id[q_root] = self._id[p_root]
            self._height[p_root] += 1
        self._count -= 1

if __name__ == '__main__':
    doctest.testmod()