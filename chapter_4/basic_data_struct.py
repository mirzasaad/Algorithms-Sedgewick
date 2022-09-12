#!/usr/bin/env python
# -*- encoding:UTF-8 -*-
import doctest
from collections import defaultdict

"""
    copy from module_1_3.py, this is for avoiding package import problems.
"""


class Node(object):

    def __init__(self, val):
        self._val = val
        self.next_node = None

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

    @property
    def next_node(self):
        return self._next_node

    @next_node.setter
    def next_node(self, node):
        self._next_node = node


class Stack(object):

    def __init__(self):
        self._first = None
        self._size = 0

    def __iter__(self):
        node = self._first
        while node:
            yield node.val
            node = node.next_node

    def is_empty(self):
        return self._first is None

    def size(self):
        return self._size

    def push(self, val):
        node = Node(val)
        old = self._first
        self._first = node
        self._first.next_node = old
        self._size += 1

    def pop(self):
        if self._first:
            old = self._first
            self._first = self._first.next_node
            self._size -= 1
            return old.val
        return None

    # 1.3.7 practice
    def peek(self):
        if self._first:
            return self._first.val
        return None


class Queue(object):

    def __init__(self, q=None):
        self._first = None
        self._last = None
        self._size = 0
        if q:
            for item in q:
                self.enqueue(item)

    def __iter__(self):
        node = self._first
        while node:
            yield node.val
            node = node.next_node

    def is_empty(self):
        return self._first is None

    def size(self):
        return self._size

    def enqueue(self, val):
        old_last = self._last
        self._last = Node(val)
        self._last.next_node = None
        if self.is_empty():
            self._first = self._last
        else:
            old_last.next_node = self._last
        self._size += 1

    def dequeue(self):
        if not self.is_empty():
            val = self._first.val
            self._first = self._first.next_node
            if self.is_empty():
                self._last = None
            self._size -= 1
            return val
        return None


class Bag(object):

    def __init__(self):
        self._first = None
        self._size = 0

    def __iter__(self):
        node = self._first
        while node is not None:
            yield node.val
            node = node.next_node

    def __contains__(self, item):
        tmp = self._first
        while tmp:
            if tmp == item:
                return True
        return False

    def add(self, val):
        node = Node(val)
        old = self._first
        self._first = node
        self._first.next_node = old
        self._size += 1

    def is_empty(self):
        return self._first is None

    def size(self):
        return self._size


class MinPQ(object):

    def __init__(self, data=None):
        self._pq = []
        if data:
            for item in data:
                self.insert(data)

    def is_empty(self):
        return len(self._pq) == 0

    def size(self):
        return len(self._pq)

    def swim(self, pos):
        while pos > 0 and self._pq[(pos - 1) // 2] > self._pq[pos]:
            self._pq[(pos - 1) //
                     2], self._pq[pos] = self._pq[pos], self._pq[(pos - 1) // 2]
            pos = (pos - 1) // 2

    def sink(self, pos):
        length = len(self._pq) - 1
        while 2 * pos + 1 <= length:
            index = 2 * pos + 1
            if index < length and self._pq[index] > self._pq[index + 1]:
                index += 1
            if self._pq[pos] <= self._pq[index]:
                break
            self._pq[index], self._pq[pos] = self._pq[pos], self._pq[index]
            pos = index

    def insert(self, val):
        self._pq.append(val)
        self.swim(len(self._pq) - 1)

    def del_min(self):
        min_val = self._pq[0]
        last_index = len(self._pq) - 1
        self._pq[0], self._pq[last_index] = self._pq[last_index], self._pq[0]
        self._pq.pop(last_index)
        self.sink(0)
        return min_val

    def min_val(self):
        return self._pq[0]


class DisjointNode(object):

    def __init__(self, parent, size=1):
        self._parent = parent
        self._size = size

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, new_parent):
        self._parent = new_parent

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, val):
        assert val > 0
        self._size = val


class GenericUnionFind(object):

    """
    >>> guf = GenericUnionFind()
    >>> connections = [(4, 3), (3, 8), (6, 5), (9, 4),
    ...                (2, 1), (8, 9), (5, 0), (7, 2), (6, 1), (1, 0), (6, 7)]
    >>> for i, j in connections:
    ...     guf.union(i, j)
    ...
    >>> guf.connected(1, 4)
    False
    >>> guf.connected(8, 4)
    True
    >>> guf.connected(1, 5)
    True
    >>> guf.connected(1, 7)
    True
    """

    def __init__(self, tuple_data=None):
        self._id = {}
        if tuple_data:
            for a, b in tuple_data:
                self.union(a, b)

    def count(self):
        pass

    def connected(self, p, q):
        return self.find(p) and self.find(q) and self.find(p) == self.find(q)

    def find(self, node):
        if node not in self._id:
            return None
        tmp = node
        while self._id[tmp].parent != tmp:
            tmp = self._id[tmp].parent
        return self._id[tmp].parent

    def union(self, p, q):
        p_root = self.find(p)
        q_root = self.find(q)

        if p_root == q_root:
            if p_root is None and q_root is None:
                self._id[p] = DisjointNode(q)
                self._id[q] = DisjointNode(q, 2)
                return
            return

        if p_root is None:
            self._id[p] = DisjointNode(q_root, 1)
            self._id[q_root].size += 1
            return

        if q_root is None:
            self._id[q] = DisjointNode(p_root, 1)
            self._id[p_root].size += 1
            return

        if self._id[p_root].size < self._id[q_root].size:
            self._id[p_root].parent = q_root
            self._id[q_root].size += self._id[p_root].size
        else:
            self._id[q_root].parent = p_root
            self._id[p_root].size += self._id[q_root].size


class MaxPQ(object):

    def __init__(self, data=None):
        self._pq = []
        if data:
            for item in data:
                self.insert(item)

    def is_empty(self):
        return len(self._pq) == 0

    def size(self):
        return len(self._pq)

    def swim(self, pos):
        while pos > 0 and self._pq[(pos - 1) // 2] < self._pq[pos]:
            self._pq[(pos - 1) //
                     2], self._pq[pos] = self._pq[pos], self._pq[(pos - 1) // 2]
            pos = (pos - 1) // 2

    def sink(self, pos):
        length = len(self._pq) - 1
        while 2 * pos + 1 <= length:
            index = 2 * pos + 1
            if index < length and self._pq[index] < self._pq[index + 1]:
                index += 1
            if self._pq[pos] >= self._pq[index]:
                break
            self._pq[index], self._pq[pos] = self._pq[pos], self._pq[index]
            pos = index

    def insert(self, val):
        self._pq.append(val)
        self.swim(len(self._pq) - 1)

    def del_max(self):
        max_val = self._pq[0]
        last_index = len(self._pq) - 1
        self._pq[0], self._pq[last_index] = self._pq[last_index], self._pq[0]
        self._pq.pop(last_index)
        self.sink(0)
        return max_val

    def max_val(self):
        return self._pq[0]


class IndexMinPQ(object):

    def __init__(self, max_size):
        assert max_size > 0
        self._max_size = max_size
        self._index = [-1] * (max_size + 1)
        self._reverse_index = [-1] * (max_size + 1)
        self._keys = [None] * (max_size + 1)
        self._keys_size = 0

    def is_empty(self):
        return self._keys_size == 0

    def size(self):
        return self._keys_size

    def contains(self, index):
        if index >= self._max_size:
            return False
        return self._reverse_index[index] != -1

    def insert(self, index, element):
        if index >= self._max_size or self.contains(index):
            return

        self._keys_size += 1
        self._index[self._keys_size] = index
        self._reverse_index[index] = self._keys_size
        self._keys[index] = element
        self.swim(self._keys_size)

    def min_index(self):
        return None if self._keys_size == 0 else self._index[1]

    def min_key(self):
        return None if self._keys_size == 0 else self._keys[self._index[1]]

    def exchange(self, pos_a, pos_b):
        self._index[pos_a], self._index[pos_b] = self._index[pos_b], self._index[pos_a]
        self._reverse_index[self._index[pos_a]] = pos_a
        self._reverse_index[self._index[pos_b]] = pos_b

    def swim(self, pos):
        while pos > 1 and self._keys[self._index[pos // 2]] > self._keys[self._index[pos]]:
            self.exchange(pos // 2, pos)
            pos //= 2

    def sink(self, pos):
        length = self._keys_size
        while 2 * pos <= length:
            tmp = 2 * pos
            if tmp < length and self._keys[self._index[tmp]] > self._keys[self._index[tmp + 1]]:
                tmp += 1
            if not self._keys[self._index[tmp]] < self._keys[self._index[pos]]:
                break
            self.exchange(tmp, pos)
            pos = tmp

    def change_key(self, i, key):
        if i < 0 or i >= self._max_size or not self.contains(i):
            return
        self._keys[i] = key
        self.swim(self._reverse_index[i])
        self.sink(self._reverse_index[i])

    def delete_min(self):
        if self._keys_size == 0:
            return
        min_index = self._index[1]
        self.exchange(1, self._keys_size)
        self._keys_size -= 1
        self.sink(1)
        self._reverse_index[min_index] = -1
        self._keys[self._index[self._keys_size + 1]] = None
        self._index[self._keys_size + 1] = -1
        return min_index


# data structure for EdgeWeightedDiGraph Topological

class DirectedCycle(object):

    def __init__(self, graph):
        self._marked = defaultdict(bool)
        self._edge_to = {}
        self._on_stack = defaultdict(bool)
        self._cycle = Stack()
        for v in graph.vertices():
            if not self._marked[v]:
                self.dfs(graph, v)

    def dfs(self, graph, vertex):
        self._on_stack[vertex] = True
        self._marked[vertex] = True

        for edge in graph.adjacent_edges(vertex):
            end = edge.end
            if self.has_cycle():
                return
            elif not self._marked[end]:
                self._edge_to[end] = vertex
                self.dfs(graph, end)
            elif self._on_stack[end]:
                tmp = vertex
                while tmp != end:
                    self._cycle.push(tmp)
                    tmp = self._edge_to[tmp]
                self._cycle.push(end)
                self._cycle.push(vertex)
        self._on_stack[vertex] = False

    def has_cycle(self):
        return not self._cycle.is_empty()

    def cycle(self):
        return self._cycle


class DepthFirstOrder(object):

    def __init__(self, graph):
        self._pre = Queue()
        self._post = Queue()
        self._reverse_post = Stack()
        self._marked = defaultdict(bool)

        for v in graph.vertices():
            if not self._marked[v]:
                self.dfs(graph, v)

    def dfs(self, graph, vertex):
        self._pre.enqueue(vertex)
        self._marked[vertex] = True
        for edge in graph.adjacent_edges(vertex):
            if not self._marked[edge.end]:
                self.dfs(graph, edge.end)

        self._post.enqueue(vertex)
        self._reverse_post.push(vertex)

    def prefix(self):
        return self._pre

    def postfix(self):
        return self._post

    def reverse_postfix(self):
        return self._reverse_post


class Topological(object):

    def __init__(self, graph):
        cycle_finder = DirectedCycle(graph)
        self._order = None
        if not cycle_finder.has_cycle():
            df_order = DepthFirstOrder(graph)
            self._order = df_order.reverse_postfix()

    def order(self):
        return self._order

    def is_DAG(self):
        return self._order is not None


if __name__ == '__main__':
    doctest.testmod()
