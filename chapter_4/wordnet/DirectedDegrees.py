from collections import defaultdict
import doctest
from Digraph import Digraph


class DirectedDegrees(object):

    """
    >>> test_data = ((4, 2), (2, 3), (3, 2), (6, 0), (0, 1), (2, 0),
    ...              (11, 12), (12, 9), (9, 10), (9, 11), (7, 9), (10, 12),
    ...              (11, 4), (4, 3), (3, 5), (6, 8), (8, 6), (5, 4), (0, 5),
    ...              (6, 4), (6, 9), (7, 6))
    >>> graph = Digraph()
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    ...
    >>> degree = DirectedDegrees(graph)
    >>> degree.indegree(0)
    2
    >>> degree.outdegree(0)
    2
    >>> degree.indegree(1)
    1
    >>> degree.outdegree(1)
    0
    >>> degree.indegree(9)
    3
    >>> degree.outdegree(9)
    2
    >>> degree.is_map()
    False
    >>> [i for i in degree.sources()]
    []
    """

    def __init__(self, graph: Digraph):
        self._indegree = defaultdict(int)
        self._outdegree = defaultdict(int)
        length = 0
        for vertex in graph.vertices():
            length += 1
            for neighbour in graph.get_adjacent_vertices(vertex):
                self._outdegree[vertex] += 1
                self._indegree[neighbour] += 1
        
        self._sources = (k for k, v in self._indegree.items() if v == 0)
        self._sinks = (k for k, v in self._outdegree.items() if v == 0)
        # A digraph where self-loops are allowed and every vertex has outdegree 1 is called a map
        self._is_map = len([k for k, v in self._outdegree.items() if v == 1]) == length

    def indegree(self, vertex):
        return self._indegree[vertex]

    def outdegree(self, vertex):
        return self._outdegree[vertex]

    def sources(self):
        return self._sources

    def sinks(self):
        return self._sinks

    def is_map(self):
        return self._is_map

if __name__ == '__main__':
    doctest.testmod()