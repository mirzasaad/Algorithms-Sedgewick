from collections import defaultdict
import doctest
from Digraph import Digraph

class DirectedCycle(object):

    """
      Using Depth-First-Search algorithm to check
    whether a cycle exists in a directed graph.
    There is an assist attribute call _on_stack,
    if an adjacent vertex is in _on_stack(True),
    that means a cycle exists.
    >>> graph = Digraph()
    >>> test_data = [(4, 2), (2, 3), (3, 2), (6, 0), (0, 1), (2, 0),
    ...              (11, 12), (12, 9), (9, 10), (9, 11), (8, 9), (10, 12),
    ...              (11, 4), (4, 3), (3, 5), (7, 8), (8, 7), (5, 4), (0, 5),
    ...              (6, 4), (6, 9), (7, 6)]
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    ...
    >>> dc = DirectedCycle(graph)
    >>> dc.has_cycle()
    True
    """

    def __init__(self, graph: Digraph):
        self._marked = defaultdict(bool)
        self._on_stack = defaultdict(bool)
        self._has_cycle = False

        for vertex in graph.vertices():
            if not self._marked[vertex]:
                if self.dfs(graph, vertex):
                    self._has_cycle = True
                    break

    def dfs(self, graph: Digraph, vertex):
        self._marked[vertex] = True
        self._on_stack[vertex] = True

        for neighbour in graph.get_adjacent_vertices(vertex):
            if not self._marked[neighbour]:
                if self.dfs(graph, neighbour):
                    return True
            elif self._on_stack[neighbour]:
                    return True

        self._on_stack[vertex] = False
        return False

    def has_cycle(self):
        return self._has_cycle

if __name__ == '__main__':
    doctest.testmod()