from collections import defaultdict
import doctest
from basic_data_struct import Stack
from module_4_2 import Digraph, DirectedCycle, BreadthFirstOrder

class ShortestDirectedCycle(object):

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
    >>> dc = ShortestDirectedCycle(graph)
    >>> dc.smallest_cycle()
    [3, 2]
    >>> dc.smallest_cycle_length()
    2
    """

    def __init__(self, graph: Digraph):
        self._marked = defaultdict(bool)
        self._length = float('inf')

        dc = DirectedCycle(graph)

        if not dc.has_cycle():
            raise Exception('no cycle found')

        reverse = graph.reverse_graph()

        for vertex in graph.vertices():
            bfs = BreadthFirstOrder(reverse, vertex)

            for neighbour in graph.get_adjacent_vertices(vertex):
                if bfs.has_path_to(neighbour) and bfs.dist_to(neighbour) + 1 < self._length:
                    self._length = bfs.dist_to(neighbour) + 1;
                    self._cycle = []

                    for v in bfs.path_to(neighbour):
                        self._cycle.append(v)

    def smallest_cycle(self):
        return self._cycle[::-1]   

    def smallest_cycle_length(self):
        return len(self._cycle)

class DirectedCycles(object):

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
    >>> dc = DirectedCycles(graph)
    >>> dc.cycles()
    {(0, 2, 4, 5), (9, 10, 12), (2, 3), (3, 4, 5), (9, 11, 12), (7, 8)}
    """

    def __init__(self, graph: Digraph):
        self._cycles = set()

        dc = DirectedCycle(graph)

        if not dc.has_cycle():
            raise Exception('no cycle found')

        reverse = graph.reverse_graph()

        for vertex in graph.vertices():
            bfs = BreadthFirstOrder(reverse, vertex)

            for neighbour in graph.get_adjacent_vertices(vertex):
                if bfs.has_path_to(neighbour) and bfs.dist_to(neighbour):

                    cycle = []

                    for v in bfs.path_to(neighbour):
                        cycle.append(v)

                    self._cycles.add(tuple(sorted(cycle)))

    def cycles(self):
        return self._cycles

    def smallest_cycle_length(self):
        return len(self._cycle)




if __name__ == '__main__':
    doctest.testmod()

test_data = ((4, 2), (2, 3), (3, 2), (6, 0), (0, 1), (2, 0),
             (11, 12), (12, 9), (9, 10), (9, 11), (7, 9), (10, 12),
             (11, 4), (4, 3), (3, 5), (6, 8), (8, 6), (5, 4), (0, 5),
             (6, 4), (6, 9), (7, 6))

graph = Digraph()

for a, b in test_data:
    graph.add_edge(a, b)

