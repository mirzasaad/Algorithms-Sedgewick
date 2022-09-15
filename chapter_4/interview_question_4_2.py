from collections import defaultdict
import doctest
from operator import ne
import queue
from basic_data_struct import Stack, Queue

from module_4_2 import Digraph, DirectedCycle, BreadthFirstOrder, hamilton_path_exists, Degrees

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
    [2, 3, 2]
    >>> dc.smallest_cycle_length()
    3
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

                    self._cycle.append(vertex)

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

class ShortestDirectedCycleV2(object):

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
    >>> dc = ShortestDirectedCycleV2(graph)
    >>> dc.smallest_cycle()
    [3, 2, 3]
    >>> dc.has_cycle()
    True
    """

    def __init__(self, graph: Digraph):
        self._marked = defaultdict(bool)
        self._on_stack = defaultdict(bool)
        self._edge_to = {}
        self._cycles = []

        for vertex in graph.vertices():
            if not self._marked[vertex]:
                self._dfs(graph, vertex)

    def _dfs(self, graph: Digraph, vertex):
        self._on_stack[vertex] = True
        self._marked[vertex] = True

        for neighbour in graph.get_adjacent_vertices(vertex):
            if not self._marked[neighbour]:
                self._edge_to[neighbour] = vertex
                self._dfs(graph, neighbour)
            elif self._on_stack[neighbour]:
                tmp = vertex
                cycle = []
                while tmp != neighbour:
                    cycle.append(tmp)
                    tmp = self._edge_to[tmp]
                cycle.append(neighbour)
                cycle.append(vertex)

                self._cycles.append(cycle)
        self._on_stack[vertex] = False
    
    def smallest_cycle(self):
        return min(self._cycles, key=lambda cycle: len(cycle))

    def has_cycle(self):
        return self._cycles and len(self._cycles) > 0

class Hamiltonian(object):
    """
    
    """
    def __init__(self, graph: Digraph) -> None:
        if not hamilton_path_exists(graph):
            return 
        
        pass

if __name__ == '__main__':
    doctest.testmod()