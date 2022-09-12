from collections import defaultdict
import copy
import doctest

class Digraph(object):

    """
      Directed graph implementation. Every edges is directed, so if v is
    reachable from w, w might not be reachable from v.There would ba an
    assist data structure to mark all available vertices, because
    self._adj.keys() is only for the vertices which outdegree is not 0.
    Directed graph is almost the same with Undirected graph,many codes
    from Gragh can be reusable.
    >>> # 4.2.6 practice
    >>> graph = Digraph()
    >>> test_data = [(4, 2), (2, 3), (3, 2), (6, 0), (0, 1), (2, 0),
    ...              (11, 12), (12, 9), (9, 10), (9, 11), (8, 9), (10, 12),
    ...              (11, 4), (4, 3), (3, 5), (7, 8), (8, 7), (5, 4), (0, 5),
    ...              (6, 4), (6, 9), (7, 6)]
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    ...
    >>> graph.vertices_size()
    13
    >>> graph.edges_size()
    22
    >>> [i for i in graph.get_adjacent_vertices(2)]
    [3, 0]
    >>> [j for j in graph.get_adjacent_vertices(6)]
    [0, 4, 9]
    >>> [v for v in graph.vertices()]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    >>> graph
    13 vertices, 22 edges
    4: 2 3
    2: 3 0
    3: 2 5
    6: 0 4 9
    0: 1 5
    11: 12 4
    12: 9
    9: 10 11
    8: 9 7
    10: 12
    7: 8 6
    5: 4
    <BLANKLINE>
    >>>
    """

    def __init__(self, graph=None):
        self._edges_size = 0
        self._adj = defaultdict(list)
        self._vertices = set()

        # 4.2.3 practice, generate graph from another graph.
        if graph:
            self._adj = copy.deepcopy(graph._adj)
            self._edges_size = graph.edges_size()
            self._vertices = copy.copy(graph.vertices())

    def vertices_size(self):
        return len(self._vertices)

    def edges_size(self):
        return self._edges_size

     # 4.2.4 practice, add has_edge method for Digraph
    def has_edge(self, start, end):
        edge = next((i for i in self._adj[start] if i == end), None)
        return edge is not None

    def add_edge(self, start, end):
        # 4.2.5 practice, parallel edge and self cycle are not allowed
        if self.has_edge(start, end) or start == end:
            return

        self._adj[start].append(end)

        self._vertices.add(start)
        self._vertices.add(end)

        self._edges_size += 1

    def get_adjacent_vertices(self, vertex):
        return self._adj[vertex]

    def vertices(self):
        return self._vertices

    def reverse_graph(self):
        reverse_graph = Digraph()

        for vertex in self.vertices():
            for neighbour in self.get_adjacent_vertices(vertex):
                reverse_graph.add_edge(neighbour, vertex)

        return reverse_graph

    def __repr__(self):
        s = str(len(self._vertices)) + ' vertices, ' + \
            str(self._edges_size) + ' edges\n'
        for k in self._adj:
            try:
                lst = ' '.join([vertex for vertex in self._adj[k]])
            except TypeError:
                lst = ' '.join([str(vertex) for vertex in self._adj[k]])
            s += '{}: {}\n'.format(k, lst)
        return s

if __name__ == '__main__':
    doctest.testmod()