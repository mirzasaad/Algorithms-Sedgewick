
from collections import defaultdict, deque
import doctest

from Digraph import Digraph

class BreadthFirstDirectedPaths():
    """
      Breadth-First-Search algorithm implementation. This algorithm
    uses queue as assist data structure. First enqueue the start_vertex,
    marked it as visited and dequeue the vertex, then marked all the
    adjacent vertices of start_vertex and enqueue them. Continue this process
    util all connected vertices are marked.
      With Breadth-First-Search algorithm, we can find the shortest path from x to y.
    The worst scenario of running time is proportional to O(V + E) (V is the number
    of vertices and E is the number of edges).
    >>> g = Digraph()
    >>> test_data = [(0, 5), (2, 4), (2, 3), (1, 2), (0, 1), (3, 4), (3, 5), (0, 2)]
    >>> for a, b in test_data:
    ...     g.add_edge(a, b)
    ...
    >>> bfp = BreadthFirstDirectedPaths(g, [0])
    >>> [bfp.pathTo(i) for i in range(6)]
    [deque([0]), deque([0, 1]), deque([0, 2]), deque([0, 2, 3]), deque([0, 2, 4]), deque([0, 5])]
    >>> [i for i in bfp.pathTo(4)]
    [0, 2, 4]
    >>> [i for i in bfp.pathTo(5)]
    [0, 5]
    >>> bfp.distanceTo(4)
    2
    >>> bfp.distanceTo(5)
    1
    >>> bfp.distanceTo('not a vertex')
    -1
    """
    def __init__(self, G, source):
        self._graph = G
        self._marked = defaultdict(bool)
        self._disTo = defaultdict(int)
        self._edgeTo = {}
        self._start_node = source

        for v in G.vertices():
            self._disTo[v] = 0

        self.bfs(source)

    def bfs(self, source):
        
        queue = deque([])

        for s in source:
            queue.append(s)
            self._marked[s] = True
            self._disTo[s] = 0

        while queue:
            v = queue.popleft()
            for w in self._graph.get_adjacent_vertices(v):
                if not self._marked[w]:
                    self._marked[w] = True
                    self._edgeTo[w] = v
                    self._disTo[w] = self._disTo[v] + 1

                    queue.append(w)

    def has_path_to(self, v):
        return self._marked[v]

    def pathTo(self, v):
        if not self.has_path_to(v):
            return None

        temp, path = v, deque([])
        while temp is not None:
            path.appendleft(temp)
            temp = self._edgeTo.get(temp, None)

        return path

    def marked(self, vertex):
        return self._marked[vertex]

    def distanceTo(self, v):
        if v not in self._graph.vertices():
            return -1

        return self._disTo[v]

if __name__ == '__main__':
    doctest.testmod()