
from collections import defaultdict, deque
import collections
import doctest
from enum import Enum
from operator import itemgetter, ne
import random
from typing import Deque
from basic_data_struct import Stack, Queue
from module_4_2 import Degrees, Digraph, DirectedCycle, KosarajuSCC
from module_4_1 import Graph, BreadthFirstPaths, ConnectedComponent, DepthFirstPaths, Cycle

class EulerianType(Enum):
    NOT_A_EULER_GRAPH = 0
    EULER_PATH = 1
    EULER_CIRCUIT_OR_CYCLE = 2

# https://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/EulerianCycle.java.html
# NOTE euler cycle and circuit is same thing (dont confuse)

class EulerianFluery(object):
    """
    https://www.geeksforgeeks.org/fleurys-algorithm-for-printing-eulerian-path/
    https://gist.github.com/SuryaPratapK/b750d259655ad0523e281762fa93d2d2#file-euler-graph-L45
    https://jlmartin.ku.edu/courses/math105-F11/Lectures/chapter5-part2.pdf

    Question 3
    Euler cycle. An Euler cycle in a graph is a cycle (not necessarily simple) that uses every edge in the graph exactly one.

    Show that a connected graph has an Euler cycle if and only if every vertex has even degree.
    Design a linear-time algorithm to determine whether a graph has an Euler cycle, and if so, find one.

    Eulerian Path is a path in graph that visits every edge exactly once. 
    Eulerian Circuit is an Eulerian Path which starts and ends on the same vertex. 

    Eulerian Cycle(circuit): An undirected graph has Eulerian cycle if following two conditions are true. 

        All vertices with non-zero degree are connected. We don’t care about vertices with zero degree because they don’t belong to Eulerian Cycle or Path (we only consider all edges). 
        All vertices have even degree.

    Eulerian Path: An undirected graph has Eulerian Path if following two conditions are true. 

        Same as condition (a) for Eulerian Cycle.
        If zero or two vertices have odd degree and all other vertices have even degree. Note that only one vertex with odd degree is not possible in an undirected graph (sum of all degrees is always even in an undirected graph)
        Note that a graph with no edges is considered Eulerian because there are no edges to traverse.

    A Graph must have nodes with even degree and odd degree. All the odd degree nodes are either start or end but all the even degree node will be only intermediate nodes.
    But in case when all the nodes has even degree, then it contains the Eulerian Tour, as we we'll start from some node and will end to this same node, as we need to consume all the edges.
    
    >>> G = Graph()
    >>> test_data = [(1, 0), (0, 2), (2, 1), (0, 3), (3, 4), (4, 0)]
    >>> for a, b in test_data:
    ...     G.add_edge(a, b)
    >>> euler = EulerianFluery(G)
    >>> '[(isEuler ===>, {}), (isConnected ===>, {}), (hasCycle ===>, {})]'.format(euler.isEuler(), euler._cc.count() == 1, euler._cycle.has_cycle())
    '[(isEuler ===>, EulerianType.EULER_CIRCUIT_OR_CYCLE), (isConnected ===>, True), (hasCycle ===>, True)]'
   
   
    >>> G = Graph()    
    >>> test_data = [(1, 0), (0, 2), (2, 1), (0, 3), (3, 4)]
    >>> for a, b in test_data:
    ...     G.add_edge(a, b)
    >>> euler = EulerianFluery(G)
    >>> '[(isEuler ===>, {}), (isConnected ===>, {}), (hasCycle ===>, {})]'.format(euler.isEuler(), euler._cc.count() == 1, euler._cycle.has_cycle())
    '[(isEuler ===>, EulerianType.EULER_PATH), (isConnected ===>, True), (hasCycle ===>, True)]'
   
   
    >>> G = Graph()
    >>> test_data = [(1, 0), (0, 2), (2, 1), (0, 3), (3, 4), (1, 3)]
    >>> for a, b in test_data:
    ...     G.add_edge(a, b)
    >>> euler = EulerianFluery(G)
    >>> '[(isEuler ===>, {}), (isConnected ===>, {}), (hasCycle ===>, {})]'.format(euler.isEuler(), euler._cc.count() == 1, euler._cycle.has_cycle())
    '[(isEuler ===>, EulerianType.NOT_A_EULER_GRAPH), (isConnected ===>, True), (hasCycle ===>, True)]'
    
    
    >>> G = Graph()
    >>> test_data = [(0, 1), (1, 2), (2, 0)]
    >>> for a, b in test_data:
    ...     G.add_edge(a, b)
    >>> euler = EulerianFluery(G)
    >>> '[(isEuler ===>, {}), (isConnected ===>, {}), (hasCycle ===>, {})]'.format(euler.isEuler(), euler._cc.count() == 1, euler._cycle.has_cycle())
    '[(isEuler ===>, EulerianType.EULER_CIRCUIT_OR_CYCLE), (isConnected ===>, True), (hasCycle ===>, True)]'
    """

    def __init__(self, g: Graph) -> None:
        self._cc = ConnectedComponent(g)
        self._cycle = Cycle(g)
        self._graph = g

    def isEuler(self):
        if self._cc.count() > 1 or not self._cycle.has_cycle():
            return EulerianType.NOT_A_EULER_GRAPH

        number_of_odd_vertex_degree = self.odd_vertices()

        if number_of_odd_vertex_degree > 2:  # only start and end node can have odd degree, others  verteces should have even degrees
            return EulerianType.NOT_A_EULER_GRAPH

        return EulerianType.EULER_CIRCUIT_OR_CYCLE if number_of_odd_vertex_degree == 0 else EulerianType.EULER_PATH

    def odd_vertices(self):
        number_of_odd__vertex_degree = 0

        for vertex in self._graph.vertices():
            if self._graph.degree(vertex) & 1:
                number_of_odd__vertex_degree += 1

        return number_of_odd__vertex_degree

    def get_random_vertex_with_odd_degree(self):
        return random.sample([v for v in self._graph.vertices() if self._graph.degree(v) & 1], 1)[-1]

    def get_random_vertex(self):
        return random.sample(self._graph.vertices(), 1)[-1]

    def trail(self):
        isEuler = self.isEuler()

        if isEuler == EulerianType.NOT_A_EULER_GRAPH:
            raise Exception('Not a Euler Graph')

        # if the graph is euler curcuit(all vertex has even degree we can start anywhere), if the graph is euler cycle(only 2 vertex has odd degree)
        # we have to start to from one odd degree vertex

        # Following is Fleury’s Algorithm for printing the Eulerian trail or cycle

        # Make sure the graph has either 0 or 2 odd vertices.
        # If there are 0 odd vertices, start anywhere. If there are 2 odd vertices, start at one of them.
        # Follow edges one at a time. If you have a choice between a bridge and a non-bridge, always choose the non-bridge.
        # Stop when you run out of edges.

        startVertex = self.get_random_vertex(
        ) if isEuler == EulerianType.EULER_CIRCUIT_OR_CYCLE else self.get_random_vertex_with_odd_degree()

        result = []
        self.dfs(startVertex, result)
        return result

    def count_verteces(self, start):
        def count(vertex):
            visited[vertex] = True
            __count = 0

            for neighbour in self._graph.get_adjacent_vertices(vertex):
                if not visited[neighbour]:
                    __count += count(neighbour)

            return __count

        visited = collections.defaultdict(bool)
        return count(start)

    # check is the edge can be removed and the graph is stil connected graph
    def can_next_edge_be_removed(self, start_vertex, neighbour):
        """
        The edge start_vertex-neighbour is valid in one of the following two cases:

        1) If start_vertex is the only adjacent vertex of neighbour
        2) If there are multiple adjacents, then start_vertex-neighbour is not a bridge Do following steps to check if start_vertex-neighbour is a bridge
            2.a) count of vertices reachable from start_vertex
            2.b) Remove edge (start_vertex, neighbour) and after removing the edge, count
                vertices reachable from start_vertex
            2.c) Add the edge back to the graph
            2.d) If vertex count before removing is greater, then edge (start_vertex, neighbour) is a bridge

        """

        if self._graph.degree(start_vertex) == 1:
            return True
        else:
            before = self.count_verteces(start_vertex)
            self._graph.remove_edge(start_vertex, neighbour)

            after = self.count_verteces(start_vertex)
            self._graph.add_edge(start_vertex, neighbour)

            return False if before > after else True

    def dfs(self, vertex, result):
        # Recur for all the vertices adjacent to this vertex
        for neighbour in self._graph.get_adjacent_vertices(vertex):
            # If edge u-v is not removed and it's a a valid next edge
            if self.can_next_edge_be_removed(vertex, neighbour):
                result.append((vertex, neighbour))
                self._graph.remove_edge(vertex, neighbour)
                self.dfs(neighbour, result)


class EulerianHierholzer(object):
    """
    >>> G = Digraph()
    >>> test_data = [(1, 0), (0, 2), (2, 1), (0, 3), (3, 4), (4, 0)]
    >>> for a, b in test_data:
    ...     G.add_edge(a, b)
    >>> euler = EulerianHierholzer(G)
    >>> '[(isEuler ===>, {}), (isConnected ===>, {}), (hasCycle ===>, {})]'.format(euler.isEuler(), euler._cc.count() == 1, euler._cycle.has_cycle())
    '[(isEuler ===>, EulerianType.EULER_CIRCUIT_OR_CYCLE), (isConnected ===>, True), (hasCycle ===>, True)]'
    >>> print([i for i in euler.trail()])
    [3, 4, 0, 2, 1, 0, 3]
    """
    def __init__(self, g: Digraph) -> None:
        self._cc = KosarajuSCC(g)
        self._cycle = DirectedCycle(g)
        self._degrees = Degrees(g)
        self._graph = g

    def has_cycle(self):
        return self._cycle.has_cycle()

    def isEuler(self):
        if self._cc.count() > 1 or not self._cycle.has_cycle():
            return EulerianType.NOT_A_EULER_GRAPH

        if self.isEulerCircuit():
            return EulerianType.EULER_CIRCUIT_OR_CYCLE

        if self.isEulerPath():
            return EulerianType.EULER_PATH

        return EulerianType.NOT_A_EULER_GRAPH

    def isEulerCircuit(self):
        return all([self._degrees.indegree(v) == self._degrees.indegree(v) for v in self._graph.vertices()])

    def isEulerPath(self):

        start, end = None, None
        for vertex in self._graph.vertices():
            if self._degrees.indegree(vertex) == self._degrees.outdegree(vertex):
                continue

            if self._degrees.outdegree(vertex) - self._degrees.indegree(vertex) == 1 and start is None:
                start = vertex
            else:
                return None
            

            if self._degrees.indegree(vertex) - self._degrees.outdegree(vertex) == 1 and end is None:
                end = vertex
            else:
                return None

        if start is None or end is None:
            return None

        return start
    
    def trail(self):
        if not self.isEuler():
            return None

        result = Stack()

        start = random.sample(self._graph.vertices(), 1)[-1] if self.isEulerCircuit() else self.isEulerPath()

        result.push(start)
        self._dfs(start, result)

        return result

    def _dfs(self, vertex, result: Stack):
        while self._degrees.outdegree(vertex) != 0:
            self._degrees._outdegree[vertex] -= 1
            for neighbour in self._graph.get_adjacent_vertices(vertex):
                if self._degrees.outdegree(neighbour) > 0:
                    self._dfs(neighbour, result)

        result.push(vertex)
        
if __name__ == '__main__':
    doctest.testmod()
