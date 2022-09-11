#!/usr/bin/env python
# -*- encoding:UTF-8 -*-
import copy
import doctest
from enum import Enum
import random
from collections import defaultdict, deque
import re

from numpy import number

# from module_1_3 import Stack


class Graph(object):

    """
      Undirected graph implementation. The cost of space is proportional to O(V + E)
    (V is the number of vertices and E is the number of edges). Adding
    an edge only takes constant time. The running time of
    Checking if node v is adjacent to w and traveling all adjacent point of v
    is related to the degree of v. This implementation supports multiple
    input data types(immutable).
    TODO: Test file input.
    >>> g = Graph()
    >>> test_data = [(0, 5), (4, 3), (0, 1), (9, 12), (6, 4), (5, 4), (0, 2),  # from book tinyG.txt
    ...              (11, 12), (9, 10), (0, 6), (7, 8), (9, 11), (5, 3)]
    >>> for a, b in test_data:
    ...     g.add_edge(a, b)
    ...
    >>> g.vertices_size()
    13
    >>> len(test_data) == g.edges_size()
    True
    >>> adjacent_vertices = ' '.join([str(v) for v in g.get_adjacent_vertices(0)])
    >>> adjacent_vertices
    '5 1 2 6'
    >>> g.degree(0)
    4
    >>> g.degree(9)
    3
    >>> g.max_degree()
    4
    >>> g.number_of_self_loops()
    0
    >>> g
    13 vertices, 13 edges
    0: 5 1 2 6
    5: 0 4 3
    4: 3 6 5
    3: 4 5
    1: 0
    9: 12 10 11
    12: 9 11
    6: 4 0
    2: 0
    11: 12 9
    10: 9
    7: 8
    8: 7
    <BLANKLINE>
    >>> g2 = Graph(graph=g)
    >>> g2.add_edge(4, 9)
    >>> g.has_edge(4, 9)
    False
    >>> g2.has_edge(4, 9)
    True
    >>> g2.has_edge(9, 4)
    True
    >>> g2.add_edge(4, 9)
    >>> [i for i in g2.get_adjacent_vertices(4)]
    [3, 6, 5, 9]
    """

    def __init__(self, graph=None) -> None:
        self._edges_size = 0
        self._neighbours = defaultdict(list)

        if graph:
            self._neighbours = copy.deepcopy(graph._neighbours)
            self._edges_size = graph.edges_size()

    def vertices_size(self):
        return len(self._neighbours.keys())

    def edges_size(self):
        return self._edges_size

    def remove_edge(self, vertext_a, vertext_b):
        if not self.has_edge(vertext_a, vertext_b) or vertext_a == vertext_b:
            return

        self._neighbours[vertext_a].remove(vertext_b)
        self._neighbours[vertext_b].remove(vertext_a)

        self._edges_size -= 1

    def add_edge(self, vertext_a, vertext_b):
        # 4.1.5 practice, no self cycle or parallel edges.
        if self.has_edge(vertext_a, vertext_b) or vertext_a == vertext_b:
            return

        self._neighbours[vertext_a].append(vertext_b)
        self._neighbours[vertext_b].append(vertext_a)

        self._edges_size += 1

    # 4.1.4 practice, add has_edge method
    def has_edge(self, vertext_a, vertext_b):
        if vertext_a not in self._neighbours or vertext_b not in self._neighbours:
            return False

        edge = next(
            (i for i in self._neighbours[vertext_a] if i == vertext_b), None)
        return edge is not None

    def get_adjacent_vertices(self, vertex):
        return self._neighbours[vertex]

    def vertices(self):
        return self._neighbours.keys()

    def degree(self, v):
        assert v in self._neighbours
        return len(self._neighbours[v])

    def max_degree(self):
        result = 0

        for w in self._neighbours:
            result = max(result, self.degree(w))

        return result

        return max([self.degrees(w) for w in self._neighbours])

    def avg_degree(self):
        return float(2 * self._edges_size) / self.vertices_size()

    def number_of_self_loops(self):
        count = 0

        for vertex in self._neighbours.keys():
            for neighbour in self._neighbours[vertex]:
                if neighbour == vertex:
                    counnt += 1

        return int(count / 2)

    # 4.1.31 check the number of parallel edges with linear running time.
    def number_of_parallel_edges(self):
        count = 0

        for vertex in self._neighbours.keys():
            distinct = set(self._neighbours[vertex])
            size = len(self._neighbours[vertex])

            if size != distinct:
                count += size - distinct

        return int(count / 2)

    def __repr__(self):
        s = str(self.vertices_size()) + ' vertices, ' + \
            str(self._edges_size) + ' edges\n'
        for k in self._neighbours:
            try:
                lst = ' '.join([vertex for vertex in self._neighbours[k]])
            except TypeError:
                lst = ' '.join([str(vertex) for vertex in self._neighbours[k]])
            s += '{}: {}\n'.format(k, lst)
        return s


class DepthFirstPaths(object):

    """
      Undirected graph depth-first searching algorithms implementation.
    Depth-First-Search recurvisely reaching all vertices that are adjacent to it,
    and then treat these adjacent_vertices as start_vertex and searching again util all the
    connected vertices is marked.
    >>> g = Graph()
    >>> test_data = [(0, 5), (2, 4), (2, 3), (1, 2), (0, 1), (3, 4), (3, 5), (0, 2)]
    >>> for a, b in test_data:
    ...     g.add_edge(a, b)
    ...
    >>> dfp = DepthFirstPaths(g,  0)
    >>> [dfp.has_path_to(i) for i in range(6)]
    [True, True, True, True, True, True]
    >>> [i for i in dfp.path_to(4)]
    [0, 5, 3, 2, 4]
    >>> [i for i in dfp.path_to(1)]
    [0, 5, 3, 2, 1]
    """

    def __init__(self, graph, start_vertex):
        self._marked = defaultdict(bool)
        self._edge_to = {}
        self._start = start_vertex
        self.dfs(graph, self._start)

    def dfs(self, graph, vertex):
        self._marked[vertex] = True

        for v in graph.get_adjacent_vertices(vertex):
            if not self._marked[v]:
                self._edge_to[v] = vertex
                self.dfs(graph, v)

    def has_path_to(self, vertex):
        return self._marked[vertex]

    def vertices_size(self):
        return len(self._marked.keys())

    def path_to(self, vertex):
        if not self.has_path_to(vertex):
            return None

        current_node = vertex
        path = deque([])

        while current_node != self._start:
            path.appendleft(current_node)

            current_node = self._edge_to[current_node]

        path.appendleft(self._start)
        return path


class BreadthFirstPaths(object):

    """
      Breadth-First-Search algorithm implementation. This algorithm
    uses queue as assist data structure. First enqueue the start_vertex,
    marked it as visited and dequeue the vertex, then marked all the
    adjacent vertices of start_vertex and enqueue them. Continue this process
    util all connected vertices are marked.
      With Breadth-First-Search algorithm, we can find the shortest path from x to y.
    The worst scenario of running time is proportional to O(V + E) (V is the number
    of vertices and E is the number of edges).
    >>> g = Graph()
    >>> test_data = [(0, 5), (2, 4), (2, 3), (1, 2), (0, 1), (3, 4), (3, 5), (0, 2)]
    >>> for a, b in test_data:
    ...     g.add_edge(a, b)
    ...
    >>> bfp = BreadthFirstPaths(g, 0)
    >>> [bfp.has_path_to(i) for i in range(6)]
    [True, True, True, True, True, True]
    >>> [i for i in bfp.path_to(4)]
    [0, 2, 4]
    >>> [i for i in bfp.path_to(5)]
    [0, 5]
    >>> bfp.dist_to(4)
    2
    >>> bfp.dist_to(5)
    1
    >>> bfp.dist_to('not a vertex')
    -1
    """
    def __init__(self, graph, start_vertex):
        self._marked = defaultdict(bool)
        self._edge_to = {}

        self._start = start_vertex
        self._dist = {start_vertex: 0}
        self.bfs(graph, self._start)

    def bfs(self, graph: Graph, start):
        queue = deque([start])
        self._marked[start] = True

        while queue:
            current = queue.popleft()
            
            for neighbour in graph.get_adjacent_vertices(current):
                if self._marked[neighbour]:
                    continue
                
                self._marked[neighbour] = True
                self._edge_to[neighbour] = current
                self._dist[neighbour] = self._dist[current] + 1
                queue.append(neighbour)

    def has_path_to(self, vertex):
        return self._marked[vertex]

    def path_to(self, vertex):
        if vertex not in self._edge_to:
            return None

        result = deque([])
        current = vertex

        while current != self._start:
            result.appendleft(current)
            current = self._edge_to[current]
        
        result.appendleft(self._start)

        return result
    
    # 4.1.13 practice, implement dist_to method which only takes constant time.
    def dist_to(self, vertex):
        return self._dist.get(vertex, -1)

    def max_distance(self):
        return max(self._dist.values())

class ConnectedComponent(object):

    """
      Construct connected components using Depth-First-Search algorithm.
    Using this algorithm we need to construct all the connected components
    from the beginning which the cost of running time and space are both
    proportional to O(V + E). But it takes only constant time for querying
    if two vertices are connected.
    >>> g = Graph()
    >>> test_data = [(0, 5), (4, 3), (0, 1), (9, 12), (6, 4), (5, 4), (0, 2),
    ...              (11, 12), (9, 10), (0, 6), (7, 8), (9, 11), (5, 3)]
    >>> for a, b in test_data:
    ...     g.add_edge(a, b)
    ...
    >>> cc = ConnectedComponent(g)
    >>> cc.connected(0, 8)
    False
    >>> cc.connected(0, 4)
    True
    >>> cc.connected(0, 9)
    False
    >>> cc.vertex_id(0)
    0
    >>> cc.vertex_id(7)
    2
    >>> cc.vertex_id(11)
    1
    >>> cc.count()
    3
    """
    def __init__(self, graph: Graph):
        self._marked = defaultdict(bool)
        self._id = defaultdict(int)
        self._count = 0

        for vertex in graph.vertices():
            if self._marked[vertex]:
                continue
        
            self.dfs(graph, vertex)
            self._count += 1

    def dfs(self, graph: Graph, vertex):
        self._marked[vertex] = True
        self._id[vertex] = self._count

        for neighbour in graph.get_adjacent_vertices(vertex):
            if self._marked[neighbour]:
                continue

            self.dfs(graph, neighbour)

    def connected(self, vertex_1, vertex_2):
        return self._id[vertex_1] == self._id[vertex_2]

    def vertex_id(self, vertex):
        return self._id[vertex]

    def count(self):
        return self._count

class State(Enum):
    NOT_VISITED = 0
    VISITING = 1
    VISITED = 2

class Cycle(object):

    """
    Using Depth-First-Search algorithm to check whether a graph has a cycle.
    if a graph is tree-like structure(no cycle), then has_cycle is never reached.
    >>> g = Graph()
    >>> test_data = [(0, 1), (0, 2), (0, 6), (0, 5), (3, 5), (6, 4)]
    >>> for a, b in test_data:
    ...     g.add_edge(a, b)
    ...
    >>> cycle = Cycle(g)
    >>> cycle.has_cycle()
    False
    >>> g2 = Graph()
    >>> has_cycle_data = [(0, 1), (0, 2), (0, 6), (0, 5), (3, 5), (6, 4), (3, 4)]
    >>> for a, b in has_cycle_data:
    ...     g2.add_edge(a, b)
    ...
    >>> cycle2 = Cycle(g2)
    >>> cycle2.has_cycle()
    True
    """

    def __init__(self, graph: Graph):
        self._marked = defaultdict(bool)
        self._has_cycle = False

        for vertex in graph.vertices():
            if not self._marked[vertex]:
                self.dfs(graph, vertex, vertex)

    def dfs(self, graph: Graph, vertex, parent):
        self._marked[vertex] = True

        for neighbour in graph.get_adjacent_vertices(vertex):
            if not self._marked[neighbour]:
                self.dfs(graph, neighbour, vertex)
            else:
                if neighbour != parent:
                    self._has_cycle = True

    def has_cycle(self):
        return self._has_cycle

class Color(Enum):
    RED = 0
    BLUE = 1

class TwoColor(object):

    """
    Using Depth-First-Search algorithm to solve Two-Color problems.
    >>> g = Graph()
    >>> test_data = [(0, 5), (2, 4), (2, 3), (1, 2), (0, 1), (3, 4), (3, 5), (0, 2)]
    >>> for a, b in test_data:
    ...     g.add_edge(a, b)
    ...
    >>> tc = TwoColor(g)
    >>> tc.is_bipartite()
    False
    >>> g = Graph()
    >>> test_data = [(0, 1), (0, 2), (1, 3)]
    >>> for a, b in test_data:
    ...     g.add_edge(a, b)
    ...
    >>> tc = TwoColor(g)
    >>> tc.is_bipartite()
    True
    """

    def __init__(self, graph):
        self._marked = defaultdict(bool)
        self._color = defaultdict(bool)
        self._is_twocolorable = True

        for vertex in graph.vertices():
            if not self._marked[vertex]:
                self.dfs(graph, vertex)

    def dfs(self, graph, vertex):
        self._marked[vertex] = True
        for v in graph.get_adjacent_vertices(vertex):
            if not self._marked[v]:
                self._color[v] = not self._color[vertex]
                self.dfs(graph, v)
            else:
                if self._color[v] == self._color[vertex]:
                    self._is_twocolorable = False

    def is_bipartite(self):
        return self._is_twocolorable

# 4.1.16 practice, implement GraphProperties class.
class GraphProperties(object):

    """
    >>> g = Graph()
    >>> test_data = [(0, 5), (2, 4), (2, 3), (1, 2), (0, 1), (3, 4), (3, 5), (0, 2)]
    >>> for a, b in test_data:
    ...     g.add_edge(a, b)
    ...
    >>> gp = GraphProperties(g)
    >>> gp.eccentricity(0)
    2
    >>> gp.eccentricity(1)
    2
    >>> gp.diameter()
    2
    >>> gp.radius()
    2
    """

    def __init__(self, graph: Graph):
        self._eccentricities = {}
        self._diameter = 0
        self._radius = 9999999999

        dfp = DepthFirstPaths(graph, random.sample(graph.vertices(), 1)[0])
        if dfp.vertices_size() != graph.vertices_size():
            raise Exception('graph is not connected.')

        for vertex in graph.vertices():
            bfp = BreadthFirstPaths(graph, vertex)
            dist = bfp.max_distance()

            if dist < self._radius:
                self._radius = dist
            if dist > self._diameter:
                self._diameter = dist
            self._eccentricities[vertex] = dist
    
    def eccentricity(self, vertex):
        return self._eccentricities.get(vertex, -1)

    def diameter(self):
        return self._diameter

    def radius(self):
        return self._radius

    def center(self):
        centers = [k for k, v in self._eccentricities.items() if v == self._radius]
        random.shuffle(centers)
        return centers[0]

    # 4.1.17 practice
    def girth(self):
        pass

g = Graph()
test_data = [(0, 5), (2, 4), (2, 3), (1, 2), (0, 1), (3, 4), (3, 5), (0, 2)]
for a, b in test_data:
    g.add_edge(a, b)

gp = GraphProperties(g)

if __name__ == '__main__':
    doctest.testmod()

#########################################
#
#   CYCLE
#
#########################################

g = Graph()
test_data = [(0, 1), (0, 2), (0, 6), (0, 5), (3, 5), (6, 4)]
for a, b in test_data:
    g.add_edge(a, b)

g2 = Graph()
has_cycle_data = [(0, 1), (0, 2), (0, 6), (0, 5), (3, 5), (6, 4), (3, 4)]
for a, b in has_cycle_data:
    g2.add_edge(a, b)


def cycle(graph: Graph, visited, root, parent):
    visited[root] = True
		    
    for neighbour in graph.get_adjacent_vertices(root):
        if visited[neighbour] == True and neighbour != parent:
            return True
        elif visited[neighbour] == False:
            if cycle(graph, visited, neighbour, root):
                return True
    
    return False

def has_cycle(g: Graph):
    visited = defaultdict(bool)

    for vertex in g.vertices():
        if not visited.get(vertex, False):
            res = cycle(g, visited, vertex, -1)
            
            if res:
                return True
    
    return False

#########################################
#
#   CYCLE
#
#########################################

# print(has_cycle(g))
# print(has_cycle(g2))