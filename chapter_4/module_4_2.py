import copy
import doctest
from collections import defaultdict, deque
from basic_data_struct import Stack, Queue

"""
Shortest directed cycle. Given a digraph G, design an efficient algorithm to find a directed cycle with the minimum number of edges (or report that the graph is acyclic). The running time of your algorithm should be at most proportional to V(E+V) and use space proportional to E+V, where V is the number of vertices and E is the number of edges.

run BFS from each vertex

Hamiltonian path in a DAG. Given a directed acyclic graph, design a linear-time algorithm to determine whether it has a Hamiltonian path (a simple path that visits every vertex), and if so, find one.

if more than one vertex has 0 in degree or more than one vertex has 0 out degree, then none, else topological sort from the vertex that has 0 in degree, when encountering branches, record the vertex we randomly shift to, if we reach the termination points and stil have unvisited vertices, we should return back to same vertex with multiple children again, when we recurse on another vertex and check if it will end on the vertex posterior to the one we record, if not, then no.
or https://stackoverflow.com/questions/16124844/algorithm-for-finding-a-hamilton-path-in-a-dag

Reachable vertex.
DAG: Design a linear-time algorithm to determine whether a DAG has a vertex that is reachable from every other vertex, and if so, find one.
Digraph: Design a linear-time algorithm to determine whether a digraph has a vertex that is reachable from every other vertex, and if so, find one.

DAG:if it's reachable from every other vertex, then it must have 0 out degree, run dfs from this vertex on the reverse graph
Digraph:compute the strongly connected components(Kosaraju's algorithm), run dfs on random vertex of the last scc, and check if it can reach every other scc.
"""

# NOTE there is a difference between tringly connected and transitive close for directedGraph but means same thing for unidrected graph
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

class DirectedDFS(object):

    """
      Depth-First-Search algorithm with directed graph, which can solve directed
    graph reachable problem.
    >>> graph = Digraph()
    >>> test_data = [(4, 2), (2, 3), (3, 2), (6, 0), (0, 1), (2, 0),
    ...              (11, 12), (12, 9), (9, 10), (9, 11), (8, 9), (10, 12),
    ...              (11, 4), (4, 3), (3, 5), (7, 8), (8, 7), (5, 4), (0, 5),
    ...              (6, 4), (6, 9), (7, 6)]
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    ...
    >>> dfs = DirectedDFS(graph, 1)
    >>> [i for i in graph.vertices() if dfs.marked(i)]
    [1]
    >>> dfs1 = DirectedDFS(graph, 2)
    >>> [i for i in graph.vertices() if dfs1.marked(i)]
    [0, 1, 2, 3, 4, 5]
    >>> dfs2 = DirectedDFS(graph, 1, 2, 6)
    >>> [i for i in graph.vertices() if dfs2.marked(i)]
    [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12]
    """

    # multiple sources for dfs
    def __init__(self, graph, *sources):
        self._marked = defaultdict(bool)
        self._edgeTo = defaultdict()
        self._graph = graph

        for vertex in sources:
            if not self._marked[vertex]:
                self.dfs(graph, vertex)

    def dfs(self, graph: Digraph, vertex):
        self._marked[vertex] = True

        for neighour in graph.get_adjacent_vertices(vertex):
            if not self._marked[neighour]:
                self._edgeTo[neighour] = vertex 
                self.dfs(graph, neighour)

    def marked(self, vertex):
        return self._marked[vertex]

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

class DepthFirstOrder(object):

    def __init__(self, graph: Digraph):
        self._pre = Queue()
        self._post = Queue()
        self._reverse_post = Stack()
        self._marked = defaultdict(bool)

        # self.dfs(graph, 0)
        for vertex in graph.vertices():
            if not self._marked[vertex]:
                self.dfs(graph, vertex)

    def dfs(self, graph: Digraph, vertex):
        self._pre.enqueue(vertex)
        self._marked[vertex] = True

        for neighbour in graph.get_adjacent_vertices(vertex):
            if not self._marked[neighbour]:
                self.dfs(graph, neighbour)
        
        self._post.enqueue(vertex)
        self._reverse_post.push(vertex)

    def prefix(self):
        return self._pre

    def postfix(self):
        return self._post

    def reverse_postfix(self):
        return self._reverse_post

class BreadthFirstOrder(object):

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
    >>> bfp = BreadthFirstOrder(g, 0)
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

    def bfs(self, graph: Digraph, start):
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

# NOTE graph has to be asyclic and, and the topological order is reverst_posstfix
class Topological(object):

    """
      Topological-Sorting implementation. Topological-Sorting
    has to be applied on a directed acyclic graph. If there is
    an edge u->w, then u is before w. This implementation is using
    Depth-First-Search algorithm, for any edge v->w, dfs(w)
    will return before dfs(v), because the input graph should
    not contain any cycle.
      Another Topological-Sorting implementation is using queue to
    enqueue a vertex which indegree is 0. Then dequeue and marked
    it, enqueue all its adjacent vertex util all the vertices in the
    graph is marked. This implementation is not given.
    >>> test_data = [(2, 3), (0, 6), (0, 1), (2, 0), (11, 12),
    ...              (9, 12), (9, 10), (9, 11), (3, 5), (8, 7),
    ...              (5, 4), (0, 5), (6, 4), (6, 9), (7, 6)]
    >>> graph = Digraph()
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    ...
    >>> topo = Topological(graph)
    >>> topo.is_DAG()
    True
    >>> [i for i in topo.order()]
    [8, 7, 2, 3, 0, 5, 1, 6, 9, 11, 10, 12, 4]
    """

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

# NOTE
"""
We have been careful to maintain a distinction between reachability in digraphs and connectivity in undirected graphs. 
In an undirect- ed graph, two vertices v and w are connected if there is a path connecting them—we 
can use that path to get from v to w or to get from w to v. In a digraph, by contrast, a vertex w is reachable 
from a vertex v if there is a directed path from v to w, but there may or may not be a directed path back to v 
from w. To complete our study of digraphs, we consider the
Definition. Two vertices v and w are strongly connected if they are mutually reachable: that is, if there is 
a directed path from v to w and a directed path from w to v. A digraph is strongly connected if all its 
vertices are strongly connected to one another.
Strongly connected digraphs natural analog of connectivity in undirected graphs.
"""
class KosarajuSCC(object):

    """
    # NOTE we dont need asyclic graph for reverse_postfix, it is only required for toplogical order
    for strongly components we need cycles, and there is differece in sstrongly connected and transitive closure
    find the reverse postfix of reverse graph
    traverse over reverse_postfix of topological sort and marked verteces same as undirected graph
    >>> test_data = ((4, 2), (2, 3), (3, 2), (6, 0), (0, 1), (2, 0),
    ...              (11, 12), (12, 9), (9, 10), (9, 11), (7, 9), (10, 12),
    ...              (11, 4), (4, 3), (3, 5), (6, 8), (8, 6), (5, 4), (0, 5),
    ...              (6, 4), (6, 9), (7, 6))
    >>> graph = Digraph()
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    ...
    >>> scc = KosarajuSCC(graph)
    >>> count = scc.count()
    >>> output = defaultdict(Queue)
    >>> for v in sorted(graph.vertices()):
    ...     output[scc.vertex_id(v)].enqueue(v)
    ...
    >>> ['{}: {}'.format(k, ', '.join(map(str, v))) for k, v in output.items()]
    ['1: 0, 2, 3, 4, 5', '0: 1', '3: 6, 8', '4: 7', '2: 9, 10, 11, 12']
    >>> topo = Topological(graph)
    >>> topo.is_DAG()
    False
    """
    def __init__(self, graph: Digraph):
        self._marked = defaultdict(bool)
        self._id = {}
        self._count = 0
        order = DepthFirstOrder(graph.reverse_graph())

        for v in order.reverse_postfix():
            if not self._marked[v]:
                self.dfs(graph, v)
                self._count += 1

    def dfs(self, graph, vertex):
        self._marked[vertex] = True
        self._id[vertex] = self._count
        for v in graph.get_adjacent_vertices(vertex):
            if not self._marked[v]:
                self.dfs(graph, v)

    def strongly_connected(self, vertex_1, vertex_2):
        return self._id[vertex_1] == self._id[vertex_2]

    def vertex_id(self, vertex):
        return self._id[vertex]

    def count(self):
        return self._count

class TransitiveClosure(object):

    """
      This class can check if v is reachable
    from w in a directed graph using DirectedDFS.
    The cost of running time is proportional to
    O(V(V + E)), and the cost of space is proportional
    to O(V*V), so this is not a good solution for
    large scale graphs.
    >>> test_data = ((4, 2), (2, 3), (3, 2), (6, 0), (0, 1), (2, 0),
    ...              (11, 12), (12, 9), (9, 10), (9, 11), (7, 9), (10, 12),
    ...              (11, 4), (4, 3), (3, 5), (6, 8), (8, 6), (5, 4), (0, 5),
    ...              (6, 4), (6, 9), (7, 6))
    >>> graph = Digraph()
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    ...
    >>> tc = TransitiveClosure(graph)
    >>> tc.reachable(1, 5)
    False
    >>> tc.reachable(1, 0)
    False
    >>> tc.reachable(0, 1)
    True
    >>> tc.reachable(0, 9)
    False
    >>> tc.reachable(8, 12)
    True
    """
    def __init__(self, graph: Digraph):
        self._all = {}

        for vertex in graph.vertices():
            self._all[vertex] = DirectedDFS(graph, vertex)

    def reachable(self, start, end):
        return self._all[start].marked(end)

# 4.2.7 practice, implement Degrees class
# which compute degrees of vertices in a directed graph.
class Degrees(object):

    """
    >>> test_data = ((4, 2), (2, 3), (3, 2), (6, 0), (0, 1), (2, 0),
    ...              (11, 12), (12, 9), (9, 10), (9, 11), (7, 9), (10, 12),
    ...              (11, 4), (4, 3), (3, 5), (6, 8), (8, 6), (5, 4), (0, 5),
    ...              (6, 4), (6, 9), (7, 6))
    >>> graph = Digraph()
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    ...
    >>> degree = Degrees(graph)
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

# 4.2.20 practice, check if euler cycle exists.
class Euler(object):

    """
    https://www.techiedelight.com/eulerian-path-directed-graph/
    Euler cycle(circuit) exists if graph is strongly connected, and all indegree and out degree for all the vertices are same
    Euler path exisits if 2 of vertices has outdegree - indegree == 1
    use 
    >>> test_data = ((4, 2), (2, 3), (3, 2), (6, 0), (0, 1), (2, 0),
    ...              (11, 12), (12, 9), (9, 10), (9, 11), (7, 9), (10, 12),
    ...              (11, 4), (4, 3), (3, 5), (6, 8), (8, 6), (5, 4), (0, 5),
    ...              (6, 4), (6, 9), (7, 6))
    >>> graph = Digraph()
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    ...
    >>> euler = Euler(graph)
    >>> euler.is_euler_cycle_exists()
    False
    >>> euler.is_DAG()
    False
    >>> euler.is_graph_strongly_connected()
    False
    >>> test_data = (0, 1), (1, 2), (2, 3), (3, 1), (1, 4), (4, 3), (3, 0), (0, 5), (5, 4)
    >>> graph = Digraph()
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    >>> euler = Euler(graph)
    >>> euler.is_graph_strongly_connected()
    True
    >>> euler.is_DAG()
    False
    >>> euler.is_euler_cycle_exists()
    False
    >>> euler.is_euler_path_exists()
    True
    >>> test_data = [(1, 0), (0, 2), (2, 1), (0, 3), (3, 4), (4, 0)]
    >>> graph = Digraph()
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    >>> euler = Euler(graph)
    >>> euler.is_graph_strongly_connected()
    True
    >>> euler.is_DAG()
    False
    >>> euler.is_euler_cycle_exists()
    True
    >>> euler.is_euler_path_exists()
    False
    """

    def __init__(self, graph):
        self._graph = graph

        self._indegree = defaultdict(int)
        self._outdegree = defaultdict(int)
        length = 0
        for v in graph.vertices():
            length += 1
            for adj in graph.get_adjacent_vertices(v):
                self._indegree[adj] += 1
                self._outdegree[v] += 1

        self._euler_cycle_exists = len([k for k, v in self._indegree.items()
                                        if self._outdegree[k] == v]) == length

    def is_euler_path_exists(self):
        """
        The following loop checks the following conditions to determine if an
        Eulerian path can exist or not:
            a. At most one vertex in the graph has `out-degree = 1 + in-degree`.
            b. At most one vertex in the graph has `in-degree = 1 + out-degree`.
            c. Rest all vertices have `in-degree == out-degree`.
 
        If either of the above condition fails, the Euler path can't exist.
        """
        if self._euler_cycle_exists:
            assert not self.is_DAG()
            assert self.is_graph_strongly_connected()
        
        start_vertex = [vertex for vertex, out_degree in self._outdegree.items() if out_degree - self._indegree[vertex] == 1]
        end_vertex = [vertex for vertex, in_degree in self._indegree.items() if in_degree - self._outdegree[vertex] == 1]

        if len(start_vertex) != 1 and len(end_vertex) != 1:
            return False

        return True

    def is_euler_cycle_exists(self):
        """
        the graph has to be cycle
        the graph has one strong component
        all verteces has to have same out and in degreess
        """
        if self._euler_cycle_exists:
            assert not self.is_DAG()
            assert self.is_graph_strongly_connected()
        return self._euler_cycle_exists

    def is_DAG(self):
        cycle_finder = DirectedCycle(self._graph)
        return not cycle_finder.has_cycle

    def is_graph_strongly_connected(self):
        scc = KosarajuSCC(self._graph)
        return scc.count() == 1
# 4.2.24 practice, check if a graph contains hamilton path,
# the following step is very simple and is given in the book.
def hamilton_path_exists(graph):
    """
    >>> test_data = [(2, 3), (0, 6), (0, 1), (2, 0), (11, 12),
    ...              (9, 12), (9, 10), (9, 11), (3, 5), (8, 7),
    ...              (5, 4), (0, 5), (6, 4), (6, 9), (7, 6)]
    >>> graph = Digraph()
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    ...
    >>> graph = Digraph()
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    ...
    >>> hamilton_path_exists(graph)
    False
    >>> graph_2 = Digraph(graph)
    >>> graph_2.add_edge(7, 2)
    >>> graph_2.add_edge(3, 0)
    >>> graph_2.add_edge(12, 1)
    >>> graph_2.add_edge(1, 5)
    >>> graph_2.add_edge(10, 11)
    >>> hamilton_path_exists(graph_2)
    True
    """

    cycle = DirectedCycle(graph)
    if cycle.has_cycle():
        return 'NOT A DAG'

    scc = KosarajuSCC(graph)

    if scc.count() != len(graph.vertices()):
        return 'strong connected component found'

    ts = Topological(graph)
    vertices = [v for v in ts.order()]
    has_path = True
    for i in range(len(vertices) - 1):
        if not graph.has_edge(vertices[i], vertices[i+1]):
            has_path = False
    return has_path

# 4.2.25 practice
def unique_topologial_sort_order(graph):
    return hamilton_path_exists(graph)

# 4.2.30 practice, see http://algs4.cs.princeton.edu/42digraph/TopologicalX.java.html.
class TopologicalWithDegree(object):

    """
    >>> test_data = [(2, 3), (0, 6), (0, 1), (2, 0), (11, 12),
    ...              (9, 12), (9, 10), (9, 11), (3, 5), (8, 7),
    ...              (5, 4), (0, 5), (6, 4), (6, 9), (7, 6)]
    >>> graph = Digraph()
    >>> for a, b in test_data:
    ...     graph.add_edge(a, b)
    ...
    >>> twd = TopologicalWithDegree(graph)
    >>> twd.has_order()
    True
    >>> [v for v in twd.order()]
    [2, 8, 3, 0, 7, 1, 5, 6, 4, 9, 10, 11, 12]
    >>> twd.rank(8)
    1
    >>> twd.rank(10)
    10
    """

    def __init__(self, graph):
        indegree = defaultdict(int)
        self._order = Queue()
        self._rank = defaultdict(int)
        count = 0
        for v in graph.vertices():
            for adj in graph.get_adjacent_vertices(v):
                indegree[adj] += 1
        queue = Queue()
        for v in graph.vertices():
            if indegree[v] == 0:
                queue.enqueue(v)

        while not queue.is_empty():
            vertex = queue.dequeue()
            self._order.enqueue(vertex)
            self._rank[vertex] = count
            count += 1
            for v in graph.get_adjacent_vertices(vertex):
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.enqueue(v)

        if count != graph.vertices_size():
            self._order = None

        assert self.check(graph)

    def has_order(self):
        return self._order is not None

    def order(self):
        return self._order

    def rank(self, vertex):
        if vertex not in self._rank:
            return -1
        return self._rank[vertex]

    def check(self, graph):
        # digraph is acyclic
        if self.has_order():
            # check that ranks provide a valid topological order
            for vertex in graph.vertices():
                # check that vertex has a rank number
                if vertex not in self._rank:
                    return 1
                for adj in graph.get_adjacent_vertices(vertex):
                    if self._rank[vertex] > self._rank[adj]:
                        return 2
            # check that ranks provide a valid topological order
            for index, v in enumerate(self._order):
                if index != self._rank[v]:
                    return 3
            return True
        return False

if __name__ == '__main__':
    doctest.testmod()