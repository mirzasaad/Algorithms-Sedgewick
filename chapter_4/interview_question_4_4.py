

import doctest
from basic_data_struct import IndexMinPQ, Stack
from module_4_4 import DijkstraSP, EdgeWeightedDigraph, INFINITE_POSITIVE_NUMBER, DirectedEdge

#https://www.geeksforgeeks.org/monotonic-shortest-path-from-source-to-destination-in-directed-weighted-graph/
class MonoticShortestPath(DijkstraSP):

    """
      Dijkstra Shortest Path algorithm. First reach the source vertex, 'relax' all the adjacent
    edges of the source vertex, and then put all 'relaxed' edges into the priority queue or
    change the distance from the priority queue util the priority queue is empty. The cost of
    running time is proportional to O(ElogV), and the cost of the space is proportional to O(V).
    This algorithm is not applied to the graph with NEGATIVE edges. The worst case still has good
    performance.
    >>> test_data = [(1, 3, 1.1), (1, 5, 2.0), (1, 6, 3.3), (2, 5, 2.7),
    ...          (3, 4, 2.0), (3, 5, 1.1), (4, 2, 2.3), (5, 6, 2.4), (6, 2, 3.0)]
    >>> ewd = EdgeWeightedDigraph()
    
    >>> for a, b, weight in test_data:
    ...     edge = DirectedEdge(a, b, weight)
    ...     ewd.add_edge(edge)

    >>> start = 1
    >>> end = 2
    >>> msp = MonoticShortestPath(ewd, 1)

    >>> msp.has_path_to(end)
    True
    >>> [i for i in msp.path_to(end)]
    [1->3 1.1, 3->4 2.0, 4->2 2.3]
    >>> msp.dist_to(end)
    5.4
    """

    def __init__(self, graph: EdgeWeightedDigraph, source):
        self._dist_to = dict((v, INFINITE_POSITIVE_NUMBER)
                             for v in graph.vertices())
        self._edge_to = {}
        self._pq = IndexMinPQ(graph.vertices_size())

        self._pq.insert(source, 0.0)
        self._dist_to[source] = 0.0
        self._edge_to[source] = None

        while not self._pq.is_empty():
            self.relax(graph, self._pq.delete_min())

    def relax(self, graph: EdgeWeightedDigraph, vertex):

        for edge in graph.adjacent_edges(vertex):
            end = edge.end

            #NOTE monotionic condition
            if (self._edge_to [edge.start] is None or self._edge_to[edge.start].weight < edge.weight):

                if self._dist_to[end] > self._dist_to[vertex] + edge.weight:
                    self._dist_to[end] = round(
                        self._dist_to[vertex] + edge.weight, 2)
                    self._edge_to[end] = edge

                    if not self._pq.contains(end):
                        self._pq.insert(end, self._dist_to[end])
                    else:
                        self._pq.change_key(end, self._dist_to[end])

class SecondShortestPath(object):

    """
    this is not working EdgeWeightedDigraph has no remove edge func
    https://www.lavivienpost.com/shortest-path-and-2nd-shortest-path-using-dijkstra-code/#:~:text=The%20steps%20are%3A%20first%20find,path%20from%20source%20to%20destination.

    use yen alogorithm for k shortest
    """

    def __init__(self, graph: EdgeWeightedDigraph, source, target):
        self._sp = DijkstraSP(graph, source)
        
        if not self._sp.has_path_to(target):
            return

        edges = self._sp.path_to(target)
        graph = EdgeWeightedDigraph(graph)

        shortest_dist = self._sp.dist_to(target)
        self._second_shortest = shortest_dist
        self._socond_shortest_path = None
        self._has_socond_shortest_path = False

        for edge in edges:
            graph.remove_edge(edge)
            sp = DijkstraSP(graph, source)

            if sp.has_path_to(target):
                if sp.dist_to(target) > shortest_dist:
                    self._second_shortest = sp.dist_to(target)
                    self._socond_shortest_path = sp.path_to(target)
                    self._has_socond_shortest_path = True
            graph.add_edge(edge)

    def has_second_shorted_path(self):
        return self._has_socond_shortest_path
    
class SkippablePath(object):
    """
    use a modified version of Dijkstra's algorithm, maintain an array of max weight of path from s to every vertex, 
    each time we select a new edge and its end node, we choose the one with the shortest path given that we could delete the max weight from path.
    """
    def __init__(self, graph: EdgeWeightedDigraph, source, target) -> None:
        spaths = DijkstraSP(graph, source)
        tpaths = DijkstraSP(graph.reverse_graph(), target)

        self._minimum = float('inf')
        self._skippable = None
        
        for edge in graph.edges():
            v, w = edge.start, edge.end

            if spaths.dist_to(v) + tpaths.dist_to(w) < self._minimum:
                self._skippable = edge
                self._minimum = spaths.dist_to(v) + tpaths.dist_to(w)

        
        self._skippablepath = Stack()
        tmp = Stack()

        for e in tpaths.path_to(self._skippable.end):
            self._skippablepath.push(e)

        self._skippablepath.push(self._skippable)

        for e in spaths.path_to(self._skippable.start):
            tmp.push(e)

        for e in tmp:
            self._skippablepath.push(e)

    def skippablepath(self):
        return self._skippablepath

    def skippable_edge(self):
        return self._skippable

if __name__ == '__main__':
    doctest.testmod()
