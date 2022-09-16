
from collections import deque
import doctest
from typing import List

from numpy import place
from module_4_5 import FlowNetwork, FordFulkerson, FlowEdge
from module_4_2 import Digraph, DirectedDFS, BreadthFirstOrder
from basic_data_struct import GenericUnionFind

class FattestPath(object):
    """
    Fattest path. Given an edge-weighted digraph and two vertices ss and tt, 
    design an ElogE algorithm to find a fattest path from s to t. 
    The bottleneck capacity of a path is the minimum weight of an edge on the path. 
    A fattest path is a path such that no other path has a higher bottleneck capacity.

    sort all edges in descending order, begin with a graph with only s and t, 
    add one edge at a time to the graph successively, check whether s and t are connected, 
    repeat until connected. As for how to check connectivity:using quick union

    >>> edges = [
    ...     FlowEdge(v='C', w='A', flow=10,capacity=10), FlowEdge(v='C', w='D', flow=5, capacity=5), FlowEdge(v='C', w='G', flow=10,capacity=15),
    ...     FlowEdge(v='A', w='B', flow=5, capacity=9), FlowEdge(v='A', w='E', flow=5, capacity=15), FlowEdge(v='A', w='D', flow=0, capacity=4),
    ...     FlowEdge(v='D', w='E', flow=5, capacity=8), FlowEdge(v='D', w='G', flow=0, capacity=4),
    ...     FlowEdge(v='G', w='H', flow=10,capacity=16),
    ...     FlowEdge(v='B', w='E', flow=0, capacity=15), FlowEdge(v='B', w='F', flow=5, capacity=10),
    ...     FlowEdge(v='E', w='F', flow=10,capacity=10), FlowEdge(v='E', w='H', flow=0, capacity=15),
    ...     FlowEdge(v='H', w='D', flow=0, capacity=6), FlowEdge(v='H', w='F', flow=10,capacity= 10),
    ... ]

    >>> fp = FattestPath(edges, 'C', 'F')
    >>> fp.fattest_path()
    deque([[C ==(10/10)==> A], [A ==(8/9)==> B], [B ==(8/10)==> F]])
    """
    def __init__(self, edges: List[FlowEdge], source, target) -> None:
        self._edges = deque(sorted(edges, key=lambda edge: edge.capacity() - edge.flow(), reverse=True))
        self._is_connected = False
        self._fattest_edge = []
        self._graph = Digraph()
       
        unionFind = GenericUnionFind()

        while not unionFind.connected(source, target) and len(self._edges):
            edge = self._edges.popleft()

            unionFind.union(edge.start(), edge.end())
            self._fattest_edge.append(edge)

        if unionFind.connected(source, target):
            self._is_connected = True
        else:
            return

        network = FlowNetwork()
        graph = Digraph()

        for edge in edges:
            network.add_edge(edge)
            graph.add_edge(edge.start(), edge.end())

        self._fordfulkerson = FordFulkerson(graph=network, source=source, target=target)
        self._bfs = BreadthFirstOrder(graph, source)

        path = self._bfs.path_to(target)
        self._path = deque([])

        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            edge = [edge for edge in edges if start == edge.start() and end == edge.end()]
            self._path.append(edge)

    def fattest_path(self):
        return self._path

class BipartiteMatching(object):
    """
    Perfect matchings in k-regular bipartite graphs. 
    Suppose that there are n men and n women at a dance and that each man knows exactly k
    women and each woman knows exactly k men (and relationships are mutual). 
    Show that it is always possible to arrange a dance so that each man and woman are matched with someone they know.

    build a graph with source s connected to n men, which is connected the n women they know respectively, 
    all the women are connected to t, and compute the maxflow as shown in the job seeker example in lecture. 
    For rigorous proof, refer to:https://math.stackexchange.com/questions/1805181/prove-that-a-k-regular-bipartite-graph-has-a-perfect-matching/1805195
    
    >>> students = ['Alice', 'Bob', 'Dave', 'Carol', 'Eliza']
    >>> companies = ['Adobe', 'Amazon', 'Google', 'Facebook', 'Yahoo']
    >>> applications = [
    ...     #ALICE
    ...     FlowEdge('Alice', 'Adobe', 0, (1 << 63) - 1),
    ...     FlowEdge('Alice', 'Amazon', 0, (1 << 63) - 1),
    ...     FlowEdge('Alice', 'Google', 0, (1 << 63) - 1),
    ...     #BOB
    ...     FlowEdge('Bob', 'Adobe', 0, (1 << 63) - 1),
    ...     FlowEdge('Bob', 'Amazon', 0, (1 << 63) - 1),
    ...     #CAROL
    ...     FlowEdge('Carol', 'Adobe', 0, (1 << 63) - 1),
    ...     FlowEdge('Carol', 'Facebook', 0, (1 << 63) - 1),
    ...     FlowEdge('Carol', 'Google', 0, (1 << 63) - 1),
    ...     #DAVE
    ...     FlowEdge('Dave', 'Yahoo', 0, (1 << 63) - 1),
    ...     FlowEdge('Dave', 'Amazon', 0, (1 << 63) - 1),
    ...     #ELIZE
    ...     #DAVE
    ...     FlowEdge('Eliza', 'Yahoo', 0, (1 << 63) - 1),
    ...     FlowEdge('Eliza', 'Amazon', 0, (1 << 63) - 1),
    ... ]

    >>> bb = BipartiteMatching(students, companies, applications)
    
    >>> bb.can_all_students_get_a_job()
    True
    >>> bb.get_placement()
    {'Alice': 'Google', 'Bob': 'Adobe', 'Dave': 'Yahoo', 'Carol': 'Facebook', 'Eliza': 'Amazon'}
    """  

    def __init__(self, students: List[str], companies: List[str], applications: List[FlowEdge]) -> None:
        
        self._students = students
        self._companies = companies
        self._applications = applications

        self._network = FlowNetwork()

        for edge in applications:
            self._network.add_edge(edge)

        for student in students:
            self._network.add_edge(FlowEdge('source', student, 0, 1))

        for company in companies:
            self._network.add_edge(FlowEdge(company, 'target', 0, 1))

        self._ff = FordFulkerson(self._network, 'source', 'target')

    def can_all_students_get_a_job(self):
        return self._ff.max_flow() == len(self._students) == len(self._companies)

    def get_placement(self):
        placement = dict()

        for student in self._students:
            for edge in self._network.get_adjacent_edges(student):
                if edge.flow() == 1 and student == edge.start():
                    placement[edge.start()] = edge.end()

        return placement

if __name__ == '__main__':
    doctest.testmod()
 