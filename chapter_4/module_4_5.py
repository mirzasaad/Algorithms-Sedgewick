
# NOTE
"""
MAX FLOW
augment path is dfs undirected search
every edge has flow / capacity

find augment undirected path from source to sink,
if the edge is forward add to flow of the edge or
if the edge is backward minus from the flow of the edge

mincut capcity == max flow

for mincut first calculate max flow

do a augment search, "look for forward edge which is not full and backward edge which is not empty"

avoid the bad case, look at book or coursera for reference
"""


from collections import defaultdict, deque, namedtuple
import doctest
from random import uniform
from typing import List


class FlowEdge(object):
    """
    forward egde == capicity - flow
    backward edge = flow
    """

    def __init__(self, v=None, w=None, flow=0.0, capacity=None, edge=None):

        if isinstance(v, int) and v < 0:
            raise Exception('vertex index must be a non-negative integer')
        if isinstance(w, int) and w < 0:
            raise Exception('vertex index must be a non-negative integer')
        if capacity and capacity < 0.0:
            raise Exception('Edge capacity must be non-negative')
        if capacity and flow and flow > capacity:
            raise Exception('flow exceeds capacity')
        if flow and flow < 0.0:
            raise Exception('flow must be non-negative')

        self.v = v
        self.w = w
        self._capacity = capacity
        self._flow = flow
        if edge and isinstance(edge, FlowEdge):
            self.v = v
            self.w = w
            self._capacity = capacity
            self._flow = flow

    def start(self):
        return self.v

    def end(self):
        return self.w

    def capacity(self):
        return self._capacity

    def flow(self):
        return self._flow

    def other(self, vertex):
        if vertex == self.v:
            return self.w
        elif vertex == self.w:
            return self.v
        else:
            raise Exception('Invalid Endpoint')

    def residualCapacityTo(self, vertex):
        if vertex == self.v:
            return self.flow()
        elif self.w == vertex:
            return self.capacity() - self.flow()
        else:
            raise Exception('Invalid Endpoint')

    def addResidualFlowTo(self, vertex,  delta):
        if delta < 0.0:
            raise Exception('Delta must be nonnegative')

        if vertex == self.v:
            # backward edge
            self._flow -= delta
        elif vertex == self.w:
            # forward edge
            self._flow += delta
        else:
            raise Exception('Invalid Endpoint')

        # round flow to 0 or capacity if within floating-point precision

        # if abs(self._flow) <= FLOATING_POINT_EPSILON:
        #   self._flow = 0
        # if abs(self._flow - self._capacity) <= FLOATING_POINT_EPSILON:
        #   self._flow = self._capacity;

        if self._flow < 0.0:
            raise Exception('Flow is negative')
        if self._capacity < self._flow:
            raise Exception('Flow Exceeds Capacity')

    def __repr__(self):
        return "%s ==(%s/%s)==> %s" % (self.v, self._flow, self._capacity, self.w)

class FlowNetwork(object):
    def __init__(self, vertices_size=0, random_edges=None):
        if vertices_size < 0:
            raise Exception(
                'Number of vertices in a Graph must be nonnegative')
        
        self._vertices_size = 0
        self._edges_size = 0
        self._adjacent = defaultdict(list)
        self._vertices = set()

        if random_edges:
            for i in range(random_edges):
                v = uniform(0, self.vertices_size())
                w = uniform(0, self.vertices_size())
                
                capacity = uniform(0, 100)
               
                self.add_edge(FlowEdge(v, w, capacity))

    def vertices(self):
        return self._vertices

    def vertices_size(self):
        return len(self._adjacent)

    def edges_size(self):
        return self._edges_size

    def add_edge(self, flowedge: FlowEdge):
        v = flowedge.start()
        w = flowedge.end()
        
        self._adjacent[v].append(flowedge)
        self._adjacent[w].append(flowedge)
        
        self._vertices.add(v)
        self._vertices.add(w)
        self._edges_size += 1

    def get_adjacent_edges(self, v) -> List[FlowEdge]:
        return self._adjacent[v]

    def edges(self) -> List[FlowEdge]:
        edges_list = []

        for vertex in self.vertices():
            for e in self.get_adjacent_edges(vertex):
                if e.end() != vertex:
                    edges_list.append(e)
    
        return edges_list

    def __repr__(self):
        s = str(self.vertices_size()) + ' vertices, ' + \
            str(self._edges_size) + ' edges\n'
        for k in self._adjacent:
            try:
                lst = ' '.join([vertex for vertex in self._adjacent[k]])
            except TypeError:
                lst = ' '.join([str(vertex) for vertex in self._adjacent[k]])
            s += '{}: {}\n'.format(k, lst)
        return s

class FordFulkerson(object):
    """
    >>> edges = [
    ...     FlowEdge(v='C', w='A', flow=10,capacity=10), FlowEdge(v='C', w='D', flow=5, capacity=5), FlowEdge(v='C', w='G', flow=10,capacity=15),
    ...     FlowEdge(v='A', w='B', flow=5, capacity=9), FlowEdge(v='A', w='E', flow=5, capacity=15), FlowEdge(v='A', w='D', flow=0, capacity=4),
    ...     FlowEdge(v='D', w='E', flow=5, capacity=8), FlowEdge(v='D', w='G', flow=0, capacity=4),
    ...     FlowEdge(v='G', w='H', flow=10,capacity=16),
    ...     FlowEdge(v='B', w='E', flow=0, capacity=15), FlowEdge(v='B', w='F', flow=5, capacity=10),
    ...     FlowEdge(v='E', w='F', flow=10,capacity=10), FlowEdge(v='E', w='H', flow=0, capacity=15),
    ...     FlowEdge(v='H', w='D', flow=0, capacity=6), FlowEdge(v='H', w='F', flow=10,capacity= 10),
    ... ]

    >>> network = FlowNetwork()
    >>> for edge in edges:
    ...     network.add_edge(edge)

    >>> fordfulkerson = FordFulkerson(graph=network, source='C', target='F')

    >>> print('MAX FLOW ', fordfulkerson.max_flow())
    MAX FLOW  28
    >>> vertices_in_min_cut = [vertex for vertex in network.vertices() if fordfulkerson.inCut(vertex)]
    >>> edgee_in__min_cut = [edge for edge in network.edges() if edge.start() in vertices_in_min_cut and edge.end() in vertices_in_min_cut]
    >>> min_cut_value = 0

    >>> for vertex in network.vertices():
    ...     for edge in network.get_adjacent_edges(vertex):
    ...         other = edge.other(vertex)
    ...         if vertex == edge.start() and fordfulkerson.inCut(edge.start()) and not fordfulkerson.inCut(edge.end()):
    ...             min_cut_value += edge.capacity()

    >>> print('MIN CUT VERTICES', sorted(vertices_in_min_cut))
    MIN CUT VERTICES ['C', 'D', 'G', 'H']
    >>> print('MIN CUT EDGES', sorted(edgee_in__min_cut, key=lambda edge: edge.capacity()))
    MIN CUT EDGES [D ==(0/4)==> G, C ==(5/5)==> D, H ==(3/6)==> D, C ==(13/15)==> G, G ==(13/16)==> H]
    >>> print('MIN_CUT_VALUE', min_cut_value)
    MIN_CUT_VALUE 28
    """
    def __init__(self, graph=None, source=None, target=None):
        self._marked = defaultdict(bool)
        self._edge_to = defaultdict(FlowEdge)

        self._max_flow = None
        
        self._graph: FlowNetwork = graph
        self._source = source
        self._target = target

        self._result = []
        
        self._max_flow = self._get_excess_flow_from_network(source)

        while self._hasAugmentingPath(graph, source, target):
            bottle_neck = self._findBottleNeckValueInAugmentPath()

            self._augmnet_flow(bottle_neck)
            self._max_flow += bottle_neck
    
    def _hasAugmentingPath(self, graph: FlowNetwork, source, target):
        self._edge_to = defaultdict(FlowEdge)
        self._marked = defaultdict(bool)

        queue = deque([source])

        self._marked[source] = True

        while queue:
            vertex = queue.popleft()

            for edge in graph.get_adjacent_edges(vertex):
                other_vertex = edge.other(vertex)
                
                if edge.residualCapacityTo(other_vertex) > 0 and not self._marked[other_vertex]:

                    self._edge_to[other_vertex] = edge
                    self._marked[other_vertex] = True

                    queue.append(other_vertex)
        
        return self._marked[target]
    
    def _findBottleNeckValueInAugmentPath(self):
        temp = self._target
        bottle_neck_flow = float('inf')

        while temp != self._source:
            bottle_neck_flow = min(bottle_neck_flow, self._edge_to[temp].residualCapacityTo(temp))    
            temp = self._edge_to[temp].other(temp)

        return bottle_neck_flow

    def _augmnet_flow(self, bottle_neck_value):
        temp = self._target

        while temp != self._source:
            self._edge_to[temp].addResidualFlowTo(temp, bottle_neck_value)
            temp = self._edge_to[temp].other(temp)

    def value(self):
        return self._max_flow

    def inCut(self, v):
        return self._marked[v]
    
    def _get_excess_flow_from_network(self, vertex):
        total = 0

        for edge in self._graph.get_adjacent_edges(vertex):
            if vertex == edge.end():
                total -= edge.flow()
            else:
                total += edge.flow()
        
        return total
    
    def max_flow(self):
        return self._max_flow


if __name__ == '__main__':
    doctest.testmod()