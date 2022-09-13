
from module_4_3 import EdgeWeightedGraph, Edge, KruskalMST


"""
Question 1
Bottleneck minimum spanning tree.
Given a connected edge-weighted graph,
design an efficient algorithm to find a minimum bottleneck spanning tree.
The bottleneck capacity of a spanning tree is the weights of its largest edge.
A minimum bottleneck spanning tree is a spanning tree of minimum bottleneck capacity.



Can use any algorithm that help building a MST or:
Camerini's algorithm:
1. Find the median edge weight W (find kth algorithm, use pivot and recursively find kth)
2. two subgraph by the median edge,
    if the lower part connected (using DFS or BFS), then decrease W and repeat 1, 2
    if the not connected, let the connected component become one node, increase W, repeat 1, 2
    
"""


"""
Question 2
Is an edge in a MST.
Given an edge-weighted graph G and an edge e, design a linear-time algorithm to determine whether e appears in some MST of G.
Note: Since your algorithm must take linear time in the worst case, you cannot afford to compute the MST itself.
"""

class EdgeInMST(object):
    """
    Find the minimum spanning tree cost of the entire graph, using the Kruskal algorithm.
    As the inclusion of the edge (A, B) in the MST is being checked, include this edge first in the minimum spanning tree and then include other edges subsequently.
    Finally check if the cost is the same for both the spanning trees including the edge(A, B) and the calculated weight of the MST.
    If cost is the same, then edge (A, B) is a part of some MST of the graph otherwise it is not.
    """
    def __init__(self, graph: EdgeWeightedGraph, candidate: Edge) -> None:
        pass
"""
Question 3
Minimum-weight feedback edge set.
A feedback edge set of a graph is a subset of edges that contains at least one edge from every cycle in the graph.
If the edges of a feedback edge set are removed,
the resulting graph is acyclic. Given an edge-weighted graph,
design an efficient algorithm to find a feedback edge set of minimum weight. Assume the edge weights are positive.

use kruskal's algorithm, but use MaxPQ
"""
