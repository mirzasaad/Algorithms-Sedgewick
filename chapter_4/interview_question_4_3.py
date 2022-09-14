
from module_4_3 import EdgeWeightedGraph, Edge, KruskalMST


"""
Question 1
Bottleneck minimum spanning tree.
Given a connected edge-weighted graph,
design an efficient algorithm to find a minimum bottleneck spanning tree.
The bottleneck capacity of a spanning tree is the weights of its largest edge.
A minimum bottleneck spanning tree is a spanning tree of minimum bottleneck capacity.

https://stackoverflow.com/questions/14297409/how-is-a-minimum-bottleneck-spanning-tree-different-from-a-minimum-spanning-tree

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


"""

"""

class EdgeInMST(object):
    """
    Find the minimum spanning tree cost of the entire graph, using the Kruskal algorithm.
    As the inclusion of the edge (A, B) in the MST is being checked, include this edge first in the minimum spanning tree and then include other edges subsequently.
    Finally check if the cost is the same for both the spanning trees including the edge(A, B) and the calculated weight of the MST.
    If cost is the same, then edge (A, B) is a part of some MST of the graph otherwise it is not.

    Solution

    https://stackoverflow.com/questions/15049864/check-if-edge-is-included-in-some-mst-in-linear-time-non-distinct-values
    
    We will solve this using MST cycle property, which says that, "For any cycle C in the graph, 
    if the weight of an edge e of C is larger than the weights of all other edges of C, then this edge cannot belong to an MST."
    Now, run the following O(E+V) algorithm to test if the edge E connecting vertices u and v will be a part of some MST or not.

    Step 1

        Run dfs from one of the end-points(either u or v) of the edge E considering only those edges that have weight less than that of E.

    Step 2

        Case 1 If at the end of this dfs, the vertices u and v get connected, then edge E cannot be a part of some MST. 
        This is because in this case there definitely exists a cycle in the graph with the edge E 
        having the maximum weight and it cannot be a part of the MST(from the cycle property).

        Case 2 But if at the end of the dfs u and v stay disconnected, then edge E must be the 
        part of some MST as in this case E is always not the maximum weight edge in all the cycles that it is a part of.
    Share

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
