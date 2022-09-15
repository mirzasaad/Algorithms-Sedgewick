## Minimum Spanning Trees
1. Bottleneck minimum spanning tree. Given a connected edge-weighted graph, design an efficient algorithm to find a minimum bottleneck spanning tree. The bottleneck capacity of a spanning tree is the weights of its largest edge. A minimum bottleneck spanning tree is a spanning tree of minimum bottleneck capacity.

   mst is mbst(O(E) algorithm:Camerini's algorithm)
   
2. Is an edge in a MST. Given an edge-weighted graph G and an edge e, design a linear-time algorithm to determine whether e appears in some MST of G.  
   Note: Since your algorithm must take linear time in the worst case, you cannot afford to compute the MST itself.   

   consider the subgraph G' of G containing only those edges whose weight is strictly less than that of e,if it's connected, then yes, else no.
   
3. Minimum-weight feedback edge set. A feedback edge set of a graph is a subset of edges that contains at least one edge from every cycle in the graph. If the edges of a feedback edge set are removed, the resulting graph is acyclic. Given an edge-weighted graph, design an efficient algorithm to find a feedback edge set of minimum weight. Assume the edge weights are positive.

https://cstheory.stackexchange.com/questions/36222/minimum-weight-feedback-edge-set-in-undirected-graph-how-to-find-it-is-it-np
