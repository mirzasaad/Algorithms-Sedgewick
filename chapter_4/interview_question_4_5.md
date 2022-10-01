## Maximum Flow
1. Fattest path. Given an edge-weighted digraph and two vertices ss and tt, design an ElogE algorithm to find a fattest path from s to t. The bottleneck capacity of a path is the minimum weight of an edge on the path. A fattest path is a path such that no other path has a higher bottleneck capacity.

   sort all edges in descending order, begin with a graph with only s and t, add one edge at a time to the graph successively, check whether s and t are connected, repeat until connected. As for how to check connectivity:using quick union
   
2. Perfect matchings in k-regular bipartite graphs. Suppose that there are n men and n women at a dance and that each man knows exactly k women and each woman knows exactly k men (and relationships are mutual). Show that it is always possible to arrange a dance so that each man and woman are matched with someone they know.

   build a graph with source s connected to n men, which is connected the n women they know respectively, all the women are connected to t,  and compute the maxflow as shown in the job seeker example in lecture. For rigorous proof, refer to:https://math.stackexchange.com/questions/1805181/prove-that-a-k-regular-bipartite-graph-has-a-perfect-matching/1805195
   
3. Maximum weight closure problem. A subset of vertices S in a digraph is closed if there are no edges pointing from S to a vertex outside S. Given a digraph with weights (positive or negative) on the vertices, find a closed subset of vertices of maximum total weight.   

   https://en.wikipedia.org/wiki/Closure_problem  
   elaboration:  
   As Picard (1976) showed, a maximum-weight closure may be obtained from G by solving a maximum flow problem on a graph H constructed from G by adding to it two additional vertices s and t. For each vertex v with positive weight in G, the augmented graph H contains an edge from s to v with capacity equal to the weight of v, and for each vertex v with negative weight in G, the augmented graph H contains an edge from v to t whose capacity is the negation of the weight of v. All of the edges in G are given infinite capacity in H.  
   A minimum cut separating s from t in this graph cannot have any edges of G passing in the forward direction across the cut: a cut with such an edge would have infinite capacity and would not be minimum. Therefore, the set of vertices on the same side of the cut as s automatically forms a closure C. [**because it does not have an edge point to other vertices outside the set, whose capacity is infinity**] The capacity of the cut equals the weight of all positive-weight vertices minus the weight of the vertices in C [**why? as we know, the capacity of a cut(A,B) equals the sum of the flows on its edges from A to B minus the sum of the flows on its edges from from B to A, in this situtation, the edges that cross the cut can only be the subset of edges pointed from s or pointed to t, since other edges have weight of infinity, for every positive-weight vertex v, it is connected to s with some edge e by definition, if v is not at the same side of s,e crosses the cut, by maxflow property "A = set of vertices connected to s by an undirected path with no full forward or empty backward edges", e is full forward edge, it should contribute to the capacity of the cut, else v is at the same side of s, the edge (s,v) will not cross the cut and therefore does not contribute to the capacity, for every negative-weight vertex v, it is connected to t, similarly as above, if it is at the same side of s, the edge e from v to t is full forward and will contribute to the capacity of cut, else is unconcerned, therefore the mincut would be sum of the weights of positive-weight vertices at the side of t and the negative weights of negative-weight vertices at the side of s**] , which is minimized when the weight of C is maximized. By the max-flow min-cut theorem, a minimum cut, and the optimal closure derived from it, can be found by solving a maximum flow problem.
   