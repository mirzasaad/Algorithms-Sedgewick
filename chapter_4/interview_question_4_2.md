## Directed Graphs
1. Shortest directed cycle. Given a digraph G, design an efficient algorithm to find a directed cycle with the minimum number of edges (or report that the graph is acyclic). The running time of your algorithm should be at most proportional to V(E+V) and use space proportional to E+V, where V is the number of vertices and E is the number of edges.

   run BFS from each vertex
   
2. Hamiltonian path in a DAG. Given a directed acyclic graph, design a linear-time algorithm to determine whether it has a Hamiltonian path (a simple path that visits every vertex), and if so, find one.

   if more than one vertex has 0 in degree or more than one vertex has 0 out degree, then none, else topological sort from the vertex that has 0 in degree, when encountering branches, record the vertex we randomly shift to, if we reach the termination points and stil have unvisited vertices, we should return back to same vertex with multiple children again, when we recurse on another vertex and check if it will end on the vertex posterior to the one we record, if not, then no.  
   or https://stackoverflow.com/questions/16124844/algorithm-for-finding-a-hamilton-path-in-a-dag
   
3. Reachable vertex.  
   DAG: Design a linear-time algorithm to determine whether a DAG has a vertex that is reachable from every other vertex, and if so, find one.  
   Digraph: Design a linear-time algorithm to determine whether a digraph has a vertex that is reachable from every other vertex, and if so, find one.
   
   DAG:if it's reachable from every other vertex, then it must have 0 out degree, run dfs from this vertex on the reverse graph  
   Digraph:compute the strongly connected components(Kosaraju's algorithm), run dfs on random vertex of the last scc, and check if it can reach every other scc.