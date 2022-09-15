
## Shortest Paths
1. Monotonic shortest path. Given an edge-weighted digraph G, design an ElogE algorithm to find a monotonic shortest path from s to every other vertex. A path is monotonic if the sequence of edge weights along the path are either strictly increasing or strictly decreasing.

   https://stackoverflow.com/questions/22876105/find-a-monotonic-shortest-path-in-a-graph-in-oe-logv

2. Second shortest path. Given an edge-weighted digraph and let P be a shortest path from vertex s to vertex t. Design an ElogV algorithm to find a path other than P from s to t that is as short as possible. Assume all of the edge weights are strictly positive.

   compute the shortest path distances from s to every vertex and the shortest path distances from every vertex to t.
   
3. Shortest path with one skippable edge. Given an edge-weighted digraph, design an ElogV algorithm to find a shortest path from s to t where you can change the weight of any one edge to zero. Assume the edge weights are nonnegative.   
   
   use a modified version of Dijkstra's algorithm, maintain an array of max weight of path from s to every vertex, each time we select a new edge and its end node, we choose the one with the shortest path given that we could delete the max weight from path.
   
 