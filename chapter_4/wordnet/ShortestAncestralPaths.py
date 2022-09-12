
import doctest
from BreadthFirstDirectedPaths import BreadthFirstDirectedPaths
from Digraph import Digraph


class ShortestAncestralPaths():
    """
    Find the Common Ancestor between two ndoes with the shortes distance,
    use the bfs tp find the shortest distance

    >>> paths = [(1, 0), (2, 0) ,(3, 1) ,(4, 1) ,(5, 2) ,(6, 2) ,(10, 5) ,(11, 5) ,(12, 5) ,(17, 10) ,(18, 10) ,(19, 12) ,(20, 12) ,(23, 20) ,(24, 20),(7, 3) ,(8, 3) ,(9, 3) ,(13, 7) ,(14, 7) ,(15, 9) ,(16, 9) ,(21, 16) ,(22, 16)]
    >>> paths_2 = [(7,  3), (8,  3), (3,  1), (4,  1), (5,  1), (9, 5), (10, 5), (11, 10), (12, 10), (1, 0), (2, 0)]
    
    >>> G = Digraph()
    >>> for a, b in paths:
    ...     G.add_edge(a, b)
    >>> sap = ShortestAncestralPaths(G)
    >>> assert sap.ancestor([13], [16]) == 3
    >>> assert sap.ancestor([23], [24]) == 20
    >>> assert sap.ancestor([17], [6]) == 2
    >>> assert sap.ancestor([13, 23, 24], [6, 16, 17]) == 3

    >>> G = Digraph()
    >>> for a, b in paths_2:
    ...     G.add_edge(a, b)
    >>> sap = ShortestAncestralPaths(G)
    >>> assert sap.ancestor([9], [12]) == 5
    >>> assert sap.length([9], [12]) == 3
    >>> assert sap.ancestor([7], [2]) == 0
    >>> assert sap.length([7], [2]) == 4
    """
    def __init__(self, G: Digraph):
        self._graph = G
        self._ancestor = -1
        self._min_distance = float('inf')

    # length of shortest ancestral path between v and w; -1 if no such path
    def length(self, v, w):
        self.__bfs(v, w)
        
        return self._min_distance

    # a common ancestor of v and w that participates in a shortest ancestral path; -1 if no such path
    def ancestor(self, v, w):
        self.__bfs(v, w)

        return self._ancestor

    def __bfs(self, v, w):
        # if v and w:
        #   return 0

        d1 = BreadthFirstDirectedPaths(self._graph, v)
        d2 = BreadthFirstDirectedPaths(self._graph, w)

        min_distance = float('inf')
        ancestor = -1

        for vertex in self._graph.vertices():
            if d1.has_path_to(vertex) and d2.has_path_to(vertex):
                d = d1.distanceTo(vertex) + d2.distanceTo(vertex)
                if d < min_distance:
                    min_distance = d
                    ancestor = vertex

        self._ancestor = ancestor
        self._min_distance = min_distance

if __name__ == '__main__':
    doctest.testmod()