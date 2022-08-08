import doctest
from typing import List
from avl import AVL
from common import Interval, AVLNode

class IntervalSearchTree(AVL):
    """
        Interval Search Tree
        given a interval search within interval bst
        for search maintain max for every subtree, during traversal check for intersection
        >>> ist = IntervalSearchTree()
        >>> intervals = [Interval(17, 19), Interval(5, 8), Interval(21, 24), Interval(4, 8), Interval(15, 18), Interval(7, 10), Interval(16, 22)]
        >>> for interval in intervals:
        ...     ist.put(interval, interval)
        (key => Interval(17, 19), height => 0, size => 1, max => 19)
        (key => Interval(5, 8), height => 0, size => 1, max => 8)
        (key => Interval(21, 24), height => 0, size => 1, max => 24)
        (key => Interval(4, 8), height => 0, size => 1, max => 8)
        (key => Interval(15, 18), height => 0, size => 1, max => 18)
        (key => Interval(7, 10), height => 0, size => 1, max => 10)
        (key => Interval(16, 22), height => 0, size => 1, max => 22)
        >>> print(ist.search_intersection(Interval(6, 7)))
        [(key => Interval(4, 8), height => 0, size => 1, max => 8), (key => Interval(5, 8), height => 1, size => 3, max => 18), (key => Interval(7, 10), height => 0, size => 1, max => 10)]
        >>> print(ist.keys())
        [Interval(4, 8), Interval(5, 8), Interval(7, 10), Interval(15, 18), Interval(16, 22), Interval(17, 19), Interval(21, 24)]
    """
    def __init__(self) -> None:
        super().__init__()

    def  __search_intersection(self, node: AVLNode, interval: Interval, result: List[Interval]):
        if not node or node.max < interval.min:
            return

        if node.left and node.max >= interval.min:
            self.__search_intersection(node.left, interval, result)
        
        if interval.interesects(node.key):
            result.append(node)

        if node.right and node.max >= interval.max:
            self.__search_intersection(node.right, interval, result)

    def search_intersection(self, interval: Interval) -> List[Interval]:
        result: List[Interval] = []
        self.__search_intersection(self.root, interval, result)
        return result

if __name__ == '__main__':
    doctest.testmod()



