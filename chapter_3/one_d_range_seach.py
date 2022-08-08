

from avl import AVL
from module_3_3 import RBTree
from common import Interval

class OneDRangeSearch(RBTree):
    def range(self, _min, _max):
        result = []
        self.__range(self.root, Interval(_min, _max), result)
        return result

    def __range(self, node, interval: Interval, result):
        if not node:
            return
        if interval.min < node.key:
            self.__range(node.left, interval, result)
        
        if interval.contains_eq(node.key):
            result.append(node.key)
        
        if interval.max > node.key:
            self.__range(node.right, interval, result)
