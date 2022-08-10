import doctest
from typing import List

from matplotlib import patches, pyplot as plt
from node import Node, Point2D, KDTREE_SEPERATOR
from rect import Rect

class KdTree():
    """
        Kd Tree to diveide point on 2d plane
    the points starts from bottom, left both x, and y
    >>> kd = KdTree(0, 0, 1, 1)
    >>> points = [Point2D(0.7, 0.2), Point2D(0.5, 0.4), Point2D(0.2, 0.3), Point2D(0.4, 0.7), Point2D(0.9, 0.6)]
    >>> for p in points:
    ...     kd.insert(p)
    >>> print(kd.range(Rect(0, 0, 1000, 1000)))
    [Point(0.7, 0.2), Point(0.5, 0.4), Point(0.2, 0.3), Point(0.4, 0.7), Point(0.9, 0.6)]
    """
    def __init__(self, xmin=0, ymin=0, xmax=1000, ymax=1000):
        self._root = None
        self._size = 0
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def is_empty(self):
        return self._size == 0

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = root

    def __size(self, node):
        return node.size if node else 0

    def __update(self, node: Node):
        node.size = 1 + self.__size(node.lb) + self.__size(node.rt)

    def size(self):
        return self.__size(self.root)

    def draw(self, points):
        bounding_box = Rect(self.xmin, self.ymin, self.xmax, self.ymax)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        x_points, y_points = [p.x for p in points], [p.y for p in points]
        ax.scatter(x_points, y_points, s=10)
        #draw bounding box

        lt = (bounding_box.xmin, bounding_box.ymin)
        width = bounding_box.xmax - bounding_box.xmin
        height = bounding_box.ymax - bounding_box.ymin

        # Create a Rectangle patch
        width = bounding_box.xmax - bounding_box.xmin
        rect = patches.Rectangle(lt, width, height,linewidth=1,edgecolor='r',facecolor='none')

        ax.add_patch(rect)

        plt.xlabel("x - axis")
        plt.ylabel("y - axis")

        plt.show()

    def __create_root(self, point):
        base_rect = Rect(self.xmin, self.ymin, self.xmax, self.ymax)
        base_seperarator = KDTREE_SEPERATOR.VERTICAL
        return Node(point, base_rect, base_seperarator)

    def __insert(self, node: Node | None, prev_node: Node | None, point: Point2D):
        if not self._root:
            return self.__create_root(point)

        if not node:
            next_rect = prev_node.create_next_rect(point)
            next_seperator = prev_node.next_seperator()
            return Node(point, next_rect, next_seperator)
        
        if node.is_right_or_top_of(point):
            node.lb = self.__insert(node.lb, node, point)
        else:
            node.rt = self.__insert(node.rt, node, point)

        self.__update(node)

        return node


    def insert(self, point: Point2D):
        self.root = self.__insert(self.root, self.root, point)

    def __range(self, node: Node, rect: Rect, result: List[Point2D]):
        if not node:
            return

        if rect.contains(node.point):
            result.append(node.point)
            self.__range(node.lb, rect, result)
            self.__range(node.rt, rect, result)
            return

        if node.is_left_or_bottom_of(Point2D(rect.xmax, rect.ymax)):
            self.__range(node.rt, rect, result)
        elif node.is_right_or_top_of(Point2D(rect.xmin, rect.ymin)):
            self.__range(node.lb, rect, result)
        
    def range(self, rect):
        results = []
        self.__range(self._root, rect, results)
        return results

    def nearest(self, point):
        return self.__nearest(self.root, point, self.root.point) if self.root else None

    def __nearest(self, node, target_point, closest):
        if not node:
            return closest
        
        closest_distance = closest.distanceTo(target_point)

        if node.rect.distanceTo(target_point) < closest_distance:
            current_dist = node.point.distanceTo(target_point)

            if current_dist < closest_distance and node.point != target_point:
                closest = node.point

            if node.is_right_or_top_of(target_point):
                closest = self.__nearest(node.lb, target_point, closest)
                closest = self.__nearest(node.rt, target_point, closest)
            else:
                closest = self.__nearest(node.rt, target_point, closest)
                closest = self.__nearest(node.lb, target_point, closest)
        
        return closest

    def get(self, point):
        return self.__get(self.root, point)
    
    def __get(self, node: Node, point):
        if not node or node.point == point:
            return node

        if node.is_right_or_top_of(point):
            return self.__get(node.lb, point)
        else:
            return self.__get(node.rt, point)

    def contains(self, point):
        return True if self.get(point) else False

if __name__ == '__main__':
    doctest.testmod()