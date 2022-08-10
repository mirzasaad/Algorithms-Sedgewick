import math
from point import Point2D
from rect import Rect
from bintrees import RBTree

class PointSET():
    def __init__(self):
        self.points = RBTree()

    def is_empty(self):
        return self.points.is_empty()

    def size(self):
        return self.points.__len__()

    def insert(self, p):
        if not self.points.get(p):
            self.points.insert(p, '')

    def contains(self, p):
        return self.points.get(p) is not None

    def range(self, rect):
        minPoint = Point2D(rect.xmin, rect.ymin)
        maxPoint = Point2D(rect.xmax, rect.ymax + 1)
        return [p for p in self.points[minPoint:maxPoint] if p.x >= rect.xmin and p.x <= rect.xmax]

    def __get_ceiling(self, p):
        ceiling = None
        try:
            ceiling = self.points.succ_key(p)
        except:
            pass
        return ceiling

    def __get_floor(self, p):
        floor = None
        try:
            floor = self.points.prev_key(p)
        except:
            pass
        return floor

    def nearest(self, p):
        if self.is_empty():
            return None

        # 1: Find the 2 neighbour points in natural order, then find the closest distance `d`
        # 2: Widen the navigatable set to a circle of `d`, save the nearest.

        next = self.__get_ceiling(p)
        prev = self.__get_floor(p)

        if not prev and not next:
            return None

        distNext = p.distanceTo(next) if next else float('inf')
        distPrev = p.distanceTo(prev) if prev else float('-inf')
        d = min(distPrev, distNext)

        minPoint = Point2D(p.x, p.y - d)
        maxPoint = Point2D(p.x, p.y + d + 1)

        nearest = next if next else prev
        range = self.points[minPoint:maxPoint]

        for candidate in range:
            if p.distanceTo(candidate) < p.distanceTo(nearest) and candidate != p:
                nearest = candidate

        return nearest
