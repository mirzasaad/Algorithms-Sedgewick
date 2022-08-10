from enum import Enum
from rect import Rect
from point import Point2D

KDTREE_SEPERATOR = Enum('KDTREE_SEPERATOR', 'VERTICAL HORIZONTAL')

class Node():
    def __init__(self, point: Point2D, rect: Rect, sep: KDTREE_SEPERATOR):
        self.point: Point2D = point
        self.rect: Rect = rect
        self.separator: KDTREE_SEPERATOR = sep
        self.lb: Node | None = None
        self.rt: Node | None = None
        self.size: int = 1

    def next_seperator(self):
        return KDTREE_SEPERATOR.HORIZONTAL if self.separator == KDTREE_SEPERATOR.VERTICAL else KDTREE_SEPERATOR.VERTICAL

    def create_next_rect(self, p):
        return self.create_left_bottom_rect() if self.is_right_or_top_of(p) else self.create_right_top_rect()

    def create_left_bottom_rect(self):
        if self.separator == KDTREE_SEPERATOR.VERTICAL:
            return Rect(self.rect.xmin, self.rect.ymin, self.point.x, self.rect.ymax)

        return Rect(self.rect.xmin, self.rect.ymin, self.rect.xmax, self.point.y)

    def create_right_top_rect(self):
        if self.separator == KDTREE_SEPERATOR.VERTICAL:
            return Rect(self.point.x, self.rect.ymin, self.rect.xmax, self.rect.ymax)

        return Rect(self.rect.xmin, self.point.y, self.rect.xmax, self.rect.ymax)

    def is_right_or_top_of(self, other: Point2D):
        return (self.separator == KDTREE_SEPERATOR.HORIZONTAL and self.point.y > other.y) or \
            self.separator == KDTREE_SEPERATOR.VERTICAL and self.point.x > other.x

    def is_left_or_bottom_of(self, other: Point2D):
        return (self.separator == KDTREE_SEPERATOR.VERTICAL and self.point.x <= other.x) or \
            (self.separator == KDTREE_SEPERATOR.HORIZONTAL and self.point.y <= other.y)

    def __repr__(self):
        return "Node(%s, Separator =>> %s)" % (self.point, self.separator)