from enum import IntEnum
import math

class Node(object):
    def __init__(self, val):
        self._val = val
        self._next_node: Node | None = None

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

    @property
    def next_node(self):
        return self._next_node

    @next_node.setter
    def next_node(self, node):
        self._next_node = node

    def __cmp__(self, other):
        if self._val < other._val:
            return -1
        if self._val > other._val:
            return +1
        if self._val < other._val:
            return -1
        if self._val > other._val:
            return +1
        return 0

    def __eq__(self, other):
        return self.__cmp__(other) == 0

    def __ne__(self, other):
        return self.__cmp__(other) != 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __ge__(self, other):
        return self.__cmp__(other) >= 0

    def __le__(self, other):
        return self.__cmp__(other) <= 0

    def __repr__(self) -> str:
        return 'Node({})'.format(self._val)

class Point2d():
    def __init__(self, x, y):
        if (x == 0.0):
            self.x = 0.0  # convert -0.0 to +0.0
        else:
            self.x = x
        if (y == 0.0):
            self.y = 0.0  # convert -0.0 to +0.0
        else:
            self.y = y

    # Returns the polar radius of this point.
    def r(self):
        return math.sqrt(self.x * self.x + self.y * self.y)
    # Returns the angle of this point in polar coordinates.

    def theta(self):
        return math.atan2(self.y, self.x)
    # Returns the angle between this point and that point.

    def angleTo(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.atan2(dy, dx)
    #  Returns true if a→b→c is a counterclockwise turn.

    def ccw(self, a, b, c):
        area2 = (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x)
        if area2 < 0:
            return -1
        elif area2 > 0:
            return +1
        else:
            return 0
    # Returns twice the signed area of the triangle a-b-c.

    def area2(self, a, b, c):
        return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x)

    # Returns the Euclidean distance between this point and that point.
    def distanceTo(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx*dx + dy*dy)

    # Returns the square of the Euclidean distance between this point and that point.
    def distanceSquaredTo(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return dx*dx + dy*dy

    # Compares two points by y-coordinate, breaking ties by x-coordinate.
    # Formally, the invoking point (x0, y0) is less than the argument point (x1, y1) if and only if either {@code y0 < y1} or if {@code y0 == y1} and {@code x0 < x1}.

    def __cmp__(self, other):
        if self.y < other.y:
            return -1
        if self.y > other.y:
            return +1
        if self.x < other.x:
            return -1
        if self.x > other.x:
            return +1
        return 0

    def __eq__(self, other):
        return self.__cmp__(other) == 0

    def __ne__(self, other):
        return self.__cmp__(other) != 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __ge__(self, other):
        return self.__cmp__(other) >= 0

    def __le__(self, other):
        return self.__cmp__(other) <= 0

    def __repr__(self):
        return 'Point(%s, %s)' % (self.x, self.y)

class Color(IntEnum):
    Red = 0
    White = 1
    Blue = 2