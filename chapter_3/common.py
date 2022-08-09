from copy import deepcopy
import enum
from math import sqrt
from queue import Queue


class Color(enum.IntEnum):
    RED = 0
    BLACK = 1


class Node(object):

    def __init__(self, key, val, size, color=Color.BLACK):
        self._left = self._right = None
        self._key = key
        self._val = val
        self._size = size
        self._color = color
        self._max = 0
        self._height = 0

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, max):
        self._max = max

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, node):
        assert isinstance(node, (Node, type(None)))
        self._left = node

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, node):
        assert isinstance(node, (Node, type(None)))
        self._right = node

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, val):
        assert isinstance(val, int) and val >= 0
        self._size = val

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, val):
        self._key = val

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

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
        return '{}'.format(self._val)


class Interval():
    def __init__(self, min, max):
        assert min <= max
        self.min = min
        self.max = max

    def interesects(self, other):
        if other.max >= self.min and other.min <= self.max:
            return True
        return False

    def contains(self, x):
        return self.min < x < self.max

    def contains_eq(self, x):
        return x <= self.max and x >= self.min

    def __cmp__(self, other):

        if self.min < other.min:
            return -1
        elif self.min > other.min:
            return 1
        elif self.max < other.max:
            return -1
        elif self.max > other.max:
            return 1
        else:
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
        return 'Interval(%s, %s)' % (self.min, self.max)


class SegmentHV():
    def __init__(self, x1, y1, x2, y2):
        assert x1 <= x2 and y1 <= y2
        # assert not (x1 == x2 and y1 == y2)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def is_horizontal(self):
        return self.y1 == self.y2

    def is_vertical(self):
        return self.x1 == self.x2

    def __cmp__(self, other):
        if self.y1 < other.y1:
            return -1
        elif self.y1 > other.y2:
            return 1
        elif self.y2 < other.y2:
            return -1
        elif self.y2 > other.y2:
            return 1
        elif self.x1 < other.x1:
            return -1
        elif self.x1 > other.x1:
            return 1
        elif self.x2 < other.x2:
            return -1
        elif self.x2 > other.x2:
            return 1
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
        # h, v = self.is_horizontal(), self.is_vertical()
        # if h or v:
        #     return 'Horizontal Line' if h else 'Vertical Line'
        return "[Point(%s, %s) ,Point(%s, %s)]" % (self.x1, self.y1, self.x2, self.y2)


class AVLNode():
    def __init__(self, key, value, size=1, height=0):
        self._left = self._right = None
        self._key = key
        self._value = value
        self._size = size
        self._height = height
        self._max = 0

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, max):
        self._max = max

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, node):
        assert isinstance(node, (AVLNode, type(None)))
        self._left = node

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, node):
        assert isinstance(node, (AVLNode, type(None)))
        self._right = node

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        assert isinstance(value, int) and value >= 0
        self._size = value

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        self._key = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def __cmp__(self, other):
        return self.key - other.key

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
        return '(key => %s, height => %s, size => %s, max => %s)' % (self.key, self.height, self.size, self.max)


class Rect(object):
    def __init__(self, xmin, ymin, xmax, ymax) -> None:
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

    @property
    def xmin(self):
        return self._xmin

    @xmin.setter
    def xmin(self, xmin):
        self._xmin = xmin

    @property
    def ymin(self):
        return self._ymin

    @ymin.setter
    def ymin(self, ymin):
        self._ymin = ymin

    @property
    def xmax(self):
        return self._xmax

    @xmax.setter
    def xmax(self, xmax):
        self._xmax = xmax

    @property
    def ymax(self):
        return self._ymax

    @ymax.setter
    def ymax(self, ymax):
        self._ymax = ymax

    def intersects(self, other):
        return self.xmax >= other.xmin and self.ymax >= other.ymin and other.xmax >= self.xmin and other.ymax >= self.ymin

    def contains(self, p):
        return (p.x() >= self.xmin) and (p.x() <= self.xmax) and (p.y() >= self.ymin) and (p.y() <= self.ymax)

    def distanceTo(self, p):
        return sqrt(self.distanceSquaredTo(p))

    def distanceSquaredTo(self, p):
        dx = 0.0
        dy = 0.0
        if (p.x() < self.xmin):
            dx = p.x() - self.xmin
        elif (p.x() > self.xmax):
            dx = p.x() - self.xmax
        if (p.y() < self.ymin):
            dy = p.y() - self.ymin
        elif (p.y() > self.ymax):
            dy = p.y() - self.ymax
        return dx*dx + dy*dy

    def __cmp__(self, other):
        if (self.xmin != other.xmin):
            return 0
        if (self.ymin != other.ymin):
            return 0
        if (self.xmax != other.xmax):
            return 0
        if (self.ymax != other.ymax):
            return 0
        return 1

    def __cmp__(self, other):
        return self.key - other.key

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
        return 'rect[{}, {}, {}, {}]'.format(self.xmin, self.ymin, self.xmax, self.ymax)


class Point2D():
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __cmp__(self, other):
        if (self.y < other.y):
            return -1
        if (self.y > other.y):
            return +1
        if (self.x < other.x):
            return -1
        if (self.x > other.x):
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
        return 'Interval(%s, %s)' % (self.min, self.max)


class Event():
    """
        Represent A Line segment Event occured durinf line sweep algorithm
    """

    def __init__(self, time, segment, position):
        self.segment: SegmentHV = segment
        self.time = time
        self.position = position

    def __cmp__(self, other):
        if self.time < other.time:
            return -1
        elif self.time > other.time:
            return 1
        else:
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


def display_bst(root):
    lines, *_ = _display_aux(root)
    for line in lines:
        print(line)


def _display_aux(self):
    """Returns list of strings, width, height, and horizontal coordinate of the root."""
    # No child.
    if self.right is None and self.left is None:
        line = '%s' % self.val
        width = len(line)
        height = 1
        middle = width // 2
        return [line], width, height, middle

    # Only left child.
    if self.right is None:
        lines, n, p, x = _display_aux(self.left)
        s = '%s' % self.val
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
        second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
        shifted_lines = [line + u * ' ' for line in lines]
        return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

    # Only right child.
    if self.left is None:
        lines, n, p, x = _display_aux(self.right)
        s = '%s' % self.val
        u = len(s)
        first_line = s + x * '_' + (n - x) * ' '
        second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
        shifted_lines = [u * ' ' + line for line in lines]
        return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

    # Two children.
    left, n, p, x = _display_aux(self.left)
    right, m, q, y = _display_aux(self.right)
    s = '%s' % self.val
    u = len(s)
    first_line = (x + 1) * ' ' + (n - x - 1) * \
        '_' + s + y * '_' + (m - y) * ' '
    second_line = x * ' ' + '/' + \
        (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
    if p < q:
        left += [n * ' '] * (q - p)
    elif q < p:
        right += [m * ' '] * (p - q)
    zipped_lines = zip(left, right)
    lines = [first_line, second_line] + \
        [a + u * ' ' + b for a, b in zipped_lines]
    return lines, n + m + u, max(p, q) + 2, n + u // 2
