import doctest


class Point():
    """
        Point Class
    >>> p = Point(4, 4)
    >>> p1 = Point(5, 5)
    >>> assert p.compareTo(p1) == -1
    >>> assert p1.compareTo(p1) == 0
    >>> assert p1.compareTo(p) == 1
    >>> assert p.slopeTo(p2) == float('inf')
    >>> assert p.slopeTo(p3) == 0.0
    >>> assert p.slopeTo(p) == float('-inf')
    >>> assert p.slopeTo(p1) == 1.0
    >>> assert p.slopeOrder(p2, p3) == 1
    >>> assert p1.slopeOrder(p, p3) == 1
    >>> assert p1.slopeOrder(p3, p) == -1
    >>> assert p1.slopeOrder(p3, p3) == 0
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    """Compares two points by y-coordinate, breaking ties by x-coordinate.
    Formally, the invoking point (x0, y0) is less than the argument point
    (x1, y1) if and only if either y0 < y1 or if y0 = y1 and x0 < x1.
    
    @param  that the other point
    @return the value <tt>0</tt> if this point is equal to the argument
    point (x0 = x1 and y0 = y1);
    a negative integer if this point is less than the argument
    point; and a positive integer if this point is greater than the
    argument point
    """

    def compareTo(self, point):
        if self.x == point.x and self.y == point.y:
            return 0

        if self.y < point.y or (self.y == point.y and self.x < point.x):
            return -1

        return 1

    """Returns the slope between this point and the specified point.
    Formally, if the two points are (x0, y0) and (x1, y1), then the slope
    is (y1 - y0) / (x1 - x0). For completeness, the slope is defined to be
    +0.0 if the line segment connecting the two points is horizontal;
    Double.POSITIVE_INFINITY if the line segment is vertical;
    and Double.NEGATIVE_INFINITY if (x0, y0) and (x1, y1) are equal.
    @param  that the other point
    @return the slope between this point and the specified point
    """

    def slopeTo(self, point):
        if self.x == point.x and self.y == point.y:
            return float('-inf')
        elif self.x == point.x:
            return float('inf')
        elif self.y == point.y:
            return +0.0

        return (point.y - self.y) / (point.x - self.x)

    """Compares two points by the slope they make with this point.
    The slope is defined as in the slopeTo() method.
    @return the Comparator that defines this ordering on points
    """

    def slopeOrder(self, a, b):
        slopeA = self.slopeTo(a)
        slopeB = self.slopeTo(b)

        if slopeA < slopeB:
            return -1
        elif slopeA > slopeB:
            return 1
        else:
            return 0

    def __repr__(self):
        return 'Point(%d, %d)' % (self.x, self.y)

if __name__ == '__main__':
    doctest.testmod()