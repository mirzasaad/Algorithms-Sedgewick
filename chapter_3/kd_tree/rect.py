import math

class Rect():
    def __init__(self, xmin, ymin, xmax, ymax):
        if xmax < xmin or ymax < ymin:
            raise KeyError
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    # Returns true if the two rectangles intersect. This includes
    # improper intersections (at points on the boundary
    # of each rectangle) and nested intersctions
    # (when one rectangle is contained inside the other)
    def intersects(self, other):
        return self.xmax >= other.xmin and self.ymax >= other.ymin \
            and other.xmax >= self.xmin and other.ymax >= self.ymin

    # Returns true if this rectangle contain the point.
    def contains(self, p):
        return (p.x >= self.xmin) and (p.x <= self.xmax) and \
            (p.y >= self.ymin) and (p.y <= self.ymax)

    # Returns the Euclidean distance between this rectangle and the point
    def distanceTo(self, p):
        return math.sqrt(self.distanceSquaredTo(p))

    # Returns the square of the Euclidean distance between this rectangle and the point
    def distanceSquaredTo(self, p):
        dx, dy = 0.0, 0.0
        if p.x < self.xmin:
            dx = p.x - self.xmin
        elif p.x > self.xmax:
            dx = p.x - self.xmax
        elif p.y < self.ymin:
            dy = p.y - self.ymin
        elif p.y > self.ymax:
            dy = p.y - self.ymax
        return dx*dx + dy*dy

    def __repr__(self):
        return 'Rect(%s, %s, %s, %s)' % (self.xmin, self.ymin, self.xmax, self.ymax)
