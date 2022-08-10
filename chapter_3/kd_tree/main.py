

import random
import time
from rect import Rect
from kd_tree import KdTree
from point_set import PointSET
from point import Point2D


def test_brute(points, random_for_nearest):
    start = time.time()
    brute = PointSET()

    for p in points:
        brute.insert(p)

    n = points[random_for_nearest]
    nearest = brute.nearest(n)
    end = time.time()

    print('BRUTE FORCE')
    print('point =>>', n, 'nearest =>>', nearest)
    print('Time Elapsed =>>', end - start)
    print()


def test_fast(points, random_for_nearest, bounding_box):
    start = time.time()
    fast = KdTree(bounding_box.xmin, bounding_box.ymin,
                  bounding_box.xmax, bounding_box.ymax)
    print('')
    for p in points:
        fast.insert(p)

    n = points[random_for_nearest]
    nearest = fast.nearest(n)
    end = time.time()

    print('FAST KDTREE')
    print('point =>>', n, 'nearest =>>', nearest)
    print('Time Elapsed =>>', end - start)
    print()


#problem with ranndom numbers
if __name__ == '__main__':
    points = [Point2D(0.7, 0.2), Point2D(0.5, 0.4), Point2D(0.2, 0.3), Point2D(0.4, 0.7), Point2D(0.9, 0.6)]
    bounding_box = Rect(0, 0, 1, 1)

    random_for_nearest = 2

    test_brute(points, random_for_nearest)
    test_fast(points, random_for_nearest, bounding_box)