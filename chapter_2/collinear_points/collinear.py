from collections import namedtuple, deque
from functools import cmp_to_key
import copy

LineSegment = namedtuple('LineSegment', 'p1 p2')


class ColinearPoints(object):
    """Collinear Points"""

    def __check_duplicates(self, points):
        for index in range(len(points) - 1):
            point = points[index]
            next_points = points[index + 1]

            if point.compareTo(next_points) == 0:
                raise ValueError("points on same coordinate!")


    def collinear_points_brute(self, _points):
        points, line_segments = copy.deepcopy(_points), []
        if len(points) < 4:
            return

        points.sort(key=cmp_to_key(lambda p1, p2: p1.compareTo(p2)))
        self.__check_duplicates(points)

        length = len(points)
        for i in range(length - 3):
            for j in range(i + 1, length - 2):
                for k in range(j + 1, length - 1):
                    for l in range(k + 1, length):
                        slope_a = points[i].slopeTo(points[j])
                        slope_b = points[j].slopeTo(points[k])
                        slope_c = points[k].slopeTo(points[l])

                        if (slope_a == slope_b == slope_c):
                            line_segments.append(LineSegment(points[i], points[l]))

        return line_segments


    def collinear_points_fast(self, _points):
        points, line_segments, N = copy.deepcopy(_points), [], len(_points)

        points.sort(key=cmp_to_key(lambda p1, p2: p1.compareTo(p2)))
        self.__check_duplicates(points)

        for p in points:
            sorted_by_slope = copy.deepcopy(points)
            sorted_by_slope.sort(key=cmp_to_key(
                lambda p1, p2: p.slopeOrder(p1, p2)))

            x = 1
            while x < N:
                candidates = deque([sorted_by_slope[x]])
                SLOPE_REF = p.slopeTo(sorted_by_slope[x])

                x += 1
                while (x < N and p.slopeTo(sorted_by_slope[x]) == SLOPE_REF):
                    candidates.append(sorted_by_slope[x])
                    x += 1

                if len(candidates) >= 3 and p.compareTo(candidates[0]) < 0:
                    line_segments.append(LineSegment(p, candidates[-1]))

        return line_segments
