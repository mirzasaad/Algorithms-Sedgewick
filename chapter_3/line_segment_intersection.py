from nis import cat
import random
from turtle import position
from typing import List
from bintrees import RBTree
from matplotlib import pyplot as plt
import avl
import heapq
from avl import AVL
from common import SegmentHV


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


class SweepLine(object):
    """
        Sweep Line Algo
        algorithm si good
        tree has some problem with range methind
        and segment is sorting by x coordinate rather it should be y
    """
    def __init__(self, lines_size) -> None:
        self._pq = []
        self._bst = RBTree()
        self._lines_size = lines_size
        self._line_segments: List[SegmentHV] = [
            SegmentHV(-10, 0, 10, 0),
            SegmentHV(0, -10, 0, 10),
        ]
        self.__generate_line()
        self.__draw()
        self.__run()

    def __draw(self):
        for line_segment in self._line_segments:
            plt.plot([line_segment.x1, line_segment.x2], [
                     line_segment.y1, line_segment.y2], linewidth=2)

        # Set chart title.
        plt.title('Sweep Line Algorithm')

        # Set x, y label text.
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.show()

    def __generate_line(self):
        for i in range(self._lines_size // 2):
            x1 = random.randrange(0, 100)
            x2 = x1 + random.randrange(0, 100)
            y1 = random.randrange(0, 100)
            y2 = y1
            self._line_segments.append(SegmentHV(x1, y1, x2, y2))

        for i in range(self._lines_size // 2):
            x1 = random.randrange(0, 100)
            x2 = x1
            y1 = random.randrange(0, 100)
            y2 = y1 + + random.randrange(0, 100)
            self._line_segments.append(SegmentHV(x1, y1, x2, y2))

    def __run(self):
        pq = []
        for segment in self._line_segments:
            if segment.is_vertical():
                e = Event(-segment.x1, segment, 'start')
                heapq.heappush(pq, e)
            elif segment.is_horizontal():
                e1 = Event(-segment.x1, segment, 'start')
                e2 = Event(-segment.x2, segment, 'start')
                heapq.heappush(pq, e1)
                heapq.heappush(pq, e2)
        
        while pq:
            event: Event = heapq.heappop(pq)
            sweep = event.time
            segment = event.segment
            position = event.position

            if segment.is_vertical():
                # s1 = SegmentHV(float('-inf'), segment.y1, float('inf'), segment.y1)
                # s2 = SegmentHV(float('-inf'), segment.y2, float('inf'), segment.y2)
                intersections = [i for i in self._bst[segment.y1: segment.y2].values()]
                if intersections:
                    print('intersection =>>>', intersections, [segment])

            elif sweep == -segment.x1 and position == 'start':
                self._bst.insert(segment.y1, segment)
            elif position == 'end':
                self._bst.remove(segment.y1)

sl = SweepLine(10)
