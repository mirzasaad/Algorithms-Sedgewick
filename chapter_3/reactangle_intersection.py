
import random
from typing import List
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle

import heapq
from avl import AVL
from interval_search_tree import IntervalSearchTree
from common import SegmentHV, Event, Rect, Interval
from itertools import cycle
cycol = cycle('bgrcmk')

class SweepLine(object):
    """
        Sweep Line Algo
        algorithm si good
        tree has some problem with range methind
        and segment is sorting by x coordinate rather it should be y
    """

    def __init__(self, _no_of_rectangles) -> None:
        self._pq = []
        self._ist = IntervalSearchTree()
        self._no_of_rectangles = _no_of_rectangles
        self._rectangles: List[Rect] = []
        self.__generate_rects()
        self.__draw()
        self.__run()

    def __draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title("Rectangles - Sweep Line Algorithm")

        for rect in self._rectangles:
            ax.add_patch(matplotlib.patches.Rectangle((rect.xmin, rect.ymin), rect.xmax -
                         rect.xmin, rect.ymin - rect.ymax, color='red', fc='none', lw=2))

        plt.xlim([-20, 200])
        plt.ylim([-100, 150])

        plt.xlabel("x - axis")
        plt.ylabel("y - axis")

        plt.show()

    def __generate_rects(self):
        for i in range(self._no_of_rectangles):
            x1 = random.randrange(0, 100)
            x2 = x1 + random.randrange(0, 100)
            y1 = random.randrange(0, 100)
            y2 = y1 + random.randrange(0, 100)
            self._rectangles.append(Rect(x1, y1, x2, y2))

    def __run(self):
        pq = []
        for rect in self._rectangles:
            start = Event(-rect.xmin, rect, 'start')
            end = Event(-rect.xmax, rect, 'end')
            heapq.heappush(pq, start)
            heapq.heappush(pq, end)

        while pq:
            event: Event = heapq.heappop(pq)
            rect: Rect = event.segment
            position = event.position

            interval = Interval(rect.ymin, rect.ymax)

            if position == 'start':
                intersections = [item.value for item in self._ist.search_intersection(interval)];
                if intersections:
                    print('intersection ---> ', [rect] + intersections)
                
                self._ist.put(interval, rect)
            else:
                self._ist.delete(interval)
sl = SweepLine(5)
