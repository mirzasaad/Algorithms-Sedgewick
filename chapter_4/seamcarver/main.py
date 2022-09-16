from state import State
from PIL import Image
from math import sqrt
import heapq

class SeamCarver():
    def __init__(self, picture):
        picture = self._picture = Image.open(picture)
        rgb = self._rgb = picture.load()

    def rgb(self):
        return self._rgb

    def width(self):
        return self._picture.size[0]

    def height(self):
        return self._picture.size[1]

    def dx(self, x, y):
        c0 = self.rgb()[x - 1, y]
        c1 = self.rgb()[x + 1, y]
        return self.getGrad(c0, c1)

    def dy(self, x, y):
        c0 = self.rgb()[x, y - 1]
        c1 = self.rgb()[x, y + 1]
        return self.getGrad(c0, c1)

    def getGrad(self, c0, c1):
        r = c0[0] - c1[0]
        g = c0[1] - c1[1]
        b = c0[2] - c1[2]
        return r*r + g*g + b*b

    def get_energy(self, x, y, transposed):
        if transposed:
            return self.get_energy(y, x, False)
        if x < 0 or x >= self.width() or y < 0 or y >= self.height():
            raise Exception(
                "Expected x in [0, width - 1] and y in [0, height - 1]")
        if x == 0 or x == self.width() - 1 or y == 0 or y == self.height() - 1:
            return 1000
        # return self.dx(x, y) + self.dy(x, y)
        return sqrt(self.dx(x, y) + self.dy(x, y))

    def energy(self, x, y):
        return self.get_energy(x, y, False)

    def findHorizontalSeam(self):
        energy = [[None for i in range(self.height())]
                  for i in range(self.width())]

        for x in range(self.width()):
            for y in range(self.height()):
                energy[x][y] = self.get_energy(x, y, False)

        return self.getPath(energy)

    def findVerticalSeam(self):
        energy = [[None for i in range(self.width())]
                  for i in range(self.height())]

        for x in range(self.height()):
            for y in range(self.width()):
                energy[x][y] = self.get_energy(x, y, True)

        return self.getPath(energy)

    def getPath(self, energy):
        n = len(energy)
        m = len(energy[0])

        dist = [[float('inf') for i in range(m)] for i in range(n)]
        prev = [[None for i in range(m)] for i in range(n)]
        pq = []
        edgeTo = {}

        for i in range(1, m-1):
            dist[0][i] = energy[0][i]
            prev[0][i] = i
            heapq.heappush(pq, State(0, i, dist[0][i]))
        while pq:
            current = heapq.heappop(pq)

            if current.row == n - 1:
                result = [None] * n
                c = current.column
                for i in reversed(range(n)):
                    result[i] = c
                    c = prev[i][c]
                # print(tabulate(dist, list(range(self.width())), tablefmt="grid"))
                return result

            for column_range in range(-1, 2, 1):

                column_candidate = current.column + column_range
                if (column_candidate <= 0 or column_candidate >= m-1):
                    continue

                next_row = current.row + 1
                ncost = current.cost + energy[next_row][column_candidate]
                if (dist[next_row][column_candidate] <= ncost):
                    continue

                dist[next_row][column_candidate] = ncost
                prev[next_row][column_candidate] = current.column
                edgeTo[column_candidate] = current.column

                heapq.heappush(pq, State(next_row, column_candidate, ncost))
        return None

    def removeHorizontalSeam(self, seam):
        if not seam:
            raise Exception("Expected non-null seam")
        if len(seam) != self.width():
            raise Exception("Expected seam with length " + self.width())

        for i in range(self.width()):
            if abs(seam[i] - seam[i - 1]) > 1:
                raise Exception(
                    "Expected adjacent elements of seam with have a absolute difference of at most 1")

        if self.height() <= 1:
            raise Exception("Cannot remove horizontal seam on height <= 1")

        picture = Image.new(
            'RGB', (self.height() - 1, self.width()), color=None)
        for i in range(self.width()):
            k = 0
            for j in range(self.height()):
                if j != seam[i]:
                    picture.putpixel((k, i), self.rgb()[i, j])
                    k += 1
        return picture

    def removeVerticalSeam(self, seam):
        if not seam:
            raise Exception("Expected non-null seam")
        if len(seam) != self.height():
            raise Exception("Expected seam with length " + str(self.height()))

        for i in range(self.height()):
            if abs(seam[i] - seam[i - 1]) > 1:
                raise Exception(
                    "Expected adjacent elements of seam with have a absolute difference of at most 1")

        if self.width() <= 1:
            raise Exception("Cannot remove horizontal seam on width <= 1")

        picture = Image.new('RGB', (self.height(), self.width()), color=None)
        for i in range(self.height()):
            k = 0
            for j in range(self.width()):
                if i != seam[j]:
                    picture.putpixel((i, k), self.rgb()[j, i])
                    k += 1
        return picture


sc = SeamCarver('5x6.png')
print('vertical seam =>>', sc.findVerticalSeam())
print('horizontal seam =>>', sc.findHorizontalSeam())
print(sc.removeHorizontalSeam(sc.findHorizontalSeam()))
print(sc.removeVerticalSeam(sc.findVerticalSeam()))
