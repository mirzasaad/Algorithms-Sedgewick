from PIL import Image
from tabulate import tabulate

class Energy():
    def __init__(self, image):
        im = Image.open(image)
        self._pixels = im.load()
        self._cols = im.size[0]
        self._rows = im.size[1]

        self._table = []
        self._energy_table = []

        for r in range(self._rows):
            row = []
            for c in range(self._cols):
                row.append(self._pixels[c, r])
            self._table.append(row)

        for r in range(self._rows):
            row = []
            for c in range(self._cols):
                row.append(self.energy(r, c))
            self._energy_table.append(row)

    def dx(self, x, y):
        c0 = self._pixels[x - 1, y]
        c1 = self._pixels[x + 1, y]
        return self.getGrad(c0, c1)

    def dy(self, x, y):
        c0 = self._pixels[x, y - 1]
        c1 = self._pixels[x, y + 1]
        return self.getGrad(c0, c1)

    def getGrad(self, c0, c1):
        r = c0[0] - c1[0]
        g = c0[1] - c1[1]
        b = c0[2] - c1[2]
        return r*r + g*g + b*b

    def energy(self, x, y):
        if x == 0 or y == 0 or x == self._rows - 1 or y == self._cols - 1:
            return 1000
        return self.dx(y, x) + self.dy(y, x)


# e = Energy('./5x6.png')

# print('rows =>>', e._rows, 'cols =>>', e._cols)
# # print(tabulate(e._table, list(range(e._cols)), tablefmt="grid"))
# print(tabulate(e._energy_table, list(range(e._cols)), tablefmt="grid"))
