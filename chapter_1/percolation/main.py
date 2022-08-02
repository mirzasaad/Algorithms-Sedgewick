import random
from percolation import Percolation
import matplotlib.pyplot as plt

N = 5
p = Percolation(N)

while not p.percolates():
    row, col = random.randint(1, N), random.randint(1, N)
    if p.is_valid_index(row, col):
        p.open(row, col)

plt.imshow(p.get2d())
plt.show()