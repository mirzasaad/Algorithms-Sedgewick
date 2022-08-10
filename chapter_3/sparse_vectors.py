

import doctest

class SparseVector(object):
    """
        Sparse Vectors
    >>> N = 5
    >>> matrix = [ 0, 0.90, 0, 0, 0, 0, 0, 0.36, 0.36, 0.18, 0, 0, 0, 0.90, 0, 0.90, 0, 0, 0, 0, 0.47, 0, 0.47, 0, 0]
    >>> spv_1 = SparseVector(N * N - 1)
    >>> spv_2 = SparseVector(N * N - 1)
    >>> for index, number in enumerate(matrix):
    ...     spv_1.put(index, number)
    >>> for index, number in enumerate(matrix):
    ...     spv_2.put(index, number)
    >>> print(spv_1.dot(spv_2))
    3.1633999999999998
    >>> print(spv_1.plus(spv_2).dict)
    {1: 1.8, 7: 0.72, 8: 0.72, 9: 0.36, 13: 1.8, 15: 1.8, 20: 0.94, 22: 0.94}
    >>> print(spv_1.scale(0.3).dict)
    {1: 0.27, 7: 0.108, 8: 0.108, 9: 0.054, 13: 0.27, 15: 0.27, 20: 0.141, 22: 0.141}
    """

    def __init__(self, n):
        self.dimension = n
        self.dict = {}

    def put(self, index, value):
        if value == 0.0:
            return
        self.dict[index] = value

    def get(self, index):
        return self.dict[index] if index in self.dict else 0.0

    def contains(self, index):
        return self.get(index) != 0.0

    def size(self):
        return self.dimension

    def dot(self, other):
        if self.dimension != other.dimension:
            raise Exception('Vector lengths disagree')

        sum = 0.0

        # iterate over the vector with the fewest nonzeros
        if self.size() <= other.size():
            for key in self.dict.keys():
                if other.contains(key):
                    sum += other.get(key) * self.get(key)

        else:
            for k in other.dict.keys():
                if self.contains(k):
                    sum += other.get(k) * self.get(k)
        return sum

    def scale(self, alpha):
        temp = SparseVector(self.dimension)
        for key in self.dict.keys():
            temp.put(key, alpha * self.get(key))

        return temp

    def plus(self, other):
        if self.dimension != other.dimension:
            raise Exception('Vector lengths disagree')

        temp = SparseVector(other.dimension)

        for key in self.dict.keys():
            temp.put(key, self.get(key))

        for key in other.dict.keys():
            temp.put(key, self.get(key) + other.get(key))

        return temp


if __name__ == '__main__':
    doctest.testmod()
