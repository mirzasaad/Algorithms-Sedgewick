import doctest
from operator import index
import random
import bisect
import re
from textwrap import indent

from common import Cube


class MaxPQ(object):

    """
    >>> mpq = MaxPQ(10)
    >>> lst = [i for i in range(10)]
    >>> random.shuffle(lst)
    >>> for i in lst:
    ...     mpq.insert_effective(i)
    ...
    >>> mpq.min_val()
    0
    >>> print_lst = []
    >>> while not mpq.is_empty():
    ...     print_lst.append(str(mpq.del_max()))
    ...
    >>> ' '.join(print_lst)
    '9 8 7 6 5 4 3 2 1 0'
    """

    def __init__(self, size):
        self._pq = [None] * (size + 1)
        self._size = 0
        self._min = None

    def is_empty(self):
        return self._size == 0

    def size(self):
        return self._size

    def max_val(self):
        return self._pq[1]

    def min_val(self):
        return self._min

    def swim(self, index):
        while index > 1 and self._pq[index] > self._pq[index // 2]:
            self._pq[index], self._pq[index //
                                      2] = self._pq[index // 2], self._pq[index]
            index //= 2

    def sink(self, pos):
        while 2 * pos <= self._size:
            index = 2 * pos
            if index < self._size and self._pq[index] < self._pq[index + 1]:
                index += 1

            if self._pq[pos] >= self._pq[index]:
                break

            self._pq[index], self._pq[pos] = self._pq[pos], self._pq[index]
            pos = index

    def insert(self, value):
        self._size += 1
        self._pq[self._size] = value
        if self._min is None or self._min > value:
            self._mi = value
        self.swim(self._size)

    def del_max(self):
        max_val = self._pq[1]
        self._pq[self._size], self._pq[1] = self._pq[1], self._pq[self._size]
        self._pq[self._size] = None
        self._size -= 1
        self.sink(1)
        return max_val

    def swim_effective(self, index):
        val = self._pq[index]
        while index > 1 and val > self._pq[index // 2]:
            self._pq[index] = self._pq[index // 2]
            index //= 2
        self._pq[index] = val

    def insert_effective(self, val):
        self._size += 1
        self._pq[self._size] = val
        if self._min is None or self._min > val:
            self._min = val
        self.swim_effective(self._size)


class MinPQ(object):

    """
    >>> mpq = MinPQ(10)
    >>> lst = [i for i in range(10)]
    >>> random.shuffle(lst)
    >>> for i in lst:
    ...     mpq.insert(i)
    ...
    >>> print_lst = []
    >>> while not mpq.is_empty():
    ...     print_lst.append(str(mpq.del_min()))
    ...
    >>> ' '.join(print_lst)
    '0 1 2 3 4 5 6 7 8 9'
    """

    def __init__(self, size):
        self._pq = [None] * (size + 1)
        self._size = 0

    def is_empty(self):
        return self._size == 0

    def size(self):
        return self._size

    def insert(self, val):
        self._size += 1
        self._pq[self._size] = val
        self.swim(self._size)

    def swim(self, pos):
        while pos > 1 and self._pq[pos // 2] > self._pq[pos]:
            self._pq[pos], self._pq[pos //
                                    2] = self._pq[pos // 2], self._pq[pos]
            pos //= 2

    def sink(self, pos):
        while 2 * pos <= self._size:
            index = pos * 2
            if index < self._size and self._pq[index + 1] < self._pq[index]:
                index += 1

            if self._pq[pos] < self._pq[index]:
                break

            self._pq[pos], self._pq[index] = self._pq[index], self._pq[pos]

            pos = index

    def del_min(self):
        min_val = self._pq[1]
        self._pq[1], self._pq[self._size] = self._pq[self._size], self._pq[1]
        self._pq[self._size] = None
        self._size -= 1
        self.sink(1)
        return min_val

    def min_val(self):
        return self._pq[1]

# 2.4.22 practice, a little change for python version, the queue's size is not limited.


class MaxPQDynamic(object):

    """
    >>> mpq = MaxPQDynamic()
    >>> lst = [i for i in range(10)]
    >>> random.shuffle(lst)
    >>> for i in lst:
    ...     mpq.insert(i)
    ...
    >>> print_lst = []
    >>> while not mpq.is_empty():
    ...     print_lst.append(str(mpq.del_max()))
    ...
    >>> ' '.join(print_lst)
    '9 8 7 6 5 4 3 2 1 0'
    """

    def __init__(self):
        self._pq = []

    def __repr__(self):
        return '[' + ', '.join([str(item) for item in self._pq]) + ']'

    def is_empty(self):
        return len(self._pq) == 0

    def size(self):
        return len(self._pq)

    def swim(self, k):
        while k > 0 and self._pq[(k - 1) // 2] < self._pq[k]:
            self._pq[k], self._pq[(
                k - 1) // 2] = self._pq[(k - 1) // 2], self._pq[k]
            k = (k - 1) // 2

    def sink(self, k):
        while 2 * k + 1 <= self.size() - 1:
            index = 2 * k + 1
            if index < self.size() - 1 and self._pq[index] < self._pq[2 * k + 2]:
                index += 1

            if self._pq[k] >= self._pq[index]:
                break

            self._pq[index], self._pq[k] = self._pq[k], self._pq[index]

            k = index

    def insert(self, val):
        self._pq.append(val)
        self.swim(self.size() - 1)

    def del_max(self):
        val = self._pq[0]
        last_index = len(self._pq) - 1
        self._pq[last_index], self._pq[0] = self._pq[0], self._pq[last_index]
        self._pq.pop(last_index)
        self.sink(0)
        return val

    def max(self):
        return self._pq[0] if self.size() else 0


class MinPQDynamic(object):

    """
    >>> mpq = MinPQDynamic()
    >>> lst = [i for i in range(10)]
    >>> random.shuffle(lst)
    >>> for i in lst:
    ...     mpq.insert(i)
    ...
    >>> print_lst = []
    >>> while not mpq.is_empty():
    ...     print_lst.append(str(mpq.del_min()))
    ...
    >>> ' '.join(print_lst)
    '0 1 2 3 4 5 6 7 8 9'
    """

    def __init__(self):
        self._pq = []

    def is_empty(self):
        return len(self._pq) == 0

    def size(self):
        return len(self._pq)

    def min(self):
        return self._pq[0] if self.size() else 0

    def binary_swim(self, pos):
        index, vals, temp, target = [], [], pos, self._pq[pos]
        while temp:
            index.append(temp)
            vals.append(self._pq[temp])
            temp = (temp - 1) // 2

        insert_pos = bisect.bisect_left(vals, target)
        if insert_pos == len(vals):
            return

        i = insert_pos - 1
        while i < len(vals) - 1:
            self._pq[index[i + 1]] = self._pq[index[i]]
            i += 1

        self._pq[insert_pos - 1] = target

    def swim(self, k):
        while k > 0 and self._pq[(k - 1) // 2] > self._pq[k]:
            self._pq[(k - 1) // 2], self._pq[k] = self._pq[k], self._pq[(k - 1) // 2]
            k = (k - 1) // 2

    def insert(self, val):
        self._pq.append(val)
        self.swim(self.size() - 1)

    def sink(self, k):
        length = len(self._pq) - 1
        while 2 * k + 1 <= length:
            index = 2 * k + 1
            if index < length and self._pq[index] > self._pq[index + 1]:
                index += 1

            if self._pq[k] <= self._pq[index]:
                break

            self._pq[k], self._pq[index] = self._pq[index], self._pq[k]

            k = index

    def del_min(self):
        min_val = self._pq[0]
        last_index = len(self._pq) - 1
        self._pq[0], self._pq[last_index] = self._pq[last_index], self._pq[0]
        self._pq.pop(last_index)
        self.sink(0)
        return min_val

# 2.4.30 practice


class MeanHeap(object):

    """
    >>> mh = MeanHeap()
    >>> for i in range(9):
    ...     mh.insert(i)
    ...
    >>> mh.median()
    4
    >>> mh.insert(9)
    >>> mh.median()
    4.5
    >>> mh.insert(10)
    >>> mh.median()
    5
    """

    def __init__(self):
        self._min_heap = MinPQDynamic()
        self._max_heap = MaxPQDynamic()

    def size(self):
        return self._min_heap.size() + self._max_heap.size()

    def is_empty(self):
        return self._min_heap.is_empty() and self._max_heap.is_empty()

    def median(self):
        if self.is_empty():
            return 0

        if self._max_heap.size() == self._min_heap.size():
            return (self._max_heap.max() + self._min_heap.min()) / 2

        if self._max_heap.size() < self._min_heap.size():
            return self._min_heap.min()
        else:
            return self._max_heap.max()

    def insert(self, value):
        if not self.size():
            self._min_heap.insert(value)
            return

        if value <= self.median():
            self._max_heap.insert(value)
        else:
            self._min_heap.insert(value)

        self.__rebalance()

    def __rebalance(self):
        if self._max_heap.size() == self._min_heap.size():
            return

        if abs(self._max_heap.size() - self._min_heap.size()) <= 1:
            return

        if self._max_heap.size() < self._min_heap.size():
            self._max_heap.insert(self._min_heap.del_min())
        else:
            self._min_heap.insert(self._max_heap.del_max())

# 2.4.33, 2.4.34 index minimum priority queue.


class IndexMinPQ(object):

    """
    >>> test_data = 'testexmaple'
    >>> imp = IndexMinPQ(len(test_data))
    >>> imp.is_empty()
    True
    >>> for index, s in enumerate(test_data):
    ...     imp.insert(index, s)
    ...
    >>> imp.is_empty()
    False
    >>> imp.size()
    11
    >>> [imp.contains(i) for i in (12, -1, 1, 4, 10)]
    [False, False, True, True, True]
    >>> imp.min_index()
    7
    """

    def __init__(self, max_size):
        assert max_size > 0
        self._max_size = max_size
        self._index = [-1] * (max_size + 1)
        self._reverse_index = [-1] * (max_size + 1)
        self._keys = [None] * (max_size + 1)
        self._keys_size = 0

    def is_empty(self):
        return self._keys_size == 0

    def size(self):
        return self._keys_size

    def contains(self, index):
        if index < 0 or index >= self._max_size:
            return False
        return self._reverse_index[index] != -1

    def insert(self, index, element):
        if index < 0 or index >= self._max_size or self.contains(index):
            return

        self._keys_size += 1
        self._index[self._keys_size] = index
        self._reverse_index[index] = self._keys_size
        self._keys[index] = element
        self.swim(self._keys_size)

    def min_index(self):
        return None if self._keys_size == 0 else self._index[1]

    def min_key(self):
        return None if self._keys_size == 0 else self._keys[self._index[1]]

    def exchange(self, pos_a, pos_b):
        self._index[pos_a], self._index[pos_b] = self._index[pos_b], self._index[pos_a]
        self._reverse_index[self._index[pos_a]] = pos_a
        self._reverse_index[self._index[pos_b]] = pos_b

    def swim(self, pos):
        while pos > 1 and self._keys[self._index[pos // 2]] > self._keys[self._index[pos]]:
            self.exchange(pos // 2, pos)
            pos //= 2

    def sink(self, pos):
        length = self._keys_size
        while 2 * pos <= length:
            tmp = 2 * pos
            if tmp < length and self._keys[self._index[tmp]] > self._keys[self._index[tmp + 1]]:
                tmp += 1
            if not self._keys[self._index[tmp]] < self._keys[self._index[pos]]:
                break
            self.exchange(tmp, pos)
            pos = tmp

    def change_key(self, i, key):
        if i < 0 or i >= self._max_size or not self.contains(i):
            return
        self._keys[i] = key
        self.swim(self._reverse_index[i])
        self.sink(self._reverse_index[i])

    def delete_min(self):
        if self._keys_size == 0:
            return
        min_index = self._index[1]
        self.exchange(1, self._keys_size)
        self._keys_size -= 1
        self.sink(1)
        self._reverse_index[min_index] = -1
        self._keys[self._index[self._keys_size + 1]] = None
        self._index[self._keys_size + 1] = -1
        return min_index

# 2.4.25 practice, cube sum implementation.


def taxi_cab_numbers(n):
    """
        Cube Sum / Taxi cab Numeber 
        a^3 + b^3 = c^3 + d^3
        find number who satisfy this formula
    >>> taxi_cab_numbers(12)
    (9^3 + 10^3 == 1729) (1^3 + 12^3 == 1729)
    """
    pq = MinPQDynamic()
    for i in range(n):
        pq.insert(Cube(i, i))

    while not pq.is_empty():
        last = pq.del_min()
        second_last = pq.min() if pq.size() >= 1 else None
        if second_last and last._sum == second_last._sum:
            print(last, second_last)
        if last.j < n:
            pq.insert(Cube(last.i, last.j + 1))


class HeapSort(object):
    """
      Heap-sort implementation, using priority queue sink() method as util function,
    first build the maximum priority queue, and exchange list[0] and lst[size], then size minus one,
    and sink the list[0] again, util size equals zero.
    >>> hs = HeapSort()
    >>> lst = [i for i in range(10)]
    >>> random.shuffle(lst)
    >>> hs.sort(lst)
    >>> lst
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    def sink(self, lst, pos, size):
        while 2 * pos + 1 <= size:
            index = 2 * pos + 1
            if index < size and lst[index + 1] > lst[index]:
                index += 1
            if lst[pos] >= lst[index]:
                break
            lst[pos], lst[index] = lst[index], lst[pos]
            pos = index

    def __heapify(self, lst):
        length = len(lst) - 1
        for i in reversed(range(length // 2)):
            self.sink(lst, i, length)

    def __sort(self, lst):
        k = len(lst) - 1
        while k:
            lst[0], lst[k] = lst[k], lst[0]
            k -= 1
            self.sink(lst, 0, k)

    def sort(self, lst):
        self.__heapify(lst)
        self.__sort(lst)

if __name__ == '__main__':
    doctest.testmod()
