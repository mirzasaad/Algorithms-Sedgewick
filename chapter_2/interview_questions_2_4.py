


import doctest
import random
import re
from unittest import result
from common import Cube
from module_2_4 import MaxPQDynamic, MinPQDynamic
from prettytable import PrettyTable

class MeanHeap(object):

    """
    Question 1
        Dynamic median. Design a data type that supports insert in logarithmic time, 
    find-the-median in constant time, and remove-the-median in logarithmic time. 
    If the number of keys in the data type is even, find/remove the lower median.
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

        if self._min_heap.size() < self._max_heap.size():
            return self._max_heap.max()
        else:
            return  self._min_heap.min()

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

    def remove_median(self):
        if self.empty():
            return None

        if self._max_heap.size() == self._min_heap.size():
            return (self._max_heap.del_max() + self._min_heap.del_min()) // 2

        value = self._max_heap.del_max() if self._max_heap.size() > self._min_heap.size() else self._min_heap.del_min()
        
        self.__rebalance()

        return value

class RandomizedMaxPQDynamic(object):

    """
    >>> mpq = RandomizedMaxPQDynamic()
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

    def sample(self):
        index = random.randrange(0, self.size())
        return self._pq[index]

    def del_random(self):
        index = random.randrange(0, self.size())
        random_value = self._pq[index]
        self._pq[index], self._pq[self.size() - 1] = self._pq[self.size() - 1], self._pq[index]
        self._pq.pop(self.size() - 1)
        self.sink(index)
        return random_value

    def max(self):
        return self._pq[0] if self.size() else 0


"""
    Question 3
Taxicab numbers. A taxicab number is an integer that can be expressed as the sum of two cubes of 
positive integers in two different ways: a^3 + b^3 = c^3 + d^3a .
For example, 17291729 is the smallest taxicab number: 9^3 + 10^3 = 1^3 + 12^39 
Design an algorithm to find all taxicab numbers with aa, bb, cc, and dd less than nn.

VERSION 1

Create a matrix like this. Each cell is the sum of the cubes of row and column value 
m[i][j] =   i^3 + j^3

Sort all the values in the matrix, highest to lowest, using any logarithmic time algorithm, a binary sorted heap in this case
Pop out the items from the heap until the end
If exists any 4 continuous equal values, that’s the taxicab number
  | 0    1    2    3    4    5
--+-------------------------------
0 | 0    1    8    27   64   125
1 | 1    2    9    28   65   126
2 | 8    9    16   35   72   133
3 | 27   28   35   54   91   152
4 | 64   65   72   91   128  189
5 | 125  126  133  152  189  250

VERSION 2

Actually, I couldn’t think of the solution for this. I checked the answers on Stackoverflow and on the internet, took me too long to understand…

Start with the above matrix. However, since half of the matrix is duplicated, we only need to care about the other half. Of course, the matrix will never be created to save memory

  | 0    1    2    3    4    5
--+-------------------------------
0 | 0    1    8    27   64   125
1 |      2    9    28   65   126
2 |           16   35   72   133
3 |                54   91   152
4 |                     128  189
5 |                          250

Create a min heap and store the diagonal values. You can also use the max heap, simply do the reverse way.

Pop out the min value from the heap. For each value, if the current min equal to the previous min, we found the taxicab pair. Otherwise, add the value to the right of the current min to the heap and repeat there is no value left in the heap.

To illustrate it

Starting heap 2 16 54 128 250 (m[1,1],m[2,2],m[3,3],m[4,4],m[5,5])


Loop

Current heap 2 16 54 128 250
prevMin: 0
currMin: 2 (m[1,1])
Add 9 (m[2,1])


Loop

Current heap 9 16 54 128 250
prevMin: 2
currMin: 9 (m[2,1])
Add 28 (m[3,1])


Loop

Current heap 16 28 54 128 250
prevMin: 9
currMin: 16 (m[2,2])
Add 35 (m[3,2])


Loop

Current heap 28 35 54 128 250
prevMin: 16
currMin: 28 (m[3,1])
Add 65 (m[4,1])


Loop

Current heap 35 54 65 128 250
prevMin: 28
currMin: 35 (m[3,2])
Add 72 (m[4,2])
"""

class TaxiCabNumbers(object):
    """
        Cube Sum / Taxi cab Numeber 
    a^3 + b^3 = c^3 + d^3
    find number who satisfy this formula
    generate 2d matrix with i^3 + j^3 put them in min queue and 
    remove last 4 if they are equal we found the numbers

    If exists any 4 continuous equal values, that’s the taxicab number
    |   0    1    2    3    4    5
    --+-------------------------------
    0 | 0    1    8    27   64   125
    1 | 1    2    9    28   65   126
    2 | 8    9    16   35   72   133
    3 | 27   28   35   54   91   152
    4 | 64   65   72   91   128  189
    5 | 125  126  133  152  189  250

    >>> t = TaxiCabNumbers()
    >>> t.find(12)
    [(9, 12, 10, 1)]
    """
    def __init__(self) -> None:
        self._pq = MinPQDynamic()

    def find(self, size):
        result = []

        for i in range(size + 1):
            for j in range(size + 1):
                self._pq.insert(Cube(i, j))

        while self._pq.size():
            a = self._pq.del_min() if self._pq.size() else None
            b = self._pq.del_min() if self._pq.size() else None
            c = self._pq.del_min() if self._pq.size() else None
            d = self._pq.del_min() if self._pq.size() else None
            if a and b  and c and d and a == b == c == d:
                result.append((a.i, b.j, c.i, d.j))

        return result

class TaxiCabNumberFast(object):
    """
        Cube Sum / Taxi cab Numeber 
    a^3 + b^3 = c^3 + d^3
    find number who satisfy this formula
    generate just diagonal 2d matrix with i^3 + j^3 put them in min queue remove last and check if it equal last equal second last
    if not add adjacent matrix value

     add this

      | 0    1    2    3    4    5
    --+-------------------------------
    0 | 0                    
    1 |      2                
    2 |           16           
    3 |                54        
    4 |                     128    
    5 |                          250

    then add adjacent numbers

      | 0    1    2    3    4    5
    --+-------------------------------
    0 | 0    1               
    1 |      2                
    2 |           16           
    3 |                54        
    4 |                     128    
    5 |                          250

    then 

      | 0    1    2    3    4    5
    --+-------------------------------
    0 | 0    1     8         
    1 |      2                
    2 |           16           
    3 |                54        
    4 |                     128    
    5 |                          250

    then

      | 0    1    2    3    4    5
    --+-------------------------------
    0 | 0    1     8         
    1 |      2     9          
    2 |           16           
    3 |                54        
    4 |                     128    
    5 |                          250

    if second and last values are equal, numbers found

    >>> t = TaxiCabNumberFast()
    >>> t.find(12)
    (9^3 + 10^3 == 1729) (1^3 + 12^3 == 1729)
    """
    def __init__(self) -> None:
        self._pq = MinPQDynamic()

    def find(self, size):
        for i in range(size + 1):
            self._pq.insert(Cube(i, i))

        while not self._pq.is_empty():
            last = self._pq.del_min()
            second_last = self._pq.min() if self._pq.size() >= 1 else None
            if second_last and last._sum == second_last._sum:
                print(last, second_last)
            if last.j < size:
                self._pq.insert(Cube(last.i, last.j + 1))

if __name__ == '__main__':
    doctest.testmod()