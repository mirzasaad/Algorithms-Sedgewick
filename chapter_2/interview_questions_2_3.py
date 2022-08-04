

from collections import namedtuple
from copy import copy
import doctest
import random
from typing import List

ThreeWaySample = namedtuple('ThreeWaySample', 'index value')

class NutsAndBolts(object):
    """
        Nuts and bolts.
    A disorganized carpenter has a mixed pile of nn nuts and nn bolts. 
    The goal is to find the corresponding pairs of nuts and bolts. 
    Each nut fits exactly one bolt and each bolt fits exactly one nut. 
    By fitting a nut and a bolt together, the carpenter can see which one is bigger 
    (but the carpenter cannot compare two nuts or two bolts directly).
    Design an algorithm for the problem that uses at most proportional to n logn compares (probabilistically).
    >>> nb = NutsAndBolts()
    >>> nuts = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i']
    >>> bolts = ['i', 'u', 'y', 't', 'r', 'e', 'w', 'q']
    >>> nb.sort(nuts, bolts)
    >>> print(nuts == bolts)
    True
    """

    def partition(self, a, lo, hi,  pivot):
        i = lo
        j = lo

        while (j < hi):
            if a[j] < pivot:
                a[i], a[j] = a[j], a[i]
                i += 1
            elif a[j] == pivot:
                a[hi], a[j] = a[j], a[hi]
                j -= 1
            j += 1

        a[hi], a[i] = a[i], a[hi]

        return i

    def __partition(self, lst, lo, hi, pivot):
        next_node = lo
        runner = lo

        while runner < hi:
            if lst[runner] < pivot:
                lst[next_node], lst[runner] = lst[runner], lst[next_node]
                next_node += 1
                runner += 1
            elif lst[runner] == pivot:
                lst[hi], lst[runner] = lst[runner], lst[hi]
            else:
                runner += 1

        lst[hi], lst[next_node] = lst[next_node], lst[hi]

        return next_node

    def __sort(self, nuts, bolts, lo, hi):
        if (hi <= lo):
            return

        pivot_index = self.__partition(nuts, lo, hi, nuts[hi])

        self.__partition(bolts, lo, hi, nuts[pivot_index])

        self.__sort(nuts, bolts, lo, pivot_index - 1)
        self.__sort(nuts, bolts, pivot_index + 1, hi)

    def sort(self, nuts, bolts):
        if len(bolts) != len(nuts):
            raise Exception('Nuts And Bolts Should be of same length!')

        self.__sort(nuts, bolts, 0, len(nuts) - 1)

class SelectionKTwoSortedArrays(object):
    """
        Selection in two sorted arrays
        >>> sel = SelectionKTwoSortedArrays()
        >>> sel.select([1, 7, 11, 17, 21], [2, 8, 14, 20, 28], 3)
        7
    """

    def select(self, lst1, lst2, k):
        m, n = len(lst1), len(lst2)
        
        if k == 0 or m + n < k:
            return None
        
        i, j = 0, 0
        total = 0
        
        while i < m and j < n and total < k:
            if lst1[i] < lst2[j]:
                total += 1
                if total == k:
                    return lst1[i]
                i += 1
            elif lst2[j] < lst1[i]:
                total += 1
                if total == k:
                    return lst2[j]
                j += 1
        
        while i < m:
            total += 1
            if total == k:
                return lst1[i]
            i += 1
            
        while j <  n:
            total += 1
            if total == k:
                return lst2[j]
            j += 1
        
        return -1

class DecimalDominant(object):
    """
        Decimal dominants. Given an array with nn keys, 
    design an algorithm to find all values that occur more than n/10 times. 
    The expected running time of your algorithm should be linear.
    >>> qs = DecimalDominant()
    >>> arr = [1, 2, 1, 2, 1, 2, 1, 4, 3, 2, 4, 2, 3, 5, 6, 7, 8, 9, 3, 2, 4, 5, 2, 3, 6, 7, 8, 3, 2, 5]
    >>> qs.kth_get_dominant(arr, k=10)
    {1, 2, 3}
    """ 
    def __partition(self, lst, lo, hi, k, result: set):
        N = len(lst)
        pivot = lst[lo]
        lt, gt = lo, hi
        runner = lo + 1

        while runner <= gt:
            if lst[runner] < pivot:
                lst[runner], lst[lt] = lst[lt], lst[runner]
                runner += 1
                lt += 1
            elif lst[runner] > pivot:
                lst[runner], lst[gt] = lst[gt], lst[runner]
                gt -= 1
            else:
                runner += 1

        length = gt - lt + 1

        if N / k < length:
            result.add(lst[lt])

        return lt, gt


    def __get_dominant(self, lst, lo, hi, k, result):
        if hi <= lo:
            return

        lt, gt = self.__partition(lst, lo, hi, k, result)
        self.__get_dominant(lst, lo, lt - 1, k, result)
        self.__get_dominant(lst, gt + 1, hi, k, result)


    def kth_get_dominant(self, lst, k):
        random.shuffle(lst) 
        result = set()
        self.__get_dominant(lst, 0, len(lst) - 1, k, result)
        return result

if __name__ == '__main__':
    doctest.testmod()