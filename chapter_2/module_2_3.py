

from collections import namedtuple
import doctest
import random
import re
from select import select
from typing import List

from numpy import sort

INSERTION_SORT_LENGTH = 8

ThreeWaySample = namedtuple('ThreeWaySample', 'index value')


class QuickSort(object):

    """
    >>> qs = QuickSort()
    >>> lst = [3, 2, 4, 7, 8, 9, 1, 0, 14, 11, 23, 50, 26]
    >>> qs.sort(lst)
    >>> lst
    [0, 1, 2, 3, 4, 7, 8, 9, 11, 14, 23, 26, 50]
    >>> lst2 = ['E', 'A', 'S', 'Y', 'Q', 'U', 'E', 'S', 'T', 'I', 'O', 'N']
    >>> qs.sort(lst2)
    >>> lst2
    ['A', 'E', 'E', 'I', 'N', 'O', 'Q', 'S', 'S', 'T', 'U', 'Y']
    >>> lst1 = [3, 2, 4, 7, 8, 9, 1, 0, 14, 11, 23, 50, 26]
    >>> k = 2
    >>> qs.select(lst1, 2)
    2
    """

    def __insertion_sort(self, lst, lo, hi):
        for i in range(lo, hi + 1):
            j = i
            while j > 0:
                if lst[j] < lst[j - 1]:
                    lst[j], lst[j - 1] = lst[j - 1], lst[j]
                j -= 1

    # 2.3.18 practice
    def three_sample(self, lst, low, mid, high):
        if lst[low] <= lst[mid] <= lst[high] or lst[high] <= lst[mid] <= lst[low]:
            return mid
        elif lst[mid] <= lst[low] <= lst[high] or lst[high] <= lst[low] <= lst[mid]:
            return low
        else:
            return high

    # 2.3.19 practice
    def __five_sample(self, lst, low, high):
        values: List[ThreeWaySample] = []
        for _ in range(5):
            index = random.randint(low, high)
            values.append(ThreeWaySample(index, lst[index]))

        values.sort(key=lambda x: x.value)

        mid = (len(values) - 1) // 2
        return values[mid].index

    def __partition(self, lst, lo, hi):
        random_pivot_index = self.__five_sample(lst, lo, hi)

        pivot = lst[random_pivot_index]
        next_index = lo

        lst[random_pivot_index], lst[hi] = lst[hi], lst[random_pivot_index]

        for i in range(lo, hi + 1):
            if lst[i] < pivot:
                lst[next_index], lst[i] = lst[i], lst[next_index]
                next_index += 1

        lst[next_index], lst[hi] = lst[hi], lst[next_index]

        return next_index

    def __sort(self, lst, lo, hi):
        if lo >= hi:
            return

        pivot_index = self.__partition(lst, lo, hi)
        self.__sort(lst, 0, pivot_index - 1)
        self.__sort(lst, pivot_index + 1, hi)

    def ___sort(self, lst, lo, hi):
        # length = high - low + 1
        # calculate lenfth between hi and lo, and check if it is less than insertion sort cutoff point
        if lo + INSERTION_SORT_LENGTH >= hi:
            return self.__insertion_sort(lst, lo, hi)

        pivot_index = self.__partition(lst, lo, hi)
        self.__sort(lst, 0, pivot_index - 1)
        self.__sort(lst, pivot_index + 1, hi)

    def sort(self, lst):
        random.shuffle(lst)
        self.___sort(lst, 0, len(lst) - 1)

    def select(self, lst, k):
        lo,  hi = 0, len(lst) - 1

        # pivot_index = self.__partition(lst, lo, hi)

        while lo <= hi:
            pivot_index = self.__partition(lst, lo, hi)
            if pivot_index == k:
                return lst[pivot_index]
            elif pivot_index < k:
                lo = pivot_index + 1
            else:
                hi = pivot_index - 1

        return None


class QuickThreeWay(object):

    """
    >>> qtw = QuickThreeWay()
    >>> lst = [3, 2, 4, 7, 8, 9, 1, 0]
    >>> qtw.sort(lst)
    >>> lst
    [0, 1, 2, 3, 4, 7, 8, 9]
    """

    def sort(self, lst):
        random.shuffle(lst)
        self.__sort(lst, 0, len(lst) - 1)

    def __partition(self, lst, lo, hi):
        lt, runner, gt, pivot = lo, lo + 1, hi, lst[lo]
        while runner <= gt:
            if lst[runner] < pivot:
                lst[lt], lst[runner] = lst[runner], lst[lt]
                lt += 1
                runner += 1
            elif lst[runner] > pivot:
                lst[gt], lst[runner] = lst[runner], lst[gt]
                gt -= 1
            else:
                runner += 1

        return lt, gt
    def __sort(self, lst, low, high):
        if high <= low:
            return

        lt, gt = self.__partition(lst, low, high)

        self.__sort(lst, low, lt - 1)
        self.__sort(lst, gt + 1, high)


class QuickSelectMedianArray(object):
    """
        Calculate Media in unsorted array, if the length of array
    is odd the return the middle index in array else, return 2 middle elements
    >>> lst = [3, 2, 4, 7, 8, 9, 1, 0]
    >>> ws = QuickSelectMedianArray()
    >>> ws.median(lst)
    [7, 4]
    """

    def __partition(self, lst, low, high):
        pivot = lst[low]
        lt = low
        gt = high
        runner = low + 1

        while lt < gt:
            if lst[runner] < pivot:
                lst[runner], lst[lt] = lst[lt], lst[runner]
                runner += 1
                lt += 1
            elif lst[runner] > pivot:
                lst[runner], lst[gt] = lst[gt], lst[runner]
                gt -= 1
            else:
                runner += 1

        return lt

    def __median(self, lst: List[int], lo: int, hi: int, mid: int, result: List[int], isOdd: bool):
        lo,  hi = 0, len(lst) - 1

        pivot_index = self.__partition(lst, lo, hi)

        while lo <= hi:
            pivot_index = self.__partition(lst, lo, hi)
            if pivot_index == mid:
                result.append(lst[pivot_index])
                if (len(result) == 2):
                    return
            elif pivot_index == mid + 1:
                result.append(lst[pivot_index])
                if (len(result) == 2):
                    return
            elif pivot_index < mid:
                lo = pivot_index + 1
            else:
                hi = pivot_index - 1

        return None
            
    def median(self, lst):
        result = []
        length = len(lst)
        mid = length // 2
        self.__median(lst, 0, length - 1, mid, result, length % 2 != 0) 
        return result

if __name__ == '__main__':
    doctest.testmod()
