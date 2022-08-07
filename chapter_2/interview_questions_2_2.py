import doctest
from typing import List
from common import Node

class MergeWithSmallAux(object):
    """
        Merging with smaller auxiliary array. 
    Suppose that the subarray a[0] to a[n−1] is sorted and the subarray a[n] to a[2∗n−1] is sorted. 
    How can you merge the two subarrays so that a[0] a[2∗n−1] is sorted using an auxiliary array of length nn (instead of 2n2n)?
    >>> A = [40, 61, 70, 71, 99, 20, 51, 55, 75, 100]
    >>> mwsa = MergeWithSmallAux()
    >>> mwsa.merge(A)
    >>> print(A)
    [20, 40, 51, 55, 61, 70, 71, 75, 99, 100]
    """

    def merge(self, lst: List[int]):
        N = (len(lst) // 2)
        aux = [None] * N
        self.__merge(lst, aux, N)


    def __merge(self, lst: List[int], aux: List[int], mid: int):
        for k in range(mid):
            aux[k] = lst[k]
        
        #i - index of aux array
        #j - index of right part of a
        #k - index of merged array
        i = 0
        j = mid
        k = 0
        length = len(lst)
        while k < length:
            if j >= length:
                lst[k] = aux[i]
                k += 1
                i += 1
            elif i >= mid:
                lst[k] = lst[j]
                j += 1
                k += 1
            elif lst[j] > aux[i]:
                lst[k] = aux[i]
                k += 1
                i += 1
            else:
                lst[k] = lst[j]
                j += 1
                k += 1

# 2.2.19 practice, using merge function from merge-sort to count the reverse number
class ReverseCount(object):

    """
        Counting inversions. An inversion in an array a[] is a pair of entries a[i]
    and a[j] such that i<j but a[i] > a[j]. Given an array, design a
    linearithmic algorithm to count the number of inversions.

    Inversions. Develop and implement a linearithmic 
    algorithm for computing the number of inversions 
    in a given array (the number of exchanges that would be 
    performed by insertion sort for that array)
    >>> rc = ReverseCount()
    >>> rc.reverse_count([1, 7, 2, 9, 6, 4, 5, 3])
    14
    """

    def __merge_reverse_count(self, lst, lo, mid, hi):
        left = lst[lo: mid + 1]
        right = lst[mid + 1: hi + 1]
        
        i, j, k = 0, 0, lo
        m, n = len(left), len(right)
        count = 0
        mid = m - 1
        while  i < m and j < n:
            if left[i] < right[j]:
                lst[k] = left[i]
                i += 1
            else:
                lst[k] = right[j]
                j += 1
                # if lst[j] > lst[i] then we calculate inversion, calculating number if inversions
                count += mid - i + 1
            k += 1

        while i < m:
            lst[k] = left[i]
            k += 1
            i += 1

        while j < n:
            lst[k] = right[j]
            k += 1
            j += 1

        return count

    def __reverse_count(self, lst: List[int], lo: int, hi: int):
        if hi <= lo:
            return 0

        mid = lo + ((hi - lo) // 2)
        left_count = self.__reverse_count(lst, lo, mid)
        right_count = self.__reverse_count(lst, mid + 1, hi)
        count = self.__merge_reverse_count(lst, lo, mid, hi)
        return left_count + right_count + count

    def reverse_count(self, lst):
        llst = lst[:]
        return self.__reverse_count(llst, 0, len(lst) - 1)


class LinkedListShuffle(object):
    """
        Shuffling a linked list.
    Given a singly-linked list containing N items, rearrange the items uniformly at random.
    Your algorithm should consume a logarithmic (or constant) amount of extra memory and run in time proportional to NlogN in the worst case.

    Solution => break linked list in 2 recursevly like mergesort and then stich them together with merge operation
    """

    def __merge(self):
        pass

    def __shuffle(self):
        pass

    def shuffle(self, head: Node, size: int):
        self.__shuffle

if __name__ == '__main__':
    doctest.testmod()

