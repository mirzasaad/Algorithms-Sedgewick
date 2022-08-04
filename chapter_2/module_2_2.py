import doctest
import random
import sys
from typing import List

from common import Node

class MergeSort(object):

    """
      Top-bottom merge sort implementation, merge the two sub arrays
    of the whole list and make the list partial ordered,
    and the recursion process make sure the whole list is ordered.
    for a N-size array, top-bottom merge sort need 1/2NlgN to NlgN comparisons,
    and need to access array 6NlgN times at most.
    >>> ms = MergeSort()
    >>> lst = [4, 3, 2, 5, 7, 9, 0, 1, 8, 7, -1, 11, 13, 31, 24]
    >>> ms.sort(lst)
    >>> lst
    [-1, 0, 1, 2, 3, 4, 5, 7, 7, 8, 9, 11, 13, 24, 31]
    """

    def __merge(self, lst, lo, mid, hi):
        left = lst[lo:mid + 1]
        right = lst[mid + 1: hi + 1]

        if left[-1] < right[0]:
            return

        i, j, k = 0, 0, lo
        m, n = len(left), len(right)

        while i < m and j < n:
            if left[i] < right[j]:
                lst[k] = left[i]
                i += 1
            else:
                lst[k] = right[j]
                j += 1

            k += 1

        while i < m:
            lst[k] = left[i]
            i += 1
            k += 1

        while j < n:
            lst[k] = right[j]
            j += 1
            k += 1

    def __sort(self, lst, lo, hi):
        if hi <= lo:
            return

        mid = lo + ((hi - lo) // 2)
        self.__sort(lst, lo, mid)
        self.__sort(lst, mid + 1, hi)
        self.__merge(lst, lo, mid, hi)

    def sort(self, lst):
        self.__sort(lst, 0, len(lst) - 1)


class MergeSortBU(object):

    """
      Bottom-up merge sort algorithm implementation, cut the whole N-size array into
    N/sz small arrays, then merge each two of them,
    the sz parameter will be twice after merge all the subarrays,
    util the sz parameter is larger than N.
    >>> ms = MergeSortBU()
    >>> lst = [4, 3, 2, 5, 7, 9, 0, 1, 8, 7, -1]
    >>> ms.sort(lst)
    >>> lst
    [-1, 0, 1, 2, 3, 4, 5, 7, 7, 8, 9]
    """

    def sort(self, lst):
        length = len(lst)
        aux = [None] * length
        size = 1
        while size < length:
            for i in range(0, length - size, size * 2):
                self.merge(aux, lst, i, i + size - 1,
                           min(i + size * 2 - 1, length - 1))
            size *= 2

    def merge(self, aux, lst, low, mid, high):
        left, right = low, mid + 1
        for i in range(low, high + 1):
            aux[i] = lst[i]

        for j in range(low, high + 1):
            if left > mid:
                lst[j] = aux[right]
                right += 1
            elif right > high:
                lst[j] = aux[left]
                left += 1
            elif aux[left] < aux[right]:
                lst[j] = aux[left]
                left += 1
            else:
                lst[j] = aux[right]
                right += 1

# 2.2.14 practice merge two sorted list
def merge_list(lst1, lst2):
    """
    >>> merge_list([1, 2, 3, 4], [])
    [1, 2, 3, 4]
    >>> merge_list([], [1, 2, 3, 4])
    [1, 2, 3, 4]
    >>> merge_list([1, 2, 3, 4], [4, 5, 6])
    [1, 2, 3, 4, 4, 5, 6]
    >>> merge_list([1, 2, 3, 4], [1, 2, 3, 4])
    [1, 1, 2, 2, 3, 3, 4, 4]
    >>> merge_list([1, 2], [5, 6, 7, 8])
    [1, 2, 5, 6, 7, 8]
    >>> merge_list([2, 3, 5, 9], [2, 7, 11])
    [2, 2, 3, 5, 7, 9, 11]
    """


    i, j, k = 0, 0, 0
    m, n = len(lst1), len(lst2)
    size = m + n

    aux = [None] * size

    while i < m and j < n:
        if lst1[i] <= lst2[j]:
            aux[k] = lst1[i]
            i += 1
        else:
            aux[k] = lst2[j]
            j += 1
        k += 1

    while i < m:
        aux[k] = lst1[i]
        i += 1
        k += 1

    while j < n:
        aux[k] = lst2[j]
        j += 1
        k += 1

    return aux

class MergeSort(object):

    """
      Top-bottom merge sort implementation, merge the two sub arrays
    of the whole list and make the list partial ordered,
    and the recursion process make sure the whole list is ordered.
    for a N-size array, top-bottom merge sort need 1/2NlgN to NlgN comparisons,
    and need to access array 6NlgN times at most.
    >>> ms = MergeSort()
    >>> lst = [4, 3, 2, 5, 7, 9, 0, 1, 8, 7, -1, 11, 13, 31, 24]
    >>> ms.sort(lst)
    >>> lst
    [-1, 0, 1, 2, 3, 4, 5, 7, 7, 8, 9, 11, 13, 24, 31]
    """

    def __merge(self, lst, lo, mid, hi):
        left = lst[lo:mid + 1]
        right = lst[mid + 1: hi + 1]

        if left[-1] < right[0]:
            return

        i, j, k = 0, 0, lo
        m, n = len(left), len(right)

        while i < m and j < n:
            if left[i] < right[j]:
                lst[k] = left[i]
                i += 1
            else:
                lst[k] = right[j]
                j += 1

            k += 1

        while i < m:
            lst[k] = left[i]
            i += 1
            k += 1

        while j < n:
            lst[k] = right[j]
            j += 1
            k += 1

    def __sort(self, lst, lo, hi):
        if hi <= lo:
            return

        mid = lo + ((hi - lo) // 2)
        self.__sort(lst, lo, mid)
        self.__sort(lst, mid + 1, hi)
        self.__merge(lst, lo, mid, hi)

    def sort(self, lst):
        self.__sort(lst, 0, len(lst) - 1)

class LinkedListMerged(object):
    """
        Linked List Merge Sort, break linked list in two peices and merge sort them
    """
    def __merge(self, list1: Node, list2: Node):
        if not list1 or not list2:
            return list1 or list2

        head = pointer = None

        if list1 <= list2:
            head = pointer = list1
            list1 = list1.next_node
        else:
            head = pointer = list2
            list2 = list2.next_node

        while list1 and list2:
            if list1 <= list2:
                pointer.next_node = list1
                list1 = list1.next_node
            else:
                pointer.next_node = list2
                list2.next_node = list2
        
        return head
        
    def __sort(self, head: Node):
        if not head or not head.next_node:
            return

        slow, fast = head, head
        while fast.next_node and fast.next_node.next_node:
            slow = slow.next_node
            fast = fast.next_node.next_node
        
        self.__sort(slow)
        self.__sort(fast)
        return self.__merge(slow, fast)

    def sort(self, head: Node):
        self.__sort(head)
# 2.2.19 practice, using merge function from merge-sort to count the reverse number
class ReverseCount(object):

    """
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

if __name__ == '__main__':
    doctest.testmod()
