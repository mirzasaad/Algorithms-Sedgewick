from itertools import count
from turtle import right
from module_1_1 import binary_search
import doctest


def two_sum_fast(lst):
    """
      Count the number of pair of numbers add up to zero. first sort the list,
    then use binary_search the get the other number which could add up to zero,
    if in the list, then increase the counter.
    >>> lst = [-1, 1, -2, 3, 5, -5, 0, 4]
    >>> two_sum_fast(lst)
    2
    """
    lst.sort()
    count = 0
    for i in range(len(lst)):
        if binary_search(-lst[i], lst) > i:
            count += 1
    return count

def two_sum_with_target(lst, target):
    """
        Get the indices of the list which
    two elements add up to specific target. Can not use the same
    element twice.
        Using dictionary to mark the indice of the number, if
    target - number in the dictionary, return the indice.
    >>> lst = [2, 7, 11, 15]
    >>> two_sum_with_target(lst, 9)
    (0, 1)
    >>> lst2 = [3, 3]
    >>> two_sum_with_target(lst2, 6)
    (0, 1)
    >>> lst3 = [3, 2, 4, 1]
    >>> two_sum_with_target(lst3, 6)
    (1, 2)
    """
    num_indexes = {}
    for index, value in enumerate(lst):
        if value in num_indexes:
            return (num_indexes[value], index)
        num_indexes[target - value] = index

def three_sum_fast(lst):
    """
      Count how many three numbers add up to zero. first sort the list,
    then using two for-loop and binary search algorithm get the
    opposite number.
    >>> lst = [-1, 2, 1, 3, 0, 4, -4, 5, 9, -5]
    >>> three_sum_fast(lst)
    8
    """
    lst.sort()
    count = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if binary_search(-lst[i] - lst[j], lst) > j:
                count += 1
    return count

# 1.4.14 practice
def four_sum_fast(lst):
    lst.sort()
    index = set()
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            index.add((i, j, lst[i] + lst[j]))

# 1.4.16 practice
def closest_pair(lst):
    """
      Get two closest number in a list, first sort the list,
    then iterate through the list compare each
    summation of two adjacent numbers in the list,
    then get the result.
    >>> lst = [1, 0, 3, 4, 5, 9, 1]
    >>> closest_pair(lst)
    (1, 1)
    >>> lst
    [0, 1, 1, 3, 4, 5, 9]
    """
    lst.sort()
    a, b, max_value = None, None, (1 << 63) - 1

    for i in range(len(lst) - 1):
        if lst[i + 1] - lst[i] < max_value:
            max_value = lst[i + 1] - lst[i]
            a, b = lst[i], lst[i + 1]
    
    return a, b

# 1.4.17 practice
def farthest_pair(lst):
    return min(lst), max(lst)

def partial_minimum(lst):
    """
      Find the partial minimum number in the list,
    the whole process is similar to binary search algorithm.
    >>> lst = [5, 2, 3, 4, 3, 5, 6, 8, 7, 1, 9]
    >>> partial_minimum(lst)
    2
    """
    start, end = 0, len(lst) - 1
    while start <= end:
        mid = int((end + start) / 2)
        left, right = mid - 1, mid + 1
        midValue, leftValue, rightValue = lst[mid], lst[left], lst[right]

        if midValue <= leftValue and midValue <= rightValue:
            return midValue
        elif leftValue < midValue and mid - 1 >= start:
            end = mid - 1
        elif rightValue < midValue and mid + 1 <= end:
            start = mid + 1

    return  leftValue if leftValue < rightValue else rightValue

# 1.4.20 practice
def bitonic_list_search(key, lst):
    """
    >>> lst = [1, 2, 3, 9, 8, 7, 6, 5, 4, -1]
    >>> bitonic_list_search(2, lst)
    1
    >>> bitonic_list_search(9, lst)
    3
    >>> bitonic_list_search(7, lst)
    5
    """
    def find_the_point(lst):
        low, high = 0, len(lst) - 1
        while low < high:
            mid = int((low + high) / 2)
            if lst[mid] < lst[mid + 1]:
                low = mid + 1
            elif lst[mid] > lst[mid + 1]:
                high = mid
        return low

    
    def find_left(key, start, end, lst):
        while start <= end:
            mid = int((start + end) / 2)
            if lst[mid] < key:
                start = mid + 1
            elif lst[mid] > key:
                end = mid - 1
            else:
                return mid
        return -1

    def find_right(key, start, end, lst):
        while start <= end:
            mid = int((start + end) / 2)
            if lst[mid] < key:
                end = mid - 1
            elif lst[mid] > key:
                start = mid + 1
            else:
                return mid
        return -1
    
    index = find_the_point(lst)
    if key == lst[index]:
        return index
    right = find_right(key, index, len(lst) - 1, lst)
    left = find_left(key, 0, index, lst)

    return left if left > -1 else right

if __name__ == '__main__':
    doctest.testmod()
