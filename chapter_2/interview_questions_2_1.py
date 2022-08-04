from cProfile import run
import doctest
from typing import List, Dict
from collections import namedtuple
import functools
import random
from enum import IntEnum
from common import Point2d, Color

def count_intersection_points(A: List[Point2d], B: List[Point2d]):
    """Intersection of two sets. 
    Given two arrays a[] and b[], each containing nn distinct 2D points in the plane, 
    design a subquadratic algorithm to count the number of points that are contained 
    both in array a[] and array }b[].
    >>> A = [Point2d(0, 1), Point2d(0, 2), Point2d(0, 3)]
    >>> B = [Point2d(1, 0), Point2d(2, 3), Point2d(4, 5), Point2d(0, 1)]
    >>> random.shuffle(A)
    >>> random.shuffle(B)
    >>> count_intersection_points(A, B)
    3
    """

    A.sort()
    B.sort()
    i, j = 0, 0
    m, n = len(A), len(B)
    count = 0
    while i < m and j < n:
        if A[i] == A[j]:
            count += 1
            i += 1
            j += 1
        elif A[i] < A[j]:
            i += 1
        elif A[j] < A[i]:
            j += 1

    return count


def is_perm(A, B):
    """
        Permutation. Given two integer arrays of size N, 
    design a subquadratic algorithm to determine whether one is a permutation of the other. 
    That is, do they contain exactly the same entries but, possibly, in a different order.
    >>> A = [0, 1, 4]
    >>> B = [4, 1, 0]
    >>> is_perm(A, B)
    True
    """
    if len(A) != len(B):
        return False

    a = sorted(A)
    b = sorted(B)

    return a == b


def dutch_national_flag(lst: List[int]):
    """
        Dutch national flag. Given an array of nn buckets, each containing a red, white, or blue pebble, sort them by color.
        >>> lst = [2, 1, 2, 2, 1, 0, 2, 1]
        >>> dutch_national_flag(lst)
        >>> print(lst)
        [0, 1, 1, 1, 2, 2, 2, 2]
    """
    runner, left, right = 0, 0, len(lst) - 1

    while runner <= right:
        if lst[runner] - Color.White < 0:
            lst[left], lst[runner] = lst[runner], lst[left]
            left += 1
            runner += 1
        elif lst[runner] - Color.White > 0:
            lst[right], lst[runner] = lst[runner], lst[right]
            right -= 1
        else:
            runner += 1

if __name__ == '__main__':
    doctest.testmod()
