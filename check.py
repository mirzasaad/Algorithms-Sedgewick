
from collections import defaultdict
import collections
from enum import Enum
from typing import List

"""
uf
rabin karp
"""

class State(Enum):
    TO_VISIT = 0
    VISITING = 1
    VISITED = 2

def canFinish(numCourses, prerequisites):
    def build_graph():
        graph = defaultdict(list)
        for src, dest in prerequisites:
            graph[src].append(dest)
        return graph

    def dfs(v):
        if marked[v] == State.VISITING: return False
        if marked[v] == State.VISITED: return True
        
        marked[v] = State.VISITING

        for w in graph[v]:
            if not dfs(w): return False

        marked[v] = State.VISITED
        return True

    graph = build_graph()
    marked = defaultdict(bool)

    for i in range(numCourses):
        if not dfs(i):
            return False
    return True

def can(numCourses, prerequisites):
    """
    Question 45

    Course Schedule

    There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. 
    You are given an array prerequisites where prerequisites[i] = [ai, bi] 
    indicates that you must take course bi first if you want to take course ai.
    For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
    Return true if you can finish all courses. Otherwise, return false.
    """
    class State(Enum):
        TO_VISIT = 0
        VISITING = 1
        VISITED = 2

    def build_graph():
        graph = collections.defaultdict(list)
        for src, des in prerequisites:
            graph[src].append(des)

        return graph

    def dfs(vertex):
        if marked[vertex] == State.VISITING:
            return True
        if marked[vertex] == State.VISITED:
            return False

        marked[vertex] = State.VISITING

        for neighbour in graph[vertex]:
            if dfs(neighbour):
                return True

        marked[vertex] = State.VISITED
        return False

    graph = build_graph()
    marked = defaultdict(int)

    for vertex in range(numCourses):
        if dfs(vertex):
            return False
    return True


# print(canFinish(numCourses = 2, prerequisites = [[1,0]]))
# print(canFinish(numCourses = 2, prerequisites = [[1,0],[0,1]]))

# print(can(numCourses = 2, prerequisites = [[1,0]]))
# print(can(numCourses = 2, prerequisites = [[1,0],[0,1]]))


def alien_order(words: List[str]):
    graph = { character: set() for word in words for character in  word }
    
    for i in range(len(words) - 1):
        w1, w2, = words[i], words[i + 1]
        min_len = min(len(w1), len(w2))

        if len(w1) < len(w2) and w1[:min_len] == w2[:min_len]:
            return ""

        for j in range(min_len):
            if w1[j] != w1[j]:
                graph[w1[j]].add(w2[j])
                break

    def dfs(vertex):
        if visited[vertex] == State.VISITING:
            return True

        if visited[vertex] == State.VISITED:
            return False

        visited[vertex] = State.VISITED

        for neighbour in graph[vertex]:
            if dfs(neighbour):
                return True

        visited[vertex] = State.VISITED
        
        reverse_postfix.append(vertex)

        return False
    
    visited = defaultdict(int)
    reverse_postfix = []

    vertexes = list(graph.keys())

    for vertex in vertexes:
        if dfs(vertex):
            return ""

    reverse_postfix.reverse()
        
    return ''.join(reverse_postfix)


def maxProduct(nums: List[int]) -> int:
    """
    Question 56

    Maximum Product Subarray

    Given an integer array nums, find a contiguous non-empty subarray 
    within the array that has the largest product, and return the product.
    The test cases are generated so that the answer will fit in a 32-bit integer.
    A subarray is a contiguous subsequence of the array.
    """

    currMin, currMax = 1, 1
    result = 0

    for num in nums:
        candidates = [num, currMax * num, currMin * num]
        currMin = min(candidates)
        currMax = max(candidates)
        result = max(result, currMin,currMax)

    return result

def lengthOfLIS(nums: List[int]) -> int:
    """
    Question 58

    Longest Increasing Subsequence

    Given an integer array nums, return the length of the longest strictly increasing subsequence.
    A subsequence is a sequence that can be derived from an array by deleting some or no elements 
    without changing the order of the remaining elements. 
    For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].
    """

    def fn(i, prev, current):
        if i >= len(nums):
            return current

        count = 0
        skip = fn(i + 1, prev, current)
        if nums[i] > prev:
            count += fn(i + 1, nums[i], current + 1)

        return max(skip, count)

    return fn(0, float('-inf'), 0)

def longestCommonSubsequence(text1: str, text2: str) -> int:
    """
    Question 60

    Longest Common Subsequence

    Given two strings text1 and text2, return the length of their longest common subsequence. 
    If there is no common subsequence, return 0.
    A subsequence of a string is a new string generated from the original string with some characters (can be none) 
    deleted without changing the relative order of the remaining characters.
    For example, "ace" is a subsequence of "abcde".
    A common subsequence of two strings is a subsequence that is common to both strings.
    """

    def fn(i, j, current, cache={}):
        if (i, j) in cache:
            return cache[(i, j)]

        if len(text1) == i or len(text2) == j:
            return current

        sub = 0

        if text1[i] == text2[j]:
            sub += fn(i + 1, j + 1, current + 1)   
        else:
            sub += max(fn(i + 1, j, current), fn(i, j + 1, current))
        
        cache[(i, j)] = sub

        return sub   
    
    return fn(0, 0, 0)
        
def maxSubArray(nums: List[int]) -> int:
    """
    Question 61

    Maximum Subarray

    Given an integer array nums, find the contiguous subarray (containing at least one number) 
    which has the largest sum and return its sum.
    A subarray is a contiguous part of an array.
    """
    total = 0
    max_so_far = 0

    for num in nums:
        total += num
        max_so_far = max(total, max_so_far)
        total = max(total, 0)

    return max_so_far

def merge(intervals: List[List[int]]) -> List[List[int]]:
    """
    Question 64

    Merge Intervals

    Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, 
    and return an array of the non-overlapping intervals that cover all the intervals in the input.
    """
    if not intervals:
        return []

    intervals.sort(key=lambda interval: interval[0])
    current = intervals[0]
    result = []

    for _, interval in enumerate(intervals[1:]):
        if current[1] < interval[0]:
            result.append(current)
            current = interval
        elif current[0] > interval[1]:
            result.append(interval)
        else:
            current = [
                min(current[0], interval[0]),
                max(current[1], interval[1]),
            ]

    result.append(current)

    return result

def eraseOverlapIntervals(intervals: List[List[int]]) -> int:
    """
    Question 65

    Non-overlapping Intervals

    Given an array of intervals intervals where intervals[i] = [starti, endi], 
    return the minimum number of intervals you need to remove to make the 
    rest of the intervals non-overlapping.
    """
    if not intervals or len(intervals) == 1 :
        return 0

    intervals.sort(key=lambda interval: interval[0])
    count = 0
    current = intervals[0]

    for _, interval in enumerate(intervals[1:]):
        if interval == current:
            continue
        if current[1] < interval[0]:
            current = interval
        elif current[0] > interval[1]:
            continue
        else:
            current = [
                min(current[0], interval[0]),
                max(current[1], interval[1])
            ]
            count += 1
    
    return count


# print(eraseOverlapIntervals(intervals = [[1,2],[2,3],[3,4],[1,3]]))
# print(eraseOverlapIntervals(intervals = [[1,2],[1,2],[1,2]]))
# print(eraseOverlapIntervals(intervals = [[1,2],[2,3]]))
# print(merge(intervals = [[1,3],[2,6],[8,10],[15,18]]))
# print(merge(intervals = [[1,4],[4,5]]))


# print(maxSubArray(nums = [-2,1,-3,4,-1,2,1,-5,4]))

# print(longestCommonSubsequence(text1 = "abcde", text2 = "ace" ))
# print(longestCommonSubsequence(text1 = "abcde", text2 = "abce" ))
# print(longestCommonSubsequence(text1 = "ace", text2 = "ace" ))
# print(longestCommonSubsequence(text1 = "ace", text2 = "dtv" ))
# print(maxProduct([2,3,-2,4]))

def hammingWeight(n: int) -> int:
    """
    Question 71

    Number of 1 Bits

    Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

    Note:

    Note that in some languages, such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
    In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 3, the input represents the signed integer. -3.
    """

    count = 0
    while n:
        n = n & (n - 1)
        count += 1
    
    return count

def countBits(n: int) -> List[int]:
    """
    Question 72

    Counting Bits

    Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), 
    ans[i] is the number of 1's in the binary representation of i.
    """   
    result = []
    for i in range(n):
        count = 0
        n = i

        while n:
            n = n & (n - 1)
            count += 1

        result.append(count)

    return result

def reverseBits(n: int) -> int:
    """
    Question 73

    Reverse Bits

    Reverse bits of a given 32 bits unsigned integer.
    """

    result = 0

    for _ in range(32):
        result = result << 1
        if n & 1:
            result += 1
        n = n >> 1

    return result


    

print(reverseBits(43261596) == 964176192)


# print(countBits(6))
# print(hammingWeight(5))


def rabinKarp(text, pattern):
    radix = 256
    q = 997
    honer = pow(radix, len(pattern) - 1) % q

    p_hash = 0
    t_hash = 0
    m, n = len(pattern), len(text)

    for i in range(m):
        p_hash = (radix * p_hash + ord(pattern[i])) % q;
        t_hash = (radix * t_hash + ord(text[i])) % q;

    for i in range(n - m + 1):
        if p_hash == t_hash:
            return (i, i + m)
        if i < n - m:
            t_hash = (radix * (t_hash - ord(text[i]) * honer) + ord(text[i + m])) % q;

    return None

class WeightedUnionFind(object):

    """
      Weighted union find algorithm,
    put the smaller tree into the larger tree, lower the tree size.
    >>> wuf = WeightedUnionFind(10)
    >>> connections = [(4, 3), (3, 8), (6, 5), (9, 4),
    ... (2, 1), (8, 9), (5, 0), (7, 2), (6, 1), (1, 0), (6, 7)]
    >>> for i, j in connections:
    ...     wuf.union(i, j)
    ...
    >>> wuf.connected(1, 4)
    False
    >>> wuf.connected(8, 4)
    True
    >>> wuf.connected(1, 5)
    True
    >>> wuf.connected(1, 7)
    True
    """

    def __init__(self, size) -> None:
        self.id = [i for i in range(size)]
        self.weight = [1] * size
        self.count = 0

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def union(self, p, q):
        p_root = self.find(p)
        q_root = self.find(q)

        weight, id = self.weight, id

        if weight[p_root] < weight[q_root]:
            id[p_root] = id[q_root]
            weight[q_root] += weight[p_root]
        else:
            id[q_root] = id[p_root]
            weight[p_root] += weight[q_root]

        count += 1

    def find(self, node):
        root = node

        while root != self.id[root]:
            root = self.id[root]

        while node != root:
            parent = self.id[node]
            self.id[node] = root
            node = parent

        return root