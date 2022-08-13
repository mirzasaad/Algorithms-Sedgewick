
from cgitb import reset
import collections
from distutils.command.build import build
import enum
from hashlib import sha256
import heapq
from multiprocessing import dummy
from os import curdir, pread
import queue
import re
import string
from turtle import right, rt
from typing import List, Optional

# https://neetcode.io/


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self) -> str:
        return 'ListNode({} -> {})'.format(self.val, self.next)


class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return 'TreeNode({})'.format(self.val)
    # def __repr__(self) -> str:
    #     return 'TreeNode({} <- {} -> {})'.format(self.left, self.val, self.right)


def containsDuplicate_2(nums: List[int]) -> bool:
    """
    Question 1

    Contains Duplicate

    Given an integer array nums, return true if any value appears at least 
    twice in the array, and return false if every element is distinct.
    """
    nums.sort()

    for i in range(len(nums) - 1):
        if nums[i] == nums[i + 1]:
            return True

    return False


def containsDuplicate(nums: List[int]) -> bool:
    """
    Question 1

    Contains Duplicate

    Given an integer array nums, return true if any value appears at least 
    twice in the array, and return false if every element is distinct.
    """
    _dict = dict()
    for i in range(len(nums)):
        if nums[i] in _dict:
            return True
        else:
            _dict[nums[i]] = True

    return False


def containsDuplicateFast(nums: List[int]) -> bool:
    return len(set(nums)) == len(nums)


def isAnagram(s: str, t: str) -> bool:
    """
    Question 2

    Valid Anagram

    Given two strings s and t, return true if t is an anagram of s, and false otherwise.
    An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, 
    typically using all the original letters exactly once.
    """
    if len(s) != len(t):
        return False

    s.sort()
    t.sort()

    for i in range(len(s) - 1):
        if s[i] != t[i]:
            return False

    return True


def isAnagram(s: str, t: str):
    """
    Question 2

    Valid Anagram

    Given two strings s and t, return true if t is an anagram of s, and false otherwise.
    An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, 
    typically using all the original letters exactly once.
    """
    return all([s.count(charactar) == t.count(charactar) for charactar in string.ascii_lowercase])


def twoSum(nums: List[int], target: int) -> List[int]:
    """
    Question 3

    Two Sum

    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    You may assume that each input would have exactly one solution, and you may not use the same element twice.
    You can return the answer in any order.
    """

    table = dict()
    for idx, num in enumerate(nums):
        table[target - num] = idx

    for idx, num in enumerate(nums):
        if num in table and table[num] != idx:
            return [table[num], idx]

    return []


def twoSumFast(nums: List[int], target: int) -> List[int]:
    """
    Question 3

    Two Sum

    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    You may assume that each input would have exactly one solution, and you may not use the same element twice.
    You can return the answer in any order.
    """

    table = dict()
    for idx, num in enumerate(nums):
        value = target - num

        if value in table:
            return [table[value], idx]

        table[num] = idx

    return []


def groupAnagrams(strs: List[str]) -> List[List[str]]:
    """
    Question 4

    Group Anagrams

    Given an array of strings strs, group the anagrams together. 
    You can return the answer in any order.
    An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, 
    typically using all the original letters exactly once.
    """

    table = {}

    for _, str in enumerate(strs):
        _str = ''.join(sorted(str))
        if _str in table:
            table[_str].append(str)
        else:
            table[_str] = [str]

    results = []

    for i in table:
        results.append(table[i])

    return results


def groupAnagramsFast(strs: List[str]) -> List[List[str]]:
    """
    Question 4

    Group Anagrams

    Given an array of strings strs, group the anagrams together. 
    You can return the answer in any order.
    An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, 
    typically using all the original letters exactly once.
    """
    look = collections.defaultdict(list)

    for s in strs:
        look[tuple(sorted(s))].append(s)
    return look.values()


def topKFrequent(nums: List[int], k: int) -> List[int]:
    """
    Question 5

    Top K Frequent Elements

    Given an integer array nums and an integer k, 
    return the k most frequent elements. 
    You may return the answer in any order.
    """
    table = {}

    for num in nums:
        if num in table:
            table[num] += 1
        else:
            table[num] = 1

    result = []

    for key in table.keys():
        if table[key] >= k:
            result.append(key)

    return result


def topKFrequent_2(nums: List[int], k: int) -> List[int]:
    """
    Question 5

    Top K Frequent Elements

    Given an integer array nums and an integer k, 
    return the k most frequent elements. 
    You may return the answer in any order.
    """
    counter = dict(collections.Counter(nums))

    length = len(nums)
    feq = collections.defaultdict(list)

    for key, count in counter.items():
        feq[count].append(key)

    result = []

    for i in reversed(range(length + 1)):
        if i in feq:
            result.extend(feq[i])
        if len(result) == k:
            break

    return result


def productExceptSelf(nums: List[int]) -> List[int]:
    """
    Question 6

    Product of Array Except Self

    Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
    The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
    You must write an algorithm that runs in O(n) time and without using the division operation.

    take prefix and siffix products and then multiply them
    """
    length = len(nums)
    pref, suff = [1] * length, [1] * length

    current = 1
    for i in range(1, length):
        pref[i] = nums[i - 1] * current
        current = nums[i - 1] * current

    current = 1
    for j in reversed(range(0, length - 1)):
        suff[j] = nums[j + 1] * current
        current = nums[j + 1] * current

    prod = [-1] * length

    for i in range(length):
        prod[i] = pref[i] * suff[i]

    return prod


def encode(strs: List[str]) -> str:
    """
    Question 7

    Encode and Decode Strings

    Design an algorithm to encode a list of strings to a string. 
    The encoded string is then sent over the network and is 
    decoded back to the original list of strings.
    """

    encoded = ""

    for st in strs:
        encoded += '{}:{}'.format(len(st), st)

    return encoded


def decode(encoded: str) -> List[str]:
    """
    Question 7

    Encode and Decode Strings

    Design an algorithm to encode a list of strings to a string. 
    The encoded string is then sent over the network and is 
    decoded back to the original list of strings.
    """

    index = 0
    N = len(encoded)
    result = []
    while index < N:
        count = ''

        while encoded[index] != ':':
            count += encoded[index]
            index += 1

        index += 1
        count = int(count)

        j = index
        new_str = ''
        while j < index + count:
            new_str += encoded[j]
            j += 1

        index += count
        result.append(new_str)

    return result


def longestConsecutive(nums: List[int]) -> int:
    """
    Question 8

    Given an unsorted array of integers nums, 
    return the length of the longest consecutive elements sequence.
    You must write an algorithm that runs in O(n) time.
    """

    nums.sort()

    i, N = 0, len(nums)
    max_count = 0
    current = 0
    for i in range(N - 1):
        if nums[i + 1] - nums[i] == 1:
            current += 1
        else:
            current = 0

        max_count = max(current + 1, max_count)

    return max_count


def longestConsecutiveFast(nums: List[int]) -> int:
    """
    Question 8

    Given an unsorted array of integers nums, 
    return the length of the longest consecutive elements sequence.
    You must write an algorithm that runs in O(n) time.
    """

    table = set(nums)
    max_count = 0

    for num in table:
        if num + 1 not in table:
            curr_num = num
            count = 1

            while curr_num - 1 in table:
                count += 1
                curr_num -= 1

            max_count = max(max_count, count)

    return max_count


def isPalindrome(s: str) -> bool:
    """
    Question 9

    Valid Palindrome

    A phrase is a palindrome if, after converting all uppercase letters 
    into lowercase letters and removing all non-alphanumeric characters, 
    it reads the same forward and backward. Alphanumeric characters include letters and numbers.
    Given a string s, return true if it is a palindrome, or false otherwise.
    """
    if not s:
        return False

    i, j = 0, len(s) - 1

    while i < j:
        while i < j and not s[i].isalnum():
            i += 1

        while j > i and not s[j].isalnum():
            j -= 1

        if i == j:
            break

        if s[i].lower() != s[j].lower():
            return False

        i += 1
        j -= 1

    return True


def binary_search(lst, key):
    lo, hi = 0, len(lst) - 1

    while lo <= hi:
        mid = int((hi + lo) / 2)
        if lst[mid] == key:
            return mid
        elif key < lst[mid]:
            hi = mid - 1
        else:
            lo = mid + 1

    return -1


def threeSumFast(nums: List[int]) -> List[List[int]]:
    """
    Question 10

    3Sum

    Given an integer array nums, return all the triplets 
    [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
    Notice that the solution set must not contain duplicate triplets.
    """
    N = len(nums)
    results = []

    for i in range(N):
        for j in range(i + 1, N):
            k = binary_search(nums, -nums[i] - nums[j])
            if k > j:
                results.append([i, j, k])

    return results


def threeSum(nums: List[int]) -> List[List[int]]:
    """
    Question 10

    3Sum

    Given an integer array nums, return all the triplets 
    [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
    Notice that the solution set must not contain duplicate triplets.
    """
    N = len(nums)
    results = []

    nums.sort()

    for i in range(N - 2):

        if i > 0 and nums[i] == nums[i - 1]:
            # remove first pointer duplicates
            continue

        start = i + 1
        end = N - 1
        a = nums[i]

        while start < end:
            b, c = nums[start], nums[end]

            if a + b + c == 0:
                results.append([a, b, c])

                # remove duplicates
                while start < end and nums[start] == nums[start + 1]:
                    start += 1
                while end > start and nums[end] == nums[end - 1]:
                    end -= 1

                start += 1
                end -= 1
            elif a + b + c < 0:
                start += 1
            else:
                end -= 1

    return results


def maxArea(height: List[int]) -> int:
    """
    Question 11

    Container With Most Water

    You are given an integer array height of length n. 
    There are n vertical lines drawn such that the two endpoints of the 
    ith line are (i, 0) and (i, height[i]).
    Find two lines that together with the x-axis form a container, 
    such that the container contains the most water.
    Return the maximum amount of water a container can store.
    """

    max_area = 0

    i, j = 0, len(height) - 1

    while i < j:
        distance = j - i
        __height = min(height[i], height[j])
        area = distance * __height

        if height[i] < height[j]:
            i += 1
        else:
            j -= 1

        max_area = max(max_area, area)

    return max_area


def maxProfit(prices: List[int]) -> int:
    """
    Question 12

    Best Time to Buy and Sell Stock

    You are given an array prices where prices[i] is the 
    price of a given stock on the ith day.
    You want to maximize your profit by choosing a single day 
    to buy one stock and choosing a different day 
    in the future to sell that stock.
    Return the maximum profit you can achieve from this transaction. 
    If you cannot achieve any profit, return 0.
    """

    max_profit = 0
    min_so_far = (1 << 63) - 1

    for price in prices:
        min_so_far = min(min_so_far, price)
        max_profit = max(max_profit, price - min_so_far)

    return max_profit


def lengthOfLongestSubstring(s: str) -> int:
    """
    Question 13

    Longest Substring Without Repeating Characters

    Given a string s, find the length of the longest substring without repeating characters.
    """

    start = 0
    end = 0
    duplciate = False
    count = collections.Counter()
    max_seq = 0

    for end, ch in enumerate(s):
        count[ch] += 1
        if count[ch] > 1:
            duplciate = True

        while duplciate:
            temp = s[start]

            if count[temp] > 1:
                duplciate = False

            count[temp] -= 1
            start += 1

        max_seq = max(max_seq, end - start + 1)

    return max_seq


count, start, length = collections.Counter(), 0, 0


def characterReplacement_2(s: str, k: int) -> int:
    """
    Question 14

    Longest Substring Without Repeating Characters

    Given a string s, find the length of the longest substring without repeating characters.
    """
    count, start, length = [0] * 26, 0, 0
    most_common_count = 0

    for end, ch in enumerate(s):
        count[ch - 'A'] += 1
        most_common_count = max(count[ch - 'A'], most_common_count)
        distance = end - start + 1

        while distance - most_common_count > k:
            count[s[start] - 'A'] -= 1
            start += 1

        length = max(length, end - start + 1)


def characterReplacement(s: str, k: int) -> int:
    """
    Question 14

    Longest Substring Without Repeating Characters

    Given a string s, find the length of the longest substring without repeating characters.
    """
    count, start, length = collections.Counter(), 0, 0

    for end, st in enumerate(s):
        count[st] += 1
        most_common = count.most_common(1)[0][1]
        distance = end - start + 1

        while distance - most_common > k:
            distance = end - start + 1
            count[s[start]] -= 1
            start += 1

        length = max(length, end - start + 1)

    return length


def minWindow(text: str, pattern: str) -> str:
    """
    Question 15

    Minimum Window Substring

    Given two strings s and t of lengths m and n respectively, 
    return the minimum window substring of s such that every character in t (including duplicates) is 
    included in the window. If there is no such substring, return the empty string "".
    The testcases will be generated such that the answer is unique.
    A substring is a contiguous sequence of characters within the string.
    """
    if len(pattern) > len(text):
        return ''

    Index = collections.namedtuple('Index', 'start length')

    start, end, dict = 0, 0, collections.Counter(pattern)
    substr = Index(0, 1 << 31)
    count = len(dict)

    while end < len(text):
        char = text[end]
        if char in dict:
            dict[char] -= 1
            if dict[char] == 0:
                count -= 1

        end += 1

        while count == 0:
            temp_char = text[start]
            if temp_char in dict:
                dict[temp_char] += 1

                if dict[temp_char] > 0:
                    count += 1

            distance = end - start

            if substr.length > distance:
                substr = Index(start, distance)

            start += 1

    if substr.length == 1 << 31:
        return ''

    return text[substr.start:substr.start+substr.length]


def isValid(s: str) -> bool:
    """
    Question 16

    Valid Parentheses

    Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
    An input string is valid if:
    Open brackets must be closed by the same type of brackets.
    Open brackets must be closed in the correct order.
    """
    stack = []
    lookup = {'{': '}', '[': ']', '(': ')'}

    for bracket in s:
        if bracket in lookup:
            stack.append(s)
        elif not stack or lookup[stack.pop()] != bracket:
            return False

    return True


def searchFast(nums: List[int], target: int) -> int:
    """
    Question 17

    Search in Rotated Sorted Array

    There is an integer array nums sorted in ascending order (with distinct values).
    Prior to being passed to your function, nums is possibly rotated at an unknown pivot 
    index k (1 <= k < nums.length) such that the resulting array is 
    [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
    For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
    Given the array nums after the possible rotation and an integer target,
     return the index of target if it is in nums, or -1 if it is not in nums.
    You must write an algorithm with O(log n) runtime complexity.
    """
    N = len(nums)
    lo, hi = 0, N - 1

    while lo < hi:
        mid = lo + ((hi-lo)//2)
        if nums[mid] > nums[hi]:
            lo = mid + 1
        else:
            hi = mid

    rotation, lo, hi = lo, 0, N-1

    while lo <= hi:
        mid = lo + ((hi-lo)//2)
        realmid = (mid+rotation) % N
        if nums[realmid] == target:
            return realmid
        if target > nums[realmid]:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def search(nums: List[int], target: int) -> int:
    """
    Question 17

    Search in Rotated Sorted Array

    There is an integer array nums sorted in ascending order (with distinct values).
    Prior to being passed to your function, nums is possibly rotated at an unknown pivot 
    index k (1 <= k < nums.length) such that the resulting array is 
    [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
    For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
    Given the array nums after the possible rotation and an integer target,
     return the index of target if it is in nums, or -1 if it is not in nums.
    You must write an algorithm with O(log n) runtime complexity.
    """

    def findPivot(nums):
        lo, hi = 0, len(nums) - 1

        while lo < hi:
            mid = (lo + hi) // 2

            if mid < hi and nums[mid] > nums[mid + 1]:
                return mid + 1
            if mid < lo and nums[mid] < nums[mid - 1]:
                return mid

            if nums[lo] >= nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1

        return 0

    N = len(nums)
    rotation = findPivot(nums)
    lo, hi = 0, N - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        realmid = (mid + rotation) % N

        if nums[realmid] == target:
            return realmid
        elif target < nums[realmid]:
            hi = mid - 1
        else:
            lo = mid + 1

    return -1


def findMin(nums: List[int]) -> int:
    """
    Question 18

    Find Minimum in Rotated Sorted Array

    Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:
    [4,5,6,7,0,1,2] if it was rotated 4 times.
    [0,1,2,4,5,6,7] if it was rotated 7 times.
    Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].
    Given the sorted rotated array nums of unique elements, return the minimum element of this array.
    You must write an algorithm that runs in O(log n) time.
    """

    def findPivot():
        lo, hi = 0, len(nums) - 1

        while lo < hi:
            mid = (lo + hi) // 2

            if mid < hi and nums[mid] > nums[mid + 1]:
                return mid + 1

            if lo < mid and nums[mid] < nums[mid - 1]:
                return mid

            if nums[lo] >= nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1

        return 0

    pivot = findPivot()

    return nums[pivot] if pivot < len(nums) else None


def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Question 19

    Reverse Linked List

    Given the head of a singly linked list, reverse the list, and return the reversed list.
    """
    if not head:
        return head

    current = head
    prev_node = None

    while current:
        second_node = current.next
        current.next = prev_node
        prev_node = current
        current = second_node

    return prev_node


def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Question 20

    Merge Two Sorted Lists

    You are given the heads of two sorted linked lists list1 and list2.
    Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.
    Return the head of the merged linked list.
    """

    dummy = ListNode(0)
    head = dummy

    while list1 and list2:
        if list1.val <= list2.val:
            dummy.next = ListNode(list1.val)
            list1 = list1.next
        elif list2.val < list1.val:
            dummy.next = ListNode(list2.val)
            list2 = list2.next
        dummy = dummy.next

    longer_list = list1 or list2

    while longer_list:
        dummy.next = ListNode(longer_list.val)
        longer_list = longer_list.next
        dummy = dummy.next

    return head.next


def reorderList(head: Optional[ListNode]) -> None:
    """
    Question 21

    Reorder List

    You are given the head of a singly linked-list. The list can be represented as:
    L0 → L1 → … → Ln - 1 → Ln
    Reorder the list to be on the following form:
    L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
    """

    slow, fast = head, head

    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    prev, current = None, slow.next

    while current:
        second = current.next
        current.next = prev
        prev = current
        current = second

    slow.next = None
    head1, head2 = head, prev

    while head2:
        second_head_1 = head1.next
        head1.next = head2
        head1 = second_head_1

        second_head_2 = head2.next
        head2.next = head1
        head2 = second_head_2


def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Question 22

    Remove Nth Node From End of List

    Given the head of a linked list, remove the nth node from the end of the list and return its head.
    """
    if not n:
        return head

    slow, fast = head, head

    for _ in range(n):
        fast = fast.next

    if not fast:
        return head.next

    while fast.next:
        slow = slow.next
        fast = fast.next

    slow.next = slow.next.next

    return head


def hasCycle(head: Optional[ListNode]) -> bool:
    """
    Question 23

    Linked List Cycle

    Given head, the head of a linked list, determine if the linked list has a cycle in it.
    There is a cycle in a linked list if there is some node in the list that can be reached
    again by continuously following the next pointer. Internally, pos is used to denote the index of the 
    node that tail's next pointer is connected to. Note that pos is not passed as a parameter.
    Return true if there is a cycle in the linked list. Otherwise, return false.
    """
    slow, fast = head, head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow is fast:
            return True
    return False


def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Question 24

    Input: lists = [[1,4,5],[1,3,4],[2,6]]
    Output: [1,1,2,3,4,4,5,6]
    Explanation: The linked-lists are:
    [
    1->4->5,
    1->3->4,
    2->6
    ]
    merging them into one sorted list:
    1->1->2->3->4->4->5->6
    """

    counter = collections.Counter()
    combined = ListNode(-1)
    combined_head = combined

    min_node = collections.namedtuple('smallest', 'index node')

    while len(counter) != len(lists):
        smallest = None
        for index, head in enumerate(lists):
            if not head:
                counter[index] = 1
                continue
            if not smallest or smallest.node.val >= head.val:
                smallest = min_node(index, head)

        if smallest:
            combined.next = ListNode(smallest.node.val)
            combined = combined.next
            lists[smallest.index] = lists[smallest.index].next

    return combined_head.next


def mergeKListsFast(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Question 24

    Input: lists = [[1,4,5],[1,3,4],[2,6]]
    Output: [1,1,2,3,4,4,5,6]
    Explanation: The linked-lists are:
    [
    1->4->5,
    1->3->4,
    2->6
    ]
    merging them into one sorted list:
    1->1->2->3->4->4->5->6
    """
    queue = []
    head = current = ListNode(-1)
    count = 0
    # using count for tie breaker

    for linked_list in lists:
        if linked_list:
            count += 1
            heapq.heappush(queue, (linked_list.val, count, linked_list))

    while queue:
        _, _, current.next = heapq.heappop(queue)
        current = current.next

        if current.next:
            count += 1
            heapq.heappush(queue, (current.next.val, count, current.next))

    return head.next


def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Question 25

    Invert Binary Tree

    Given the root of a binary tree, invert the tree, and return its root.
    """

    def __invertTree(node: Optional[TreeNode]) -> Optional[TreeNode]:
        if not node:
            return node

        right = __invertTree(node.left)
        left = __invertTree(node.right)

        node.left = right
        node.right = left

        return node

    return __invertTree(root)


def maxDepth(root: Optional[TreeNode]) -> int:
    """
    Question 26

    Maximum Depth of Binary Tree

    Given the root of a binary tree, return its maximum depth.
    A binary tree's maximum depth is the number of nodes along the 
    longest path from the root node down to the farthest leaf node.
    """

    def __maxDepth(node: Optional[TreeNode]) -> int:
        if not node:
            return 0

        left = __maxDepth(node.left)
        right = __maxDepth(node.right)

        return 1 + max(left, right)

    return __maxDepth(root)


def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Question 27

    Same Tree

    Given the roots of two binary trees p and q, write a function to check if they are the same or not.
    Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.
    """

    def __isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> int:
        if not p and not q:
            return True

        if p or q:
            return False

        if p.val != q.val:
            return False

        left = __isSameTree(p.left, q.left)
        right = __isSameTree(p.right, q.right)

        return left and right

    return __isSameTree(p, q)


def isSubtree(root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    """
    Question 28

    Subtree of Another Tree

    Given the roots of two binary trees root and subRoot, return true if there is a 
    subtree of root with the same structure and node values of subRoot and false otherwise.
    A subtree of a binary tree tree is a tree that consists of a node in tree and all of 
    this node's descendants. The tree tree could also be considered as a subtree of itself.
    """
    def __isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> int:
        if not p and not q:
            return True

        if p or q:
            return False

        if p.val != q.val:
            return False

        left = __isSameTree(p.left, q.left)
        right = __isSameTree(p.right, q.right)

        return left and right

    def dfs(p: Optional[TreeNode], q: Optional[TreeNode]):
        if not p:
            return False

        if p.val == q.val and __isSameTree(p, q):
            return True

        return dfs(p.left, q) or dfs(p.right, q)

    return dfs(root, subRoot)


def isSubtreeFast(root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    """
    Question 28

    Subtree of Another Tree

    Given the roots of two binary trees root and subRoot, return true if there is a 
    subtree of root with the same structure and node values of subRoot and false otherwise.
    A subtree of a binary tree tree is a tree that consists of a node in tree and all of 
    this node's descendants. The tree tree could also be considered as a subtree of itself.
    """
    def __hash(node: TreeNode):
        s = sha256()
        s.update(str(node.val))
        return s.digest()

    def __merkel(node: Optional[TreeNode]):
        if not node:
            return '#'

        left = __merkel(node.left)
        right = __merkel(node.right)

        node.merkel = left + __hash(node) + right

        return node.merkel

    __merkel(root)
    __merkel(subRoot)

    def dfs(root: Optional[TreeNode], subRoot: Optional[TreeNode]):
        if not root:
            return False

        isEqual = root.merkel == subRoot.merkel

        return isEqual or dfs(root.left) or dfs(root.right)

    return dfs(root, subRoot)


def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """
    Question 29

    Lowest Common Ancestor of a Binary Search Tree

    Given a binary search tree (BST), find the lowest common ancestor (LCA) 
    node of two given nodes in the BST.
    According to the definition of LCA on Wikipedia: 
    “The lowest common ancestor is defined between two nodes p and q as the lowest 
    node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
    """

    CommonNode = collections.namedtuple('CommonNode', 'node found')

    def __lca(node: Optional[TreeNode], p: Optional[TreeNode], q: Optional[TreeNode]):
        if not node:
            return CommonNode(None, False)

        if node.val == p.val or node.val == q.val:
            return CommonNode(node.val, True)

        left = __lca(node.left, p, q)
        right = __lca(node.right, p, q)

        if left.found and right.found:
            return CommonNode(node.val, True)

        found = left.found or right.node
        __node = None if not found else (
            left.node if left.found else right.node)

        return CommonNode(__node, found)

    return __lca(root, p, q)


def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """
    Question 29

    Lowest Common Ancestor of a Binary Search Tree

    Given a binary search tree (BST), find the lowest common ancestor (LCA) 
    node of two given nodes in the BST.
    According to the definition of LCA on Wikipedia: 
    “The lowest common ancestor is defined between two nodes p and q as the lowest 
    node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
    """

    while root:
        if root.val > p.val and root.val > q.val:
            root = root.left
        elif root.val < p.val and root.val < q.val:
            root = root.right
        else:
            return root

    return None


def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Question no 30

    Binary Tree Level Order Traversal

    Given the root of a binary tree, return the level order 
    traversal of its nodes' values. (i.e., from left to right, level by level).
    """

    level_order = [root]
    result = []

    while root and level_order:
        result.append([node.val for node in level_order if node])
        level_order = [child for node in level_order for child in (
            node.left, node.right) if child]

    return result


def isValidBST(root: Optional[TreeNode]) -> bool:
    """
    Question 31

    Validate Binary Search Tree

    Given the root of a binary tree, determine if it is a valid binary search tree (BST).
    A valid BST is defined as follows

    The left subtree of a node contains only nodes with keys less than the node's key.
    The right subtree of a node contains only nodes with keys greater than the node's key.
    Both the left and right subtrees must also be binary search trees.
    """
    def __isValidBST(node: Optional[TreeNode], min, max):
        if not node:
            return True

        if node.left <= min or node.right >= max:
            return False

        left = __isValidBST(node.left, min, node.val)
        right = __isValidBST(node.right, node.val, max)

        return left and right

    return __isValidBST(root, float('-inf'), float('inf'))


def kthSmallestRecursive(root: Optional[TreeNode], k: int) -> int:
    """
    Question 32

    Kth Smallest Element in a BST

    Given the root of a binary search tree, and an integer k, 
    return the kth smallest value (1-indexed) of all the values of the nodes in the tree.
    """

    result = None

    def kthSmallest(node: Optional[TreeNode]):
        if not node:
            return

        kthSmallest(node.left)
        k -= 1
        if k == 0:
            result = node
            return
        kthSmallest(node.right)

    kthSmallest(root)

    return result


def kthSmallestIterative(root: Optional[TreeNode], k: int) -> int:
    """
    Question 33

    Kth Smallest Element in a BST

    Given the root of a binary search tree, and an integer k, 
    return the kth smallest value (1-indexed) of all the values of the nodes in the tree.
    """

    stack = []
    while stack or root:

        while root:
            stack.append(root)
            root = root.left

        root = root.pop()

        k -= 1
        if k == 0:
            return root

        root = root.right

    return reset


def kthSmallestMorris(root: Optional[TreeNode], k: int) -> int:
    """
    Question 34

    Kth Smallest Element in a BST

    Given the root of a binary search tree, and an integer k, 
    return the kth smallest value (1-indexed) of all the values of the nodes in the tree.
    """

    if k == 0:
        return None

    current = root
    while current:
        if not current.left:
            k -= 1
            if k == 0:
                return current.val
            current = current.right
        else:
            pre = current.left

            while pre.right and pre.right is not current:
                pre = pre.right

            if pre.right is current:
                k -= 1
                if k == 0:
                    return current.val

                pre.right = None
                current = current.right
            else:
                pre.right = current
                current = current.left

    return None


def buildTree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Question 35

    Construct Binary Tree from Preorder and Inorder Traversal

    Given two integer arrays preorder and inorder where preorder is the preorder 
    traversal of a binary tree and inorder is the inorder traversal of the same tree, 
    construct and return the binary tree.
    """
    current = iter(preorder)
    inorder_map = {}

    for index, value in enumerate(inorder):
        inorder_map[value] = index

    def __build(lo, hi):
        if lo > hi:
            return None

        node = TreeNode(next(current))
        mid = inorder_map[node.val]
        node.left = __build(lo, mid - 1)
        node.right = __build(mid + 1, hi)

        return node

    return __build(0, len(inorder) - 1)


def maxPathSum(root: Optional[TreeNode]) -> int:
    """
    Question 36

    Binary Tree Maximum Path Sum

    A path in a binary tree is a sequence of nodes where each pair of 
    adjacent nodes in the sequence has an edge connecting them. A node can only 
    appear in the sequence at most once. Note that the path does not need to pass through the root.
    The path sum of a path is the sum of the node's values in the path.
    Given the root of a binary tree, return the maximum path sum of any non-empty path.
    """
    result = [root.val]

    def __maxPathSum(node: Optional[TreeNode]):
        if not node:
            return 0

        left = __maxPathSum(node.left)
        right = __maxPathSum(node.right)

        left_max = max(0, left)
        right_max = max(0, right)

        max_without_split = node.val + left_max + right_max
        max_node = node.val
        max_node_plus_left = node.val + left_max
        max_node_plus_right = node.val + right_max

        result[0] = max(result[0], max_without_split, max_node,
                        max_node_plus_left, max_node_plus_right)

        max_with_split = node.val + max(left_max, right_max)

        return max_with_split

    __maxPathSum(root)

    return result[-1]


class Codec:

    def __init__(self) -> None:
        self.__NULL = 'x'
        self.__seperator = ','

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        preorder = []
        
        def dfs(node):
            if not node:
                preorder.append(self.__NULL)
                return

            preorder.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return self.__seperator.join(preorder)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        preorder = collections.deque(data.split(self.__seperator))

        def buildTree():
            if not preorder: return

            item = preorder.popleft()
            
            if item == self.__NULL:
                return None

            node = TreeNode(int(item))
            node.left = buildTree()
            node.right = buildTree()

            return node

        root = buildTree()
        return root
