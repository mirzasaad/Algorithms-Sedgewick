#!/usr/bin/env python
# -*- encoding:UTF-8 -*-
from __future__ import print_function
import re
from common import DoubleNode
import doctest


# 1.3.31 practice
class DoublyLinkedList(object):

    """
      The double-node linked list implementation
    which the node has prev and next attribute.
    >>> lst = DoublyLinkedList()
    >>> lst.push_back(1)
    >>> lst.push_front(2)
    >>> for i in lst:
    ...     print(i)
    ...
    2
    1
    >>> lst.size()
    2
    >>> lst.is_empty()
    False
    >>> lst.pop_front()
    2
    >>> lst.pop_front()
    1
    >>> lst.is_empty()
    True
    >>> lst.pop_front()
    >>> lst.push_back(1)
    >>> lst.push_back(2)
    >>> lst.pop_back()
    2
    >>> lst.pop_back()
    1
    >>> lst.pop_back()
    >>>
    >>> lst.is_empty()
    True
    >>> lst.push_back(1)
    >>> lst.insert_after(1, DoubleNode(2))
    >>> lst.insert_before(2, DoubleNode(3))
    >>> for i in lst:
    ...     print(i)
    ...
    1
    3
    2
    >>> for i in range(10):
    ...     lst.push_back(i)
    ...
    >>> lst.remove(1)
    >>> lst.remove(3)
    >>> [i for i in lst]
    [2, 0, 2, 4, 5, 6, 7, 8, 9]
    >>> lst.remove(2)
    >>> [i for i in lst]
    [0, 4, 5, 6, 7, 8, 9]
    """

    def __init__(self):
        self._head = self._tail = None
        self._size = 0

    def __iter__(self):
        tmp = self._head
        while tmp:
            yield tmp.val
            tmp = tmp.next

    def is_empty(self):
        return self._head is None

    def size(self):
        return self._size

    def push_front(self, val):
        if not self._head and not self._tail:
            self._head = self._tail = DoubleNode(val)
            self._size += 1
            return
        
        old_head = self._head
        node = DoubleNode(val)
        node.next = old_head
        old_head.prev = node
        self._head = node
        self._size += 1

    def push_back(self, val):
        if not self._head and not self._tail:
            self._head = self._tail = DoubleNode(val)
            self._size += 1
            return

        old_tail = self._tail
        node = DoubleNode(val)
        old_tail.next = node
        node.prev = old_tail
        self._tail = node
        self._size += 1

    def pop_front(self):
        if not self._head and not self._tail:
            return None

        if self._head == self._tail:
            old = self._head
            self._head = self._tail = None
            return old.val

        old = self._head;
        self._head = self._head.next
        self._head.prev = None
        old.next = None
        return old.val

    def pop_back(self):
        if not self._head and not self._tail:
            return None

        if self._head == self._tail:
            old = self._head
            self._head = self._tail = None
            return old.val
        
        old_tail = self._tail
        self._tail = self._tail.prev
        self._tail.next = old_tail.prev = None
        return old_tail.val

    def insert_before(self, target_value, new_node):
        temp = self._head
        while temp and temp.val != target_value:
            temp = temp.next

        if not temp:
            return

        if not temp.prev:
            self._head.prev = new_node
            new_node.next = self._head
            self._head = new_node
            self._size += 1
            return
        
        prev = temp.prev
        prev.next = new_node
        new_node.prev = prev

        new_node.next = temp
        temp.prev = new_node

        self._size += 1
        return

    def insert_after(self, target_value, new_node):
        temp = self._head
        while temp and temp.val != target_value:
            temp = temp.next

        if not temp:
            return

        if not temp.next:
            temp.next = new_node
            new_node.prev = temp
            self._tail = new_node
            self._size += 1

            return

        _next = temp.next
        temp.next = new_node
        new_node.next = _next

        _next.prev = new_node
        new_node.prev = temp

        self._size += 1
        return

    def remove(self, item):
        if not self._head.next and self._head.val == item:
            self._head = None
            self._size = 0
            return

        tmp = self._head
        while tmp:
            flag = False
            if tmp.val == item:
                flag = True
                if not tmp.prev:
                    target = tmp
                    tmp = tmp.next
                    tmp.prev = target.next = None
                    self._head = tmp
                else:
                    prev_node, next_node = tmp.prev, tmp.next
                    tmp.prev = tmp.next = None
                    prev_node.next, next_node.prev = next_node, prev_node
                    tmp = next_node
                self._size -= 1
            if not flag:
                tmp = tmp.next

if __name__ == '__main__':
    doctest.testmod()