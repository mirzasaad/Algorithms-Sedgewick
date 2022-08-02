import doctest
from chapter_1.common import Node

class Deque(object):

    '''
      Double queue datastructure implementaion.
    >>> d = Deque()
    >>> d.push_left(1)
    >>> d.push_right(2)
    >>> d.push_right(3)
    >>> d.push_left(0)
    >>> [i for i in d]
    [0, 1, 2, 3]
    >>> d.pop_left()
    0
    >>> d.pop_right()
    3
    >>> d.pop_left()
    1
    >>> d.pop_left()
    2
    >>> d.is_empty()
    True
    >>> d.size()
    0
    '''

    def __init__(self):
        self._head = self._tail = None
        self._size = 0

    def __iter__(self):
        tmp = self._head
        while tmp:
            yield tmp.val
            tmp = tmp.next_node

    def is_empty(self):
        return self._head is None and self._tail is None

    def size(self):
        return self._size
    
    def push_left(self, val):
        old = self._head
        node = Node(val)
        if old is None:
            self._head = self._tail = node
        else:
            node._next_node = old
            self._head = node
        self._size += 1

    def push_right(self, val):
        old = self._tail
        node = Node(val)
        if old is None:
            self._head = self._tail = node
        else:
            self._tail._next_node = node
            self._tail = node
        self._size += 1

    def pop_left(self):
        if self.is_empty():
            return None
        
        old = self._head;
        self._head = self._head.next_node;
        if self._head is None:
            self._head = self._tail = None;
        
        self._size -= 1

        return old.val

    def pop_right(self):
        if self.is_empty():
            return None

        prev, current = None, self._head;
        if self._head == self._tail:
            old = self._head;
            self._head = self._tail = None;
            self._size -=1
            return old.val

        while current.next_node:
            prev = current
            current = current.next_node

        old = self._tail
        prev.next_node = None;
        self._tail = prev
        self._size -=1
        return old.val

if __name__ == '__main__':
    doctest.testmod()