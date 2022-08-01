
from __future__ import print_function
import re
import string
import doctest
import random
from abc import ABCMeta, abstractmethod
from unittest import removeResult
from common import Node, DoubleNode
from collections import deque

class BaseDataType(metaclass=ABCMeta):

    @abstractmethod
    def __iter__(self):
        while False:
            yield None

    @abstractmethod
    def size(self):
        return NotImplemented

    @abstractmethod
    def is_empty(self):
        return NotImplemented

class Stack(object):

    """
      Stack LIFO data structure linked-list implementation.
    >>> s = Stack()
    >>> s.peek()
    >>> s.push(1)
    >>> s.push(2)
    >>> s.push(3)
    >>> s.size()
    3
    >>> s.peek()
    3
    >>> for item in s:
    ...     print(item)
    ...
    3
    2
    1
    >>>
    >>> s.is_empty()
    False
    >>> s.pop()
    3
    >>> s.pop()
    2
    >>> s.pop()
    1
    >>> s.pop()
    >>> s.size()
    0
    >>> s.is_empty()
    True
    """

    def __init__(self):
        self._head = None
        self._size = 0

    def __iter__(self):
        node = self._head
        while node:
            yield node.val
            node = node.next_node

    def is_empty(self):
        return self._head is None

    def size(self):
        return self._size

    def push(self, val):
        node = Node(val)
        old = self._head
        self._head = node
        self._head.next_node = old
        self._size += 1

    def pop(self):
        if self._head:
            old = self._head
            self._head = self._head.next_node
            self._size -= 1
            return old.val
        return None

    # 1.3.7 practice
    def peek(self):
        if self._head:
            return self._head.val
        return None

class Queue(object):

    """
      Queue FIFO data structure linked-list implementation.
    >>> q = Queue()
    >>> q.is_empty()
    True
    >>> q.size()
    0
    >>> q.enqueue(1)
    >>> q.enqueue(2)
    >>> q.enqueue(3)
    >>> q.enqueue(4)
    >>> q.size()
    4
    >>> q.is_empty()
    False
    >>> [item for item in q]
    [1, 2, 3, 4]
    >>> q.dequeue()
    1
    >>> q.dequeue()
    2
    >>> q.dequeue()
    3
    >>> q.dequeue()
    4
    >>> q.dequeue()
    >>> q.dequeue()
    >>> q.size()
    0
    >>> old = Queue()
    >>> for i in range(5):
    ...     old.enqueue(i)
    ...
    >>> new_queue = Queue(old)
    >>> [i for i in new_queue]
    [0, 1, 2, 3, 4]
    >>> new_queue.enqueue(6)
    >>> [i for i in old]
    [0, 1, 2, 3, 4]
    >>> [i for i in new_queue]
    [0, 1, 2, 3, 4, 6]
    """

    # 1.3.41 practice

    def __init__(self, q=None):
        self._head = None;
        self._tail = None;
        self._head = self._tail;
        self._size = 0
        if q:
            for item in q:
                self.enqueue(item)

    def __iter__(self):
        node = self._head;
        while node:
            yield node.val
            node = node.next_node

    def is_empty(self):
        return self.size() == 0

    def size(self):
        return self._size

    def enqueue(self, val):
        old = self._tail
        node = Node(val)
        self._tail = node;
        
        if (self.is_empty()):
            self._head = self._tail
        else:
            old.next_node = self._tail
        self._size += 1

    def dequeue(self):
        if self._head is not None:
            old = self._head;
            self._head = old.next_node
            self._size -= 1
            return old.val
        return None

class Bag(object):

    '''
      Bag data structure linked-list implementation.
    >>> bag = Bag()
    >>> bag.size()
    0
    >>> bag.is_empty()
    True
    >>> for i in range(1, 6):
    ...     bag.add(i)
    ...
    >>> bag.size()
    5
    >>> [i for i in bag]
    [5, 4, 3, 2, 1]
    >>> bag.is_empty()
    False
    '''

    def __init__(self):
        self._head = None
        self._size = 0

    def __iter__(self):
        node = self._head
        while node is not None:
            yield node.val
            node = node.next_node

    def is_empty(self):
        return self._size == 0

    def size(self):
        return self._size

    def add(self, val):
        old = self._head;
        node = Node(val)
        node.next_node = old
        self._head = node
        self._size += 1
    
class BaseConverter(object):

    """
      Convert decimal number to x base number using stack.
    >>> BaseConverter.convert_decimal_integer(50, 2)
    '110010'
    >>> BaseConverter.convert_decimal_integer(8, 2)
    '1000'
    >>> BaseConverter.convert_decimal_integer(15, 16)
    'F'
    >>> BaseConverter.convert_decimal_integer(9, 8)
    '11'
    """
    digits = '0123456789ABCDEF'

    @staticmethod
    def convert_decimal_integer(dec_num, base):
        stack = Stack()

        number = dec_num
        while number > 0:
            stack.push(number % base);
            number = number // base
        
        return ''.join([BaseConverter.digits[item] for item in stack])

class Evaluate(object):

    """
      Dijkstra infix evaluate algorithm, using stack for data structure.
    >>> evaluate = Evaluate()
    >>> evaluate.calculate('(1+((2+3)*(4*5)))')
    101.0
    >>> evaluate.calculate('((1-2)*(8/4))')
    -2.0
    """

    def __init__(self):
        self._ops_stack = Stack()
        self._vals_stack = Stack()
        self._ops_char = ('+', '-', '*', '/')

    def calculate(self,  infix_string):
        for value in infix_string:
            if value in self._ops_char:
                self._ops_stack.push(value)
            elif value == '(':
                continue
            elif value == ')':
                ops = self._ops_stack.pop()
                a, b = self._vals_stack.pop(), self._vals_stack.pop();
                value = 0
                match ops:
                    case '+':
                        value = b + a
                    case '-':
                        value = b - a
                    case '*':
                        value = a * b
                    case '/':
                        value = b / a
                
                self._vals_stack.push(value);
            else:
                self._vals_stack.push(float(value))

        return self._vals_stack.peek()

# stack example
# 1.3.4 practice
class Parentheses(object):

    '''
      Using stack data structure for judging if parenthese is symmetric.
    >>> p = Parentheses()
    >>> p.is_parenthese_symmetric('[()]{}{[()()]()}')
    True
    >>> p.is_parenthese_symmetric('[(])}}{}{]])')
    False
    >>> p.is_parenthese_symmetric('{{{{}}}')
    False
    >>> p.is_parenthese_symmetric('{}}}}}}}{{{')
    False
    '''

    def __init__(self):
        self._left_parenthese_stack = Stack()
        self._left_parentheses = ('[', '{', '(')
        self._right_parentheses = (']', '}', ')')
    
    def __is_match(self, left, right):
        return self._left_parentheses.index(left) == self._right_parentheses.index(right)

    def is_parenthese_symmetric(self, parenthese_string):
        for char in parenthese_string:
            if char in self._left_parentheses:
                self._left_parenthese_stack.push(char)
            elif char in self._right_parentheses:
                if not self._left_parenthese_stack.is_empty():
                    last_parenthesis = self._left_parenthese_stack.peek()
                    if not self.__is_match(last_parenthesis, char):
                        return False
                    self._left_parenthese_stack.pop()
                else:
                    return False
            else:
                return False
        
        return self._left_parenthese_stack.is_empty()

# stack example
# 1.3.5 practice
def get_binary(integer):
    '''
      Using stack for getting integer binary representation.
    >>> get_binary(50)
    '110010'
    >>> get_binary(8)
    '1000'
    '''
    stack = Stack()
    while integer > 0:
        stack.push(integer % 2)
        integer //= 2
    
    result = [str(item) for item in stack]
    return ''.join(result)

# 1.3.9 practice
class CompleteInfixString(object):

    '''
      Using stack for complete infix string,
    the basic principle is similar to dijkstra infix arithmetic algorithm.
    >>> cis = CompleteInfixString()
    >>> cis.complete_string('1+2)*3-4)*5-6)))')
    '((1+2)*((3-4)*(5-6)))'
    '''

    def __init__(self):
        self._ops_stack = Stack()
        self._vals_stack = Stack()
        self._ops_char = ('+', '-', '*', '/')

    def complete_string(self, incmplt_string):
        for char in incmplt_string:
            if char in self._ops_char:
                self._ops_stack.push(char)
            elif char == ')':
                b, a = self._vals_stack.pop(), self._vals_stack.pop()
                ops = self._ops_stack.pop()
                val = '({}{}{})'.format(a, ops, b)
                self._vals_stack.push(val)
            else:
                self._vals_stack.push(char)
        
        return self._vals_stack.peek()

# 1.3.10 practice
class InfixToPostfix(object):

    '''
      Turn infix string to postfix string using stack.
    >>> itp = InfixToPostfix()
    >>> itp.infix_to_postfix("(A+B)*(C+D)")
    'A B + C D + *'
    >>> itp.infix_to_postfix("(A+B)*C")
    'A B + C *'
    >>> itp.infix_to_postfix("A+B*C")
    'A B C * +'
    '''
    operand = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    ops = {'*': 3, '/': 3, '+': 2, '-': 2, '(': 1}

    def __init__(self):
        self._ops_stack = Stack()
        self._vals = deque([])
        self._ops_char = ('+', '-', '*', '/')
    
    def infix_to_postfix_old(self, infix_string):
        postfix_list = []
        for i in infix_string:
            if i in InfixToPostfix.operand:
                postfix_list.append(i)
            elif i == '(':
                self._ops_stack.push(i)
            elif i == ')':
                token = self._ops_stack.pop()
                while token != '(':
                    postfix_list.append(token)
                    token = self._ops_stack.pop()
            else:
                while (not self._ops_stack.is_empty() and
                       InfixToPostfix.ops[self._ops_stack.peek()] >= InfixToPostfix.ops[i]):
                    postfix_list.append(self._ops_stack.pop())
                self._ops_stack.push(i)

        while not self._ops_stack.is_empty():
            postfix_list.append(self._ops_stack.pop())
        return ' '.join(postfix_list)

    def infix_to_postfix(self, infix_string):
        result = []
        for char in infix_string:
            if char in InfixToPostfix.operand:
                self._vals.append(char)
            elif char in self._ops_char:
                self._ops_stack.push(char)
            elif char == '(':
                continue
            elif char == ')':
                a, b = self._vals.popleft(), self._vals.popleft()
                ops = self._ops_stack.pop()
                result.append('{} {} {}'.format(a, b, ops))

        while len(self._vals):
            result.append(self._vals.popleft())

        while not self._ops_stack.is_empty():
            result.append(self._ops_stack.pop())

        return ' '.join(result)

# 1.3.11 practice
class PostfixEvaluate(object):

    '''
      Using stack for postfix evaluation.
    >>> pfe = PostfixEvaluate()
    >>> pfe.evaluate('78+32+/')
    3.0
    '''

    def __init__(self):
        self._operand_stack = Stack()
        self._operations = {
            '+': lambda a, b : a + b,
            '-': lambda a, b : a - b,
            '/': lambda a, b : a / b,
            '*': lambda a, b : a * b
        }

    def evaluate(self, postfix_string):
        for char in postfix_string:
            if char in string.digits:
                self._operand_stack.push(char)
            else:
                b, a = self._operand_stack.pop(), self._operand_stack.pop()
                result = self._operations[char](int(a), int(b))
                self._operand_stack.push(result)
        
        return self._operand_stack.pop()

# 1.3.32 practice
class Steque(object):

    """
      Steque data structure, combining stack operation and queue operation.
    >>> s = Steque()
    >>> for i in range(1, 10):
    ...     s.push(i)
    ...
    >>> s.pop()
    9
    >>> s.pop()
    8
    >>> s.enqueue(10)
    >>> ' '.join([str(i) for i in s])
    '7 6 5 4 3 2 1 10'
    >>> s2 = Steque()
    >>> for i in range(10):
    ...     s2.enqueue(i)
    ...
    >>> ' '.join([str(i) for i in s2])
    '0 1 2 3 4 5 6 7 8 9'
    >>> print_list = []
    >>> while not s2.is_empty():
    ...     print_list.append(s2.pop())
    ...
    >>> print_list
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def __init__(self):
        self._head, self._tail = None, None
        self._head = self._tail
        self._size = 0

    def __iter__(self):
        node = self._head
        while node:
            yield node.val
            node = node.next_node

    def size(self):
        return self._size;

    def is_empty(self):
        return self.size() == 0

    def push(self, val):
        old = self._head
        self._head = Node(val)
        self._head.next_node = old
        if old is None:
            self._tail = self._head
        self._size += 1
    
    def pop(self):
        if not self.is_empty():
            node = self._head;
            self._head = self._head.next_node
            if self._head is None:
                self._head = self._tail = None
            self._size -= 1
            return node.val

        return None
    
    def enqueue(self, val):
        old = self._tail;
        node = Node(val)
        if not old:
            self._head = self._tail = node
        else:
            old.next_node = node
            self._tail = node;
        self._size += 1

# 1.3.33 practice deque implementation.
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

# 1.3.34 random bag implementation.
class RandomBag(object):

    def __init__(self):
        self._bag = []

    def __iter__(self):
        random.shuffle(self._bag)
        for i in self._bag:
            yield i

    def is_empty(self):
        return len(self._bag) == 0

    def size(self):
        return len(self._bag)

    def add(self, item):
        self._bag.append(item)

# 1.3.35 random queue implementation.
class RandomQueue(object):

    def __init__(self):
        self._queue = []

    def is_empty(self):
        return len(self._queue) == 0

    def size(self):
        return len(self._queue)

    def enqueue(self, item):
        self._queue.append(item)

    def dequeue(self):
        if len(self._queue):
            index = random.randint(0, len(self._queue) - 1)
            return self._queue.pop(index)
        return None

    def sample(self):
        if len(self._queue):
            index = random.randint(0, len(self._queue) - 1)
            return self._queue[index]
        return None

    # 1.3.36 practice
    def __iter__(self):
        random.shuffle(self._queue)
        for i in self._queue:
            yield i

# 1.3.38
class GeneralizeQueue(object):

    """
    >>> queue = GeneralizeQueue()
    >>> for i in range(10):
    ...     queue.insert(i)
    ...
    >>> queue.delete(10)
    9
    >>> queue.delete(1)
    0
    >>> queue.delete(4)
    4
    >>> ' '.join([str(i) for i in queue])
    '1 2 3 5 6 7 8'
    """

    def __init__(self):
        self._head = None
        self._tail = None
        self._size = 0

    def __iter__(self):
        tmp = self._head
        while tmp:
            yield tmp.val
            tmp = tmp.next_node

    def __repr__(self) -> str:
        return ' '.join([str(item) for item in self])

    def is_empty(self):
        return self._head is None

    def size(self):
        return self._size

    def insert(self, val):
        old = self._tail
        self._tail = Node(val)
        if not self._head:
            self._head = self._tail
        else:
            old.next_node = self._tail
        self._size += 1

    def delete2(self, k):
        if k > self._size:
            return

        if k == 1:
            old = self._head
            self._head = self._head.next_node
            old.next_node = None
            self._size -= 1
            return old.val

        tmp = self._head
        count = 0

        while tmp and count != k - 2:
            tmp = tmp.next_node
            count += 1

        old = tmp.next_node
        tmp.next_node = tmp.next_node.next_node
        self._size -= 1
        old.next_node = None
        return old.val

    def delete(self, index):
        if index > self.size():
            return

        if index == 1:
            old = self._head
            self._head = old.next_node
            old.next_node = None
            self._size -= 1
            return old.val

        cur, prev, idx = self._head, None, 1

        while cur.next_node and idx < index:
            prev = cur
            cur = cur.next_node
            idx += 1

        old = prev.next_node
        prev.next_node = prev.next_node.next_node
        self._size -= 1
        old.next_node = None

        if prev.next_node is None:
            self._tail = prev

        return old.val

# 1.3.40 practice
class MoveToFront(object):

    """
      Move to front implementation, if insert new value into the list,
    then insert into the head of the list, else move the node to the head
    >>> mtf = MoveToFront()
    >>> mtf.push(5)
    >>> mtf.push(4)
    >>> mtf.push(3)
    >>> mtf.push(2)
    >>> mtf.push(1)
    >>> mtf.push(1)
    >>> mtf.push(3)
    >>> mtf.push('abcde')
    >>> for i in mtf:
    ...     print(i)
    ...
    abcde
    3
    1
    2
    4
    5
    """

    def __init__(self):
        self._head = None
        self._set = set()

    def __iter__(self):
        tmp = self._head
        while tmp:
            yield tmp.val
            tmp = tmp.next
    
    def __repr__(self) -> str:
        return ''.join([str(item) for item  in self])

    def push(self, val):
        if not self._head:
            self._head = DoubleNode(val)
            self._set.add(val)
        elif val not in self._set:
            old = self._head
            node = DoubleNode(val)
            old.prev = node
            node.next = old
            self._head = node
            self._set.add(val)
        elif val in self._set:
            if val == self._head.val:
                return
            # find current node with given value
            temp = self._head

            while temp and temp.val != val:
                temp = temp.next

            prev, curr = temp.prev, temp.next
            # extract node from list
            prev.next = curr
            curr.prev = prev
            temp.prev = temp.next = None
            temp = None
            # insert into head
            old = self._head
            self._head = DoubleNode(val)
            self._head.next = old
            old.prev = self._head

BaseDataType.register(RandomBag)
BaseDataType.register(RandomQueue)
BaseDataType.register(Deque)
BaseDataType.register(Steque)
BaseDataType.register(Bag)
BaseDataType.register(Stack)
BaseDataType.register(Queue)

if __name__ == '__main__':
    doctest.testmod()
