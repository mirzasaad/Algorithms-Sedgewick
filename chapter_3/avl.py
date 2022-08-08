from collections import namedtuple, deque
import doctest
import string
from common import Interval

NodeWithBalanceStatus = namedtuple('NodeWithBalanceStatus', 'status, height')

class Node():
    def __init__(self, key, value, size=1, height=0):
        self._left = self._right = None
        self._key = key
        self._value = value
        self._size = size
        self._height = height

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, node):
        assert isinstance(node, (Node, type(None)))
        self._left = node

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, node):
        assert isinstance(node, (Node, type(None)))
        self._right = node

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        assert isinstance(value, int) and value >= 0
        self._size = value

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        self._key = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def __cmp__(self, other):
        return self.key - other.key

    def __eq__(self, other):
        return self.__cmp__(other) == 0

    def __ne__(self, other):
        return self.__cmp__(other) != 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __ge__(self, other):
        return self.__cmp__(other) >= 0

    def __le__(self, other):
        return self.__cmp__(other) <= 0

    def __repr__(self):
        return '(key => %s, height => %s, size => %s)' % (self.key, self.height, self.size)


class AVL():
    """
    >>> avl = AVL()
    >>> for s in string.ascii_lowercase:
    ...     avl.put(s, s)
    (key => a, height => 0, size => 1)
    (key => b, height => 0, size => 1)
    (key => c, height => 0, size => 1)
    (key => d, height => 0, size => 1)
    (key => e, height => 0, size => 1)
    (key => f, height => 0, size => 1)
    (key => g, height => 0, size => 1)
    (key => h, height => 0, size => 1)
    (key => i, height => 0, size => 1)
    (key => j, height => 0, size => 1)
    (key => k, height => 0, size => 1)
    (key => l, height => 0, size => 1)
    (key => m, height => 0, size => 1)
    (key => n, height => 0, size => 1)
    (key => o, height => 0, size => 1)
    (key => p, height => 0, size => 1)
    (key => q, height => 0, size => 1)
    (key => r, height => 0, size => 1)
    (key => s, height => 0, size => 1)
    (key => t, height => 0, size => 1)
    (key => u, height => 0, size => 1)
    (key => v, height => 0, size => 1)
    (key => w, height => 0, size => 1)
    (key => x, height => 0, size => 1)
    (key => y, height => 0, size => 1)
    (key => z, height => 0, size => 1)
    >>> orders = traverse(avl._root, {'preorder': [], 'inorder': [], 'postorder': []})
    >>> for t, a in enumerate(orders):
    ...     print(a, orders[a])
    preorder ['p', 'h', 'd', 'b', 'a', 'c', 'f', 'e', 'g', 'l', 'j', 'i', 'k', 'n', 'm', 'o', 't', 'r', 'q', 's', 'x', 'v', 'u', 'w', 'y', 'z']
    inorder ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    postorder ['a', 'c', 'b', 'e', 'g', 'f', 'd', 'i', 'k', 'j', 'm', 'o', 'n', 'l', 'h', 'q', 's', 'r', 'u', 'w', 'v', 'z', 'y', 'x', 't', 'p']
    >>> avl.delete_min()
    >>> avl.delete_max()
    >>> avl.delete('p')
    >>> avl.check()
    True
    """
    def __init__(self):
        self._last_visited = None
        self._root = None

    @property
    def last_visited(self):
        return self._last_visited

    @last_visited.setter
    def last_visited(self, value):
        self._last_visited = value

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = value

    def size(self):
        return self.__node_size(self._root)

    def is_empty(self):
        return self._root is None

    def __node_size(self, node):
        return 0 if not node else node.size

    def __height(self, node):
        return -1 if not node else node.height

    def __maximum_value(self, node):
        if not node.right:
            return node
        return self.__maximum_value(node.right)

    def maximum_value(self):
        return self.__maximum_value(self.root)

    def __minimum_value(self, node):
        if not node.left:
            return node
        return self.__minimum_value(node.left)

    def minimum_value(self):
        return self.__minimum_value(self.root)

    def __update(self, node, update_height=True, update_size=True):
        if update_size:
            node.size = 1 + \
                self.__node_size(node.left) + self.__node_size(node.right)
        if update_height:
            node.height = 1 + max(self.__height(node.left),
                                  self.__height(node.right))

    def __rotate_left(self, node):
        rotate_node = node.right
        node.right = rotate_node.left
        rotate_node.left = node

        rotate_node.size = node.size
        self.__update(node)
        rotate_node.height = 1 + \
            max(self.__height(node.left), self.__height(node.right))
        return rotate_node

    def __rotate_right(self, node):
        rotate_node = node.left
        node.left = rotate_node.right
        rotate_node.right = node

        rotate_node.size = node.size
        self.__update(node)
        rotate_node.height = 1 + \
            max(self.__height(node.left), self.__height(node.right))
        return rotate_node

    def __balance_factor(self, node):
        return self.__height(node.left) - self.__height(node.right)

    def __balance(self, node):
        if self.__balance_factor(node) < -1:
            if (self.__balance_factor(node.right) > 0):
                node.right = self.__rotate_right(node.right)
            node = self.__rotate_left(node)
        elif self.__balance_factor(node) > 1:
            if (self.__balance_factor(node.left) < 0):
                node.left = self.__rotate_left(node.left)
            node = self.__rotate_right(node)
        return node

    def __put(self, node, key, value):
        if not node:
            new_node = Node(key, value)
            self.last_visited = new_node
            return new_node
        if key < node.key:
            node.left = self.__put(node.left, key, value)
        elif key > node.key:
            node.right = self.__put(node.right, key, value)
        else:
            self.last_visited = node
            node.value = value
            return node

        self.__update(node)
        return self.__balance(node)

    def put(self, key, value):
        self.root = self.__put(self.root, key, value)
        return self.last_visited

    def delete_min(self):
        self._root = self.__delete_min(self._root)

    def __delete_min(self, node):
        if not node.left:
            return node.right
        node.left = self.__delete_min(node.left)
        self.__update(node)
        return self.__balance(node)

    def delete_max(self):
        self._root = self.__delete_max(self._root)

    def __delete_max(self, node):
        if not node.right:
            return node.left
        node.right = self.__delete_max(node.right)
        self.__update(node)
        return self.__balance(node)

    def __delete(self, node, key):
        if key < node.key:
            node.left = self.__delete(node.left,  key)
        elif key > node.key:
            node.right = self.__delete(node.right, key)
        else:
            if not node.left or not node.right:
                return node.right or node.left
            x = node
            node = self.__minimum_value(node.right)
            node.right = self.__delete_min(x.right)
            node.left = x.left
        self.__update(node)
        return self.__balance(node)

    def keys(self):
        return self.keys_range(self.minimum_value().key, self.maximum_value().key)

    def keys_range(self, low, high):
        queue = []
        self.__keys(self._root, queue, low, high)
        return queue

    def __keys(self, node, queue, low, high):
        if not node:
            return
        if low < node.key:
            self.__keys(node.left, queue, low, high)
        if low <= node.key and high >= node.key:
            queue.append(node.key)
        if high > node.key:
            self.__keys(node.right, queue, low, high)

    def delete(self, key):
        self.root = self.__delete(self._root, key)

    def ___height(self, node):
        if not node:
            return -1
        else:
            return 1 + max(self.___height(node.left), self.___height(node.right))

    def height(self):
        return self.___height(self.root)

    def __is_height_balaced(self, node):
        if not node:
            return NodeWithBalanceStatus(True, -1)

        left = self.__is_height_balaced(node.left)
        if not left.status:
            return NodeWithBalanceStatus(False, left.height)

        right = self.__is_height_balaced(node.right)
        if not right.status:
            return NodeWithBalanceStatus(False, right.height)

        height = 1 + max(left.height, right.height)
        is_balance = abs(left.height - right.height) <= 1
        return NodeWithBalanceStatus(is_balance, height)

    def is_height_balaced(self):
        return self.__is_height_balaced(self.root).status

    def __is_balanced(self, node):
        return self.__is_balanced(node.left) and self.__is_balanced(node.right) and self.__balance_factor(node) <= 1 if node else True

    def is_balanced(self):
        return self.__is_balanced(self.root)

    def rank(self, key):
        return self.__rank(self.root, key)

    def __rank(self, node, key):
        if not node:
            return 0
        if key > node.key:
            return 1 + self.__node_size(node.left) + self.__rank(node.right,  key)
        elif key < node.key:
            return self.__rank(node.left, key)
        elif key == node.key:
            return self.__node_size(node.left)

    def rank_iterative(self, key):
        r, node = 0, self._root
        while node:
            if key < node.key:
                node = node.left
            elif key > node.key:
                r += 1 + self.__node_size(node.left)
                node = node.right
            elif key == node.key:
                return r + self.__node_size(node.left)
        return -1

    def __select(self, node, k):
        if not node:
            return None
        left_size = self.__node_size(node.left)
        if left_size > k:
            return self.__select(node.left, k)
        elif left_size < k:
            return self.__select(node.right, k - left_size - 1)
        else:
            return node

    def select(self, k):
        return self.__select(self.root, k)

    def __ceiling(self, node, key):
        if not node:
            return None
        if key == node.key:
            return node
        if key > node.key:
            return self.__ceiling(node.right, key)

        hi = self.__ceiling(node.left, key)
        return hi or node

    def ceiling(self, key):
        return self.__ceiling(self.root, key)

    def __floor(self, node, key):
        if not node:
            return None
        if key == node.key:
            return node
        if key < node.key:
            return self.__floor(node.left, key)
        low = self.__floor(node.right, key)
        return low or node

    def floor(self, key):
        return self.__floor(self.root, key)

    def __get(self, node, key):
        return node if not node or key == node.key else (self.__get(node.left, key) if key < node.key else self.__get(node.right, key))

    def get(self, key):
        return self.__get(self.root, key)

    def level_order(self):
        tree = deque([self.root])
        result = deque([])
        while tree:
            current = tree.popleft()
            if not current:
                break
            tree.extend([child for child in (current.left, current.right)])
            result.append(current.key)
        return result

    def is_bst(self):
        return self.__is_bst(self.root, None, None)

    def __is_bst(self, node, min, max):
        if not node:
            return True
        if min and node.key <= min:
            return False
        if max and node.key >= max:
            return False
        return self.__is_bst(node.left, min, node.key) and self.__is_bst(node.right, node.key, max)

    def is_AVL(self):
        return self.__is_AVL(self.root)

    def __is_AVL(self, node):
        if not node:
            return True
        bf = self.__balance_factor(node)
        if -1 > bf or bf > 1:
            return False
        return self.__is_AVL(node.left) and self.__is_AVL(node.right)

    def is_size_consistent(self):
        return self.__is_size_consistent(self.root)

    def __is_size_consistent(self, node):
        if not node:
            return True
        if self.__node_size(node.left) + self.__node_size(node.right) + 1 != node.size:
            return False
        return self.__is_size_consistent(node.left) and self.__is_size_consistent(node.right)

    def is_rank_consistent(self):
        for i in range(self.__node_size(self.root)):
            if i != self.rank(self.select(i).key):
                return False

        for i, k in enumerate(self.keys()):
            if self.rank(k) != i:
                return False
        return True

    def check(self):
        assert self.is_bst()
        assert self.is_height_balaced()
        assert self.is_rank_consistent()
        assert self.is_size_consistent()
        assert self.is_AVL()
        return True

    def range(self, _min, _max):
        result = []
        self.__range(self.root, Interval(_min, _max), result)
        return result

    def __range(self, node, interval, result):
        if not node:
            return
        if interval.min < node.key:
            self.__range(node.left, interval, result)
        if interval.contains_eq(node.key):
            result.append(node.value)
        if interval.max > node.key:
            self.__range(node.right, interval, result)


def traverse(root, d):
    if root:
        d['preorder'].append(root.key)
        traverse(root.left, d)
        d['inorder'].append(root.key)
        traverse(root.right, d)
        d['postorder'].append(root.key)
    return d

if __name__ == '__main__':
    doctest.testmod()
