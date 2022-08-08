from collections import deque
import doctest
import random
import string

from common import Node

# notes
# for delete
# delete min go to left most tree return node.right
# delete max go to right most tree return node left
# delete if node has both children, go to right sub tree find the minimum and replace
# it with the node which is marked for delete, and then delete the min from  right sub tree

# celing, before going to left sub tree save the max value found, i.e the current node before going to left subtree
# floor, before going to right subtree save the min value found, i.e the current node before going to right subtree

# rank => wrt to inorder traversal,
# if key is equal means the size of left subtree is rank from 0-k
# if key is smaller just go left subtree, as inorder start from left most subtree
# if key is bigger go right subtree and maintain size of current_node + left_subtree and the value from right subtree

# select =. wrt inorder traversal
# if left.size === rank then return the node
# if left.size > rank, meaning left subtree is bigger go left subtree
# if left.size < rank, meaning right subtree has the potential value, so we go right and minus the left side nodes from rank


class BST(object):

    """
      Binary search tree implementation.
    >>> bst = BST()
    >>> bst.is_empty()
    True
    >>> test_str = 'EASYQUESTION'
    >>> for (index, element) in enumerate(test_str):
    ...     bst.put(element, index)
    ...
    >>> bst.is_binary_tree()
    True
    >>> bst.get('Q')
    4
    >>> bst.get('E')
    6
    >>> bst.get('N')
    11
    >>> bst.size()
    10
    >>> bst.max_val().key
    'Y'
    >>> bst.min_val().key
    'A'
    >>> bst.select(0).key
    'A'
    >>> bst.select(3).key
    'N'
    >>> bst.select(4).key
    'O'
    >>> bst.select(9).key
    'Y'
    >>> bst.rank('A')
    0
    >>> bst.rank('E')
    1
    >>> bst.rank('Y')
    9
    >>> bst.rank('T')
    7
    >>> bst.rank('U')
    8
    >>> bst.is_empty()
    False
    >>> node = bst.select(0)
    >>> node.key
    'A'
    >>> node2 = bst.select(2)
    >>> node2.key
    'I'
    >>> node3 = bst.select(9)
    >>> node3.key
    'Y'
    >>> bst.keys()
    ['A', 'E', 'I', 'N', 'O', 'Q', 'S', 'T', 'U', 'Y']
    >>> bst.height()
    5
    >>> random_key = bst.random_key()
    >>> random_key in test_str
    True
    >>> fn = bst.floor('B')
    >>> fn.key
    'A'
    >>> fn2 = bst.floor('Z')
    >>> fn2.key
    'Y'
    >>> fn3 = bst.floor('E')
    >>> fn3.key
    'E'
    >>> cn = bst.ceiling('B')
    >>> cn.key
    'E'
    >>> cn2 = bst.ceiling('R')
    >>> cn2.key
    'S'
    >>> cn3 = bst.ceiling('S')
    >>> cn3.key
    'S'
    >>> bst.delete_min()
    >>> bst.min_val().key
    'E'
    >>> bst.delete_max()
    >>> bst.max_val().key
    'U'
    >>> bst.delete('O')
    >>> bst.delete('S')
    >>> bst.keys()
    ['E', 'I', 'N', 'Q', 'T', 'U']
    >>> bst.is_binary_tree()
    True
    >>> bst.is_ordered()
    True
    >>> bst.is_rank_consistent()
    True
    >>> bst.check()
    True
    """

    def __init__(self):
        self._root: None | Node = None
        self._exist_keys = set()
        self._last_visited_node = None

    def size(self):
        """
          Return the node's amount of the binary search tree.
        """
        if not self._root:
            return 0
        return self._root.size

    def is_empty(self):
        return self._root is None

    def node_size(self, node):
        return 0 if not node else node.size

    def get(self, key):
        temp = self._root

        while temp:
            if temp.key == key:
                self._last_visited_node = temp
                return temp
            elif key < temp.key:
                temp = temp.left
            else:
                temp = temp.right

        return None

    def __put(self, node, key, val):
        if not node:
            self._last_visited_node = Node(key, val, 1)
            return self._last_visited_node
        if key < node.key:
            node.left = self.__put(node.left, key, val)
        elif key > node.key:
            node.right = self.__put(node.right, key, val)
        elif key == node.key:
            node.value = key

        node.size = self.node_size(node.left) + self.node_size(node.right) + 1

        return node

    def put(self, key, val):
        """
          Insert a new node into the binary search tree, iterate the whole tree,
        find the appropriate location for the new node and add the new node as the tree leaf.
        """
        key_exists = key in self._exist_keys
        if not key_exists:
            self._exist_keys.add(key)
        temp = self._root
        inserted_node = None
        new_node = Node(key, val, 1)

        while temp:
            inserted_node = temp
            if not key_exists:
                temp.size += 1

            if temp.key > key:
                temp = temp.left
            elif temp.key < key:
                temp = temp.right
            elif temp.key == key:
                temp.val = val
                return

        if not inserted_node:
            self._root = new_node
            return
        else:
            if inserted_node.key < key:
                inserted_node.right = new_node
            else:
                inserted_node.left = new_node

        inserted_node.size = self.node_size(
            inserted_node.left) + self.node_size(inserted_node.right) + 1

        self._last_visited_node = new_node

    def put_recursive(self, key, val):
        key_exists = key in self._exist_keys

        if not key_exists:
            self._exist_keys.add(key)
            self._root = self.__put(self._root, key, val)
        else:
            node = self.get(key)
            node.key = val
    """
        traverse tree, before going to right subtree
        save the smallest element found during the traversal
    """

    def __floor_iterative(self, node, key):
        temp = node

        while node:
            if node.key == key:
                return node
            if key < node.key:
                node = node.left
            elif key > node.left:
                temp = node
                node = node.right

        return temp

    """
        traverse tree, before going to right subtree
        save the smallest element found during the traversal
    """

    def __floor_2(self, node, key, best):
        if not node:
            return best

        if key == node.key:
            return node
        if key < node.key:
            return self.__floor_2(node.left, key, best)
        if key < node.key:
            return self.__floor_2(node.right, key, node.key)

    def floor_2(self, key):
        return self.__floor_2(self._root, key, None)

    def __floor(self, node, key):
        if not node:
            return None

        if key == node.key:
            return node

        if key < node.key:
            return self.__floor(node.left, key)

        minimum_from_right_subtree = self.__floor(node.right, key)

        return minimum_from_right_subtree if minimum_from_right_subtree else node

    """
        traverse tree, before going to right subtree
        save the smallest element found during the traversal
    """

    def floor(self, key):
        return self.__floor(self._root, key)

    """
        traverse tree, before going to left subtree
        save the max element found during the traversal
    """

    def __ceiling(self, node, key):
        if not Node:
            return None

        if key == node.key:
            return node

        if key > node.key:
            return self.__ceiling(node.right, key)

        maximum_from_left_tree = self.__ceiling(node.right, key)

        return maximum_from_left_tree if maximum_from_left_tree else node

    def ceiling(self, key):
        return self.__ceiling(self._root, key)

    def __ceiling(self, node, key):
        temp = None

        while node:
            if node.key == key:
                return node
            if key < node.key:
                temp = node
                node = node.left
            elif key > node.key:
                node = node.right

        return temp

    def ceiling_2(self, key):
        self.__ceiling_2(self._root, key, None)

    def __ceiling_2(self, node, key, best):
        if not node:
            return best

        if node.key == key:
            return node
        elif key < node.key:
            return self.__ceiling_2(node.left, key, node.key)
        elif key > node.key:
            return self.__ceiling_2(node.right, key, best)

    def __rank(self, node, key):
        if not node:
            return -1

        if key == node.key:
            return self.node_size(node.left)
        elif key < node.key:
            return self.__rank(node.left, key)
        elif key > node.key:
            return 1 + self.node_size(node.left) + self.__rank(node.right, key)

    def rank(self, key):
        rank = self.__rank(self._root, key)
        return rank

    def __select(self, node, rank):
        if not node:
            return None

        left_size = self.node_size(node.left)

        if left_size == rank:
            return node
        elif left_size > rank:
            return self.__select(node.left, rank)
        elif left_size < rank:
            return self.__select(node.right, rank - left_size - 1)

    def select(self, rank):
        return self.__select(self._root, rank)

    def __keys(self, node, queue, low, high):
        if not node:
            return
        if low < node.key:
            self.__keys(node.left, queue, low, high)
        if low <= node.key and high >= node.key:
            queue.append(node.key)
        if high > node.key:
            self.__keys(node.right, queue, low, high)

    def keys(self):
        return self.keys_range(self.min_val().key, self.max_val().key)

    def keys_range(self, low, high):
        queue = []
        self.__keys(self._root, queue, low, high)
        return queue

    def __min_val(self, node):

        if not node.left:
            return node
        else:
            return self.__min_val(node.left)

    def min_val(self):
        return self.__min_val(self._root)

    def __max_val(self, node):
        if not node.right:
            return node

        return self.__max_val(node.right)

    def max_val(self):
        return self.__max_val(self._root)

    def delete_min(self):
        self._root = self.__delete_min(self._root)

    def __delete_min(self, node):
        if not node.left:
            return node.right

        node.left = self.__delete_min(node.left)
        node.size = 1 + self.node_size(node.left) + self.node_size(node.right)

        return node

    def delete_max(self):
        self._root = self.__delete_max(self._root)

    def __delete_max(self, node):
        # find the maximum-value node.
        if not node.right:
            return node.left
        node.right = self.__delete_max(node.right)
        node.size = self.node_size(node.left) + self.node_size(node.right) + 1
        return node

    def delete(self, key):
        self._root = self.__delete(self._root, key)

    def __delete(self, node, key):
        if not node:
            return None

        if key < node.key:
            node.left = self.__delete(node.left, key)
        elif key > node.key:
            node.right = self.__delete(node.right, key)
        elif key == node.key:
            # node's left or right side is None.
            if not node.left or not node.right:
                return (node.left or node.right)
            # node's both side is not None.
            temp = node
            node = self.__min_val(temp.right)
            node.right = self.__delete_min(temp.right)
            node.left = temp.left
        node.size = self.node_size(node.left) + self.node_size(node.right) + 1

        return node

    # 3.2.6 practice, add height function for binary tree.
    def height(self):
        return self.__height(self._root)

    def __height(self, node):
        if not node:
            return -1
        return 1 + max(self.__height(node.left), self.__height(node.right))

    # 3.2.21 randomly choose a node from bianry search tree.
    def random_key(self):
        if not self._root:
            return None
        total_size = self._root.size
        rank = random.randint(0, total_size - 1)
        random_node = self.select(rank)
        return random_node.key

    # 3.2.29 practice, check if each node's size is
    # equals to the summation of left node's size and right node's size.
    def is_binary_tree(self):
        return self.__is_binary_tree(self._root)

    def __is_binary_tree(self, node):
        if not node:
            return True
        if node.size != self.node_size(node.left) + self.node_size(node.right) + 1:
            return False
        return self.__is_binary_tree(node.left) and self.__is_binary_tree(node.right)

    # 3.2.30 practice, check if each node in binary search tree is ordered
    # (less than right node and greater than left node)
    def is_ordered(self):
        return self.__is_ordered(self._root, None, None)

    def __is_ordered(self, node, min_key, max_key):
        if not node:
            return True
        if min_key and node.key <= min_key:
            return False
        if max_key and node.key >= max_key:
            return False
        return (self.__is_ordered(node.left, min_key, node.key) and
                self.__is_ordered(node.right, node.key, max_key))

     # 3.2.24 practice, check if each node's rank is correct.
    def is_rank_consistent(self):
        for i in range(self.size()):
            if i != self.rank(self.select(i).key):
                return False

        for key in self.keys():
            if key != self.select(self.rank(key)).key:
                return False

        return True

    # 3.2.32 practice, check if a data structure is binary search tree.
    def check(self):
        if not self.is_binary_tree():
            return False
        if not self.is_ordered():
            return False
        if not self.is_rank_consistent():
            return False
        return True


if __name__ == '__main__':
    doctest.testmod()
