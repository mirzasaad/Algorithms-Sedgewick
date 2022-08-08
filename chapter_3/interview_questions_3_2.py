

from module_3_2 import BST
from common import display_bst, Node

def is_tree_binary(node, min_key, max_key):
    """
        check if tree is binary tree
    """
    if not node:
        return True
    if min_key and node.key <= node.key:
        return False
    if max_key and node.key >= node.key:
        return False

    return is_tree_binary(node.left, min_key, node.key) and is_tree_binary(node.right, node.key, max_key)

def morris_preorder_traversal(root):
    node = root
    while node:
        if not node.left:
            yield node.key
            node = node.right
        else:
            temp = node.left

            while temp.right and temp.right is not node:
                temp = temp.right

            if not temp.right:
                yield node.key
                temp.right = node
                node = node.left
            else:
                temp.right = None
                node = node.right

def morris_inorder_traversal(root):
    node = root
    while node:
        if not node.left:
            yield node.key
            node = node.right
        else:
            temp = node.left

            while temp.right and temp.right is not node:
                temp = temp.right

            if not temp.right:
                temp.right = node
                node = node.left
            else:
                temp.right = None
                yield node.key
                node = node.right

def morris_postorder_traversal(root):
    node = root
    while node:
        if not node.left:
            # yield node.key
            node = node.right
        else:
            temp = node.left

            while temp.right and temp.right is not node:
                temp = temp.right

            if not temp.right:
                temp.right = node
                node = node.left
            else:

                # predeccessor found second time
                # reverse the right references in chain from pred to p
                first = node
                middle = node.left

                while middle is not node:
                    last = middle.right
                    middle.right = first
                    first = middle
                    middle = last
                # visit the nodes from pred to p
                # again reverse the right references from pred to p

                first = node
                middle = temp
                while middle is not node:
                    yield middle.val
                    last = middle.right
                    middle.right = first
                    first = middle
                    middle = last

                temp.right = None
                node = node.right

def morrisPostorder(root):
    if (not root):
        return

    #  Create a dummy node
    dummy_node = Node(0, 0, 1)
    dummy_node.left = root
    node = dummy_node
    #  Define some useful variables
    parent = None
    middle = None
    # temp = None
    back = None
    #  iterating tree nodes
    while (node):
        if not node.left:
            #  When left child are empty then
            #  Visit to right child
            node = node.right
        else:
            #  Get to left child
            temp = node.left
            while temp.right and temp.right is not node:
                temp = temp.right

            if (temp.right is not node):
                temp.right = node
                node = node.left
            else:
                parent = node
                middle = node.left
                #  Update new path
                while middle is not node:
                    back = middle.right
                    middle.right = parent
                    print(middle.right)
                    parent = middle
                    middle = back
                
                parent = node
                middle = temp
                #  Print the resultant nodes.
                #  And correct node link in node path
                while middle is not node:
                    yield middle.val
                    print(middle.val)
                    back = middle.right
                    middle.right = parent
                    print(middle.right)
                    parent = middle
                    middle = back

                #  Unlink previous bind element
                temp.right = None
                #  Visit to right child
                node = node.right
