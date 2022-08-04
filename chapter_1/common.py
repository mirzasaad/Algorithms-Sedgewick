class Node(object):
    def __init__(self, val):
        self._val = val
        self._next_node: Node | None = None

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

    @property
    def next_node(self):
        return self._next_node

    @next_node.setter
    def next_node(self, node):
        self._next_node = node

    def __cmp__(self, other):
        if self._val < other._val:
            return -1
        if self._val > other._val:
            return +1
        if self._val < other._val:
            return -1
        if self._val > other._val:
            return +1
        return 0

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

    def __repr__(self) -> str:
        return 'Node({})'.format(self._val)


class DoubleNode(object):

    def __init__(self, val):
        self._val = val
        self._prev = self._next = None

    @property
    def prev(self):
        return self._prev

    @prev.setter
    def prev(self, node):
        self._prev = node

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, node):
        self._next = node

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

    def __repr__(self) -> str:
        return 'Node({})'.format(self._val)
