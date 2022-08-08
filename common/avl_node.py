class AVLNode():
    def __init__(self, key, value, size=1, height=0):
        self._left = self._right = None
        self._key = key
        self._value = value
        self._size = size
        self._height = height
        self._max = 0

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, max):
        self._max = max    

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
        assert isinstance(node, (AVLNode, type(None)))
        self._left = node

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, node):
        assert isinstance(node, (AVLNode, type(None)))
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
        return '(key => %s, height => %s, size => %s, max => %s)' % (self.key, self.height, self.size, self.max)
