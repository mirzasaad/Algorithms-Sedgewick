class State():
    def __init__(self, r, c, cost):
        self.row = r
        self.column = c
        self.cost = cost

    def __cmp__(self, other):
        return self.cost - other.cost

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
        return 'State(row =>> %s, col =>> %s, cost =>> %s)' % (self.row, self.column, self.cost)
