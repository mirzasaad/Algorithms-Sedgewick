

class FourSum(object):
    """
        4-SUM. Given an array  a[] of n integers, the 4-SUM problem is to determine if there exist distinct indices 
    i, j, k, and l such that a[i] + a[j] = a[k] + a[l]. Design an algorithm for the 4-SUM problem that takes time 
    proportional to n^2
    """

    def solve(self, lst):
        length = len(lst) - 1
        _dict = dict()

        for i in range(length):
            for k in range(i + 1, length):
                _dict[i + k] = (i, k)

        for i in range(length):
            for k in range(i + 1, length):
                if (i + k) in _dict:
                    if i in _dict[(i + k)] or k in _dict[(i + k)]:
                        continue
                    else:
                        return [(i, k), _dict[(i + k)]] 

        return None
            









