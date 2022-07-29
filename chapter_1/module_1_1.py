

from unicodedata import numeric


def gcd(p, q):
    '''
    Calculate greatest common divisor of two numbers.
    >>> gcd(6, 4)
    2
    >>> gcd(7, 5)
    1
    >>> gcd(10, 5)
    5
    '''
    if (q == 0):
        return p

    r = p % q
    return gcd(q, r)

def is_prime(number):
    '''
    Determine whether a number is prime.
    >>> is_prime(1)
    False
    >>> is_prime(2)
    True
    >>> is_prime(3)
    True
    >>> is_prime(4)
    False
    >>> is_prime(101)
    True
    >>> is_prime(65535)
    False
    '''
    if (number < 2):
        return False
    i = 2
    while i * i <= number:
        if (number % i == 0):
            return False
        i += 1
    
    return True

def sqrt(number):
    '''
    Calculate the square of the number(Newton's method).
    >>> sqrt(4)
    2.0
    >>> sqrt(9)
    3.0
    >>> sqrt(1)
    1
    >>> sqrt(256)
    16.0
    '''

    if (number == 0 or number == 1):
        return number

    i, result = 1, 1
    while (result <= number):
        i += 1
        result = i * i
    
    return i - 1

def harmonic(number):
    '''
    Calculate the harmonic number of the given number.
    >>> harmonic(2)
    1.5
    >>> harmonic(3)
    1.8333333333333333
    '''
    return sum([1 / i for i in range(1, number + 1)])


def binary_search(key, lst):
    '''
    Determine whether the key in target list.
    Return the index of the key in the given ascending list(i - 1),
    if the key not in the list, return -1.
    >>> binary_search(3, [1, 2, 3, 4, 5])
    2
    >>> binary_search(1, [1, 2, 3, 4, 5, 6, 7, 9])
    0
    >>> binary_search(9, [1, 2, 3, 4, 5, 6, 7, 9])
    7
    >>> binary_search(999, [1, 2, 3, 4, 5, 6, 7, 9])
    -1
    '''

    assert isinstance(key, int)
    assert isinstance(lst, (list, tuple))

    low, high = 0, len(lst) - 1

    while low <= high:
        mid = int((high + low) / 2)
        if (lst[mid] == key):
            return mid
        elif (key < lst[mid]):
            high = mid - 1
        elif (key > lst[mid]):
            low = mid + 1

    return -1

def sort3num(a, b, c):
    '''
    return ascending three numbers.
    >>> sort3num(3, 2, 1)
    (1, 2, 3)
    '''

    if a > b:
        a, b = b, a
    if a > c:
        a, c = c, a
    if b > c:
        b, c = c, b
    
    return a, b, c

def exR1(number):
    if number <= 0:
        return ''
    return exR1(number - 3) + str(number) + exR1(number - 2) + str(number)

# 1.1.29 practice
def rank(key, lst):
    '''
    Return the rank of the key in the given list, there may be duplicated keys.
    >>> rank(3, [1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10])
    2
    >>> rank(4, [1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5])
    4
    '''

    assert isinstance(key, int)
    assert isinstance(lst, (list, tuple))

    lo, hi = 0, len(lst) - 1

    while lo <= hi:
        mid = int((lo + hi) / 2)
        if (lst[mid] == key):
            index = mid
            while lst[index] == key:
                index -=1
            return index + 1
        elif key < lst[mid]:
            hi = mid - 1
        elif key > lst[mid]:
            lo = mid + 1

    return -1

print(rank(3, [1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10]))
print(rank(5, [1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5]))