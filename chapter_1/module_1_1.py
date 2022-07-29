

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
        print(i)
        if (number % i == 0):
            return False
        i += 1
    
    return True

print(is_prime(23))