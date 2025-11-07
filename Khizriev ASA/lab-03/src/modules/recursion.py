# modules/recursion.py

def factorial(n):
    """Рекурсивный факториал числа n"""
    if n < 1:
        return -1
    if n == 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n):
    """Рекурсивное n-ое число Фибоначчи"""
    if n < 1:
        return -1
    if n < 3:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


def fast_power(x, p):
    """Быстрое возведение x в степень p"""
    if p < 0:
        return fast_power(1 / x, -p)
    if p == 0:
        return 1
    if p % 2 == 0:
        return fast_power(x * x, p // 2)
    return x * fast_power(x * x, (p - 1) // 2)
