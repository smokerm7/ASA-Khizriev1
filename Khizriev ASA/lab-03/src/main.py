# main.py

from modules.recursion import factorial, fibonacci, fast_power
from modules.memoization import compare_fib, visualize_fib
from modules.recursion_tasks import binary_search_rec, print_tree, hanoi

# Recursion demo
print(factorial(1), factorial(5), factorial(7), factorial(9), factorial(-1))
print(fibonacci(1), fibonacci(5), fibonacci(7), fibonacci(15), fibonacci(-1))
print(fast_power(5, 7), fast_power(2, 21), fast_power(2, 65))

# Memoization demo
print(compare_fib(35))

# Recursion tasks demo
arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
print(binary_search_rec(arr, 7, 0, len(arr)-1))
print_tree("./src")
hanoi(3, "A", "C", "B")

# Fibonacci visualization
visualize_fib([5, 10, 15, 20, 25, 30, 35])
