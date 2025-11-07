# modules/memoization.py

import timeit
import matplotlib.pyplot as plt

fib_calls = 0
fib_memo_calls = 0


def naive_fib(n):
    global fib_calls
    fib_calls += 1
    if n < 1:
        return -1
    if n < 3:
        return 1
    return naive_fib(n - 1) + naive_fib(n - 2)


def fib_memo(n, memo={}):
    global fib_memo_calls
    fib_memo_calls += 1
    if n in memo:
        return memo[n]
    if n < 1:
        return -1
    if n < 3:
        return 1
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


def compare_fib(n):
    global fib_calls, fib_memo_calls

    start = timeit.default_timer()
    naive_fib(n)
    end = timeit.default_timer()
    naive_time = (end - start) * 1000
    naive_count = fib_calls
    fib_calls = 0

    start = timeit.default_timer()
    fib_memo(n)
    end = timeit.default_timer()
    memo_time = (end - start) * 1000
    memo_count = fib_memo_calls
    fib_memo_calls = 0

    return {"time": (naive_time, memo_time), "calls": (naive_count, memo_count)}


def visualize_fib(sizes):
    naive_times, memo_times = [], []
    for n in sizes:
        res = compare_fib(n)
        naive_times.append(res["time"][0])
        memo_times.append(res["time"][1])

    plt.plot(sizes, naive_times, "ro-", label="Naive")
    plt.plot(sizes, memo_times, "bo-", label="Memoized")
    plt.xlabel("n")
    plt.ylabel("Time ms")
    plt.title("Naive vs Memoized Fibonacci")
    plt.legend()
    plt.grid(True)
    plt.savefig("ОТЧЁТ/fibonacci_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Naive times:", naive_times)
    print("Memoized times:", memo_times)

