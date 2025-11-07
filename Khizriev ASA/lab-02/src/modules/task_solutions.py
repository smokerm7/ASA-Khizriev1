# modules/task_solutions.py

from collections import deque
import time


def check_brackets(s):
    """Проверяет баланс скобок {}, [], ()."""
    stack = []
    pairs = {"(": ")", "{": "}", "[": "]"}
    for c in s:
        if c in pairs:
            stack.append(c)
        else:
            if not stack or pairs.get(stack.pop(), None) != c:
                return False
    return not stack


def printing_queue(orders):
    """Моделируем печать документов с задержкой 2 секунды."""
    q = deque(orders)
    print("Начало печати")
    while q:
        time.sleep(2)
        print(f"{q.popleft()} напечатано")
    print("Конец печати")


def is_palindrome(seq):
    """Проверяет, является ли последовательность палиндромом."""
    d = deque(seq)
    while len(d) > 1:
        if d.popleft() != d.pop():
            return False
    return True
