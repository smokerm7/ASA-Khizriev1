# Отчет по лабораторной работе 3
# Рекурсия

**Семестр:** 3 курс 5 семестр
**Группа:** ПИЖ-б-о-23-2(1)
**Дисциплина:** Анализ сложности алгоритмов
**Студент:** Хизриев Магомед-Салах Алиевич

## Цель работы
Освоить принцип рекурсии, научиться анализировать рекурсивные алгоритмы и
 понимать механизм работы стека вызовов. Изучить типичные задачи, решаемые рекурсивно, и освоить
 технику мемоизации для оптимизации рекурсивных алгоритмов. Получить практические навыки
 реализации и отладки рекурсивных функций.

## Практическая часть

### Выполненные задачи
- [ ] Задача 1: Реализовать классические рекурсивные алгоритмы.
- [ ] Задача 2: Проанализировать их временную сложность и глубину рекурсии.
- [ ] Задача 3: Реализовать оптимизацию рекурсивных алгоритмов с помощью мемоизации.
- [ ] Задача 4: Сравнить производительность наивной рекурсии и рекурсии с мемоизацией.
- [ ] Задача 5: Решить практические задачи с применением рекурсии.


### Ключевые фрагменты кода

```PYTHON
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


```

```PYTHON
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
    print("""

```

```PYTHON
# modules/recursion_tasks.py

import os


def binary_search_rec(arr, target, left, right):
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] > target:
        return binary_search_rec(arr, target, left, mid - 1)
    return binary_search_rec(arr, target, mid + 1, right)


def print_tree(path, indent=""):
    print(f"{indent}{os.path.basename(path)}/")
    try:
        for f in os.listdir(path):
            full = os.path.join(path, f)
            if os.path.isdir(full):
                print_tree(full, indent + "    ")
            else:
                print(f"{indent}    {f}")
    except PermissionError:
        print(f"{indent}    [Permission Denied]")


def hanoi(n, source, target, aux):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    hanoi(n - 1, source, aux, target)
    print(f"Move disk {n} from {source} to {target}")
    hanoi(n - 1, aux, target, source)

```

```PYTHON
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

```

```bash
1 120 5040 362880 -1
1 5 13 610 -1
78125 2097152 36893488147419103232
{'time': (1755.4174000106286, 0.020399995264597237), 'calls': (18454929, 67)}
3
src/
    main.py
    modules/
        memoization.py
        recursion.py
        recursion_tasks.py
        __pycache__/
            memoization.cpython-313.pyc
            recursion.cpython-313.pyc
            recursion_tasks.cpython-313.pyc
Переместить диск 1 с A на C
Переместить диск 2 с A на B
Переместить диск 1 с C на B
Переместить диск 3 с A на C
Переместить диск 1 с B на A
Переместить диск 2 с B на C
Переместить диск 1 с A на C
[5, 10, 15, 20, 25, 30, 35]


[0.0021999876480549574, 0.008300004992634058, 0.09539999882690609, 1.0861999908229336, 11.520400003064424, 168.91429999668617, 1902.3602999950526] - naive
 [0.0009999930625781417, 0.000200001522898674, 0.000300002284348011, 0.000200001522898674, 0.0018999999156221747, 0.0016999983927235007, 0.0017999991541728377] - memoized
```

<image src="./report/fibonacci_comparison.png" style="display:block; margin: auto; height:400px">


## Ответы на контрольные вопросы

# Ответы на вопросы по теме "Рекурсия"

## 1. Базовый случай и рекурсивный шаг. Почему отсутствие базового случая приводит к ошибке

- **Базовый случай (условие выхода)** — Обязательное условие, которое прекращает рекурсивные
 вызовы и предотвращает зацикливание. 
- **Рекурсивный шаг** — Шаг, на котором задача разбивается на более простую подзадачу того же
 типа и производится рекурсивный вызов.

Если **базового случая нет**, функция будет вызывать саму себя бесконечно, что приведёт к **переполнению стека вызовов (RecursionError)** — программа не сможет завершить вычисления.

---

## 2. Как работает мемоизация и как она влияет на вычисление чисел Фибоначчи

**Мемоизация** —  Техника оптимизации, позволяющая избежать повторных   
 вычислений результатов функций для одних и тех же входных данных путем сохранения ранее
 вычисленных результатов в кеше (например, в словаре).

### Пример влияния на сложность:
- **Наивная рекурсия** для чисел Фибоначчи: `O(2^n)` — из-за повторных пересчётов одних и тех же значений.  
- **С мемоизацией**: `O(n)` — каждое значение вычисляется один раз и сохраняется.

---

## 3. Проблема глубокой рекурсии и её связь со стеком вызовов

Каждый рекурсивный вызов занимает место в **стеке вызовов**, где хранятся локальные переменные и адрес возврата.  
При слишком большой глубине рекурсии стек переполняется, и программа завершает работу с ошибкой `RecursionError`.  
Это особенно актуально для языков с ограниченным размером стека (например, Python по умолчанию ограничивает глубину до ~1000 вызовов).

---

## 4. Алгоритм решения задачи о Ханойских башнях для 3 дисков

Задача: нужно переместить 3 диска с **стержня A** на **стержень C**, используя **стержень B** как вспомогательный.

### Алгоритм:
1. Переместить 2 верхних диска с A → B (используя C как вспомогательный).
2. Переместить нижний (третий) диск с A → C.
3. Переместить 2 диска с B → C (используя A как вспомогательный).

### Последовательность шагов:
1. A → C  
2. A → B  
3. C → B  
4. A → C  
5. B → A  
6. B → C  
7. A → C  

**Всего шагов:** 7 = 2³ − 1.  
Общая сложность: **O(2ⁿ)**.

---

## 5. Рекурсивные и итеративные алгоритмы: преимущества и недостатки

| Подход | Преимущества | Недостатки |
|--------|---------------|------------|
| **Рекурсивный** | Простой и наглядный код, легко описывает задачи, основанные на самоподобии (деревья, графы, Ханойские башни). | Использует стек вызовов, может вызвать переполнение при большой глубине; иногда медленнее из-за накладных расходов на вызовы функций. |
| **Итеративный** | Эффективен по памяти, не зависит от глубины рекурсии, быстрее при больших объёмах данных. | Код может быть сложнее и менее интуитивен для задач с рекурсивной природой. |

---
