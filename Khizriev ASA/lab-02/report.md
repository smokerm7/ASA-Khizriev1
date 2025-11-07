# Отчет по лабораторной работе 2
# Основные структуры данных

**Семестр:** 3 курс 5 семестр
**Группа:** ПИЖ-б-о-23-2(1)
**Дисциплина:** Анализ сложности алгоритмов
**Студент:** Хизриев Магомед-Салах Алиевич

## Цель работы
Изучить понятие и особенности базовых абстрактных типов данных (стек, очередь, дек,
 связный список) и их реализаций в Python. Научиться выбирать оптимальную структуру данных для
 решения конкретной задачи, основываясь на анализе теоретической и практической сложности
 операций. Получить навыки измерения производительности и применения структур данных для
 решения практических задач.

## Практическая часть

### Выполненные задачи
- [ ] Задача 1: Реализовать класс LinkedList (связный список) для демонстрации принципов его работы.
- [ ] Задача 2: Используя встроенные типы данных (list, collections.deque), проанализировать
 эффективность операций, имитирующих поведение стека, очереди и дека.
- [ ] Задача 3: Провести сравнительный анализ производительности операций для разных структур данных
 (list vs LinkedList для вставки, list vs deque для очереди).
- [ ] Задача 4: Решить 2-3 практические задачи, выбрав оптимальную структуру данных.


### Ключевые фрагменты кода

```PYTHON

class Node:
# modules/linked_list.py

class Node:
    """Элемент односвязного списка."""

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class LinkedList:
    """Простой односвязный список."""

    def __init__(self):
        self.head = None
        self.tail = None

    def add_first(self, value):
        """Добавляет элемент в начало списка."""
        node = Node(value)
        if not self.head:
            self.head = self.tail = node
        else:
            node.next = self.head
            self.head = node

    def add_last(self, value):
        """Добавляет элемент в конец списка."""
        node = Node(value)
        if not self.head:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node

    def remove_first(self):
        """Удаляет элемент с начала списка."""
        if not self.head:
            raise Exception("Список пуст")
        self.head = self.head.next
        if not self.head:
            self.tail = None

    def print_all(self):
        """Вывод всех элементов списка."""
        if not self.head:
            print("Список пуст")
            return
        current = self.head
        while current:
            print(current.value)
            current = current.next

```

```PYTHON
# modules/perfomance_analysis.py

import timeit
from collections import deque
from modules.linked_list import LinkedList
import matplotlib.pyplot as plt


def test_list_insert(n):
    """Вставка в начало обычного списка."""
    lst = []
    start = timeit.default_timer()
    for i in range(n):
        lst.insert(0, i)
    end = timeit.default_timer()
    return (end - start) * 1000  # ms


def test_linkedlist_insert(n):
    """Вставка в начало связанного списка."""
    ll = LinkedList()
    start = timeit.default_timer()
    for i in range(n):
        ll.add_first(i)
    end = timeit.default_timer()
    return (end - start) * 1000  # ms


def test_list_queue(n):
    """Очередь на списке."""
    lst = list(range(n))
    start = timeit.default_timer()
    for _ in range(n):
        lst.pop(0)
    end = timeit.default_timer()
    return (end - start) * 1000


def test_deque_queue(n):
    """Очередь на deque."""
    dq = deque(range(n))
    start = timeit.default_timer()
    for _ in range(n):
        dq.popleft()
    end = timeit.default_timer()
    return (end - start) * 1000


def visualize(sizes=[100, 1000, 10000, 100000]):
    """Строим графики для списка, связанного списка и очередей."""
    list_times = []
    ll_times = []
    for n in sizes:
        list_times.append(test_list_insert(n))
        ll_times.append(test_linkedlist_insert(n))

    plt.plot(sizes, list_times, "ro-", label="list")
    plt.plot(sizes, ll_times, "go-", label="linked list")
    plt.xlabel("N")
    plt.ylabel("Time ms")
    plt.title("Вставка в начало")
    plt.legend()
    plt.grid(True)
    plt.savefig("./report/list_vs_linkedlist.png", dpi=300, bbox_inches="tight")
    plt.show()

    # очередь
    list_queue_times = []
    deque_times = []
    for n in sizes:
        list_queue_times.append(test_list_queue(n))
        deque_times.append(test_deque_queue(n))

    plt.plot(sizes, list_queue_times, "ro-", label="list queue")
    plt.plot(sizes, deque_times, "go-", label="deque queue")
    plt.xlabel("N")
    plt.ylabel("Time ms")
    plt.title("Очередь: list vs deque")
    plt.legend()
    plt.grid(True)
    plt.savefig("./report/queue_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Время вставки в список:", list_times)
    print("Время вставки в linked list:", ll_times)
    print("Время очереди list:", list_queue_times)
    print("Время очереди deque:", deque_times)

    print("""

```

```PYTHON
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


```

```PYTHON
# main.py

from modules import perfomance_analysis as pa
from modules import task_solutions as ts

if __name__ == "__main__":
    sizes = [100, 1000, 10000, 100000]
    pa.visualize(sizes)

    # Скобки
    print(ts.check_brackets("{[()]}"))

    # Печать
    orders = ["Отчёт по продажам", "Дипломная работа", "Рецепт пирога"]
    ts.printing_queue(orders)

    # Палиндром
    print(ts.is_palindrome("12321"))
    print(ts.is_palindrome("12332"))

```

```bash

[0.01379998866468668, 0.18279999494552612, 9.937399998307228, 1400.2086999826133] - list
 [0.027600006433203816, 0.3482000029180199, 2.2190000163391232, 33.13450000132434] -linked_list
[0.06570000550709665, 0.49249999574385583, 26.865099993301556, 1411.3209999923129] - list
 [0.04439998883754015, 0.37910000537522137, 2.0296000002417713, 51.44469998776913] - deque
6
True
Начало печати
Дипломная работа напечатано
Отчёт по продажам напечатано
Рецепт пирога напечатано
Конец печати
False
```

<image src="./report/time_complexity_plot_list.png" style="display:block; margin: auto; height:400px">
<image src="./report/time_complexity_plot_queue.png" style="display:block; margin: auto; height:400px">

## Ответы на контрольные вопросы

## 1. Отличие динамического массива (list) от связного списка по сложности операций

- **Динамический массив (`list` в Python)** хранит элементы в **непрерывной области памяти**.  
  - Вставка в начало требует **сдвига всех элементов**, поэтому имеет сложность **O(n)**.  
  - Доступ по индексу выполняется за **O(1)**, так как элемент можно найти по адресу.  

- **Связный список** хранит элементы в **узлах**, связанных ссылками.  
  - Вставка в начало — просто изменение одной ссылки, сложность **O(1)**.  
  - Доступ по индексу требует последовательного обхода, сложность **O(n)**.

---

## 2. Принцип работы стека и очереди с примерами

- **Стек (LIFO — Last In, First Out)**: последний добавленный элемент извлекается первым.  
  **Примеры использования:**
  1. Реализация механизма *undo/redo* в редакторах.
  2. Обход дерева в глубину (DFS).

- **Очередь (FIFO — First In, First Out)**: первый добавленный элемент извлекается первым.  
  **Примеры использования:**
  1. Планирование задач в операционной системе.  
  2. Обработка запросов в принтере или веб-сервере.

---

## 3. Почему `list.pop(0)` — O(n), а `deque.popleft()` — O(1)

- В `list` элементы хранятся подряд в памяти. При удалении первого элемента все остальные **сдвигаются на одну позицию**, что требует **O(n)** времени.  
- В `deque` элементы хранятся в **двухсторонней очереди**, где есть ссылки на начало и конец. Удаление первого элемента лишь изменяет ссылку, без сдвига, поэтому выполняется за **O(1)**.

---

## 4. Какая структура данных подходит для системы "отмены действий" (undo)

Наилучший выбор — **стек (LIFO)**.  
Каждое новое действие помещается на вершину стека. При выполнении "Отмены" извлекается последнее действие, которое было выполнено последним — это идеально соответствует принципу LIFO.  
Для функции *повтора (redo)* можно использовать второй стек.

---

## 5. Почему вставка в начало списка медленнее, чем в связный список

- У **списка (list)** вставка в начало требует **сдвига всех элементов вправо**, что даёт сложность **O(n)**.  
- У **связного списка** вставка в начало — это просто добавление нового узла и изменение одной ссылки (**O(1)**).  

Поэтому при вставке 1000 элементов в начало список тратит значительно больше времени, чем связный список, что и подтверждает теоретическую асимптотику.
