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

   
