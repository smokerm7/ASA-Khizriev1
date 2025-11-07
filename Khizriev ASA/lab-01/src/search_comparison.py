

# Импорт необходимых библиотек
# search_comparison_refactored.py

import matplotlib.pyplot as plt
import timeit


# --- Задача 1: Линейный поиск --- #
def linear_search(arr, target):
    """
    Линейный поиск элемента в массиве.
    Возвращает индекс target или -1, если элемент не найден.
    Сложность: O(n), где n — длина массива.
    """
    for i in range(len(arr)):      # O(n)
        if arr[i] == target:       # O(1)
            return i               # O(1)
    return -1                      # O(1)
    # Итоговая сложность: O(n)


# --- Задача 2: Бинарный поиск --- #
def binary_search(arr, target):
    """
    Бинарный поиск элемента в отсортированном массиве.
    Возвращает индекс target или -1, если элемент не найден.
    Сложность: O(log n), где n — длина массива.
    """
    left, right = 0, len(arr) - 1  # O(1)
    while left <= right:           # O(log n)
        mid = (left + right) // 2  # O(1)
        if arr[mid] == target:     # O(1)
            return mid
        elif arr[mid] < target:    # O(1)
            left = mid + 1
        else:
            right = mid - 1
    return -1
    # Итоговая сложность: O(log n)


# --- Задача 3: Теоретический анализ --- #
theory_info = """
Теоретический анализ сложности:
- Линейный поиск: O(n)
  В худшем случае требуется проверить все элементы массива.
- Бинарный поиск: O(log n)
  На каждом шаге диапазон поиска уменьшается вдвое.
"""
print(theory_info)


# --- Задача 4: Экспериментальное сравнение --- #
sizes = [1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000]


def generate_test_data(sizes):
    """
    Генерирует тестовые данные — отсортированные массивы и целевые значения.
    Возвращает словарь: {size: {'array': [...], 'targets': {...}}}
    """
    data = {}
    for size in sizes:
        arr = list(range(size))
        targets = {
            'first': arr[0],
            'middle': arr[size // 2],
            'last': arr[-1],
            'absent': -1
        }
        data[size] = {'array': arr, 'targets': targets}
    return data


test_data = generate_test_data(sizes)


def measure_time(search_func, arr, target, repeat=10):
    """
    Замеряет среднее время выполнения функции поиска по массиву.
    Возвращает среднее время (в миллисекундах).
    """
    total = 0
    for _ in range(repeat):
        total += timeit.timeit(lambda: search_func(arr, target), number=1)
    return (total / repeat) * 1000


# Сохранение результатов
element_keys = ['first', 'middle', 'last', 'absent']
results_linear = {}
results_binary = {}

print("Замеры времени (мс):\n")
print("{:>10} {:>15} {:>15}".format("Размер", "Linear Search", "Binary Search"))

for size, info in test_data.items():
    arr = info['array']
    target = info['targets']['last']  # анализ поиска последнего элемента
    t_linear = measure_time(linear_search, arr, target)
    t_binary = measure_time(binary_search, arr, target)
    results_linear[size] = t_linear
    results_binary[size] = t_binary
    print("{:>10} {:>15.4f} {:>15.4f}".format(size, t_linear, t_binary))


# --- Задача 5: Визуализация результатов --- #
def plot_results(sizes, results_linear, results_binary):
    """
    Строит графики зависимости времени поиска от размера массива:
    - линейная шкала
    - логарифмическая шкала по оси Y
    """
    y_linear = [results_linear[size] for size in sizes]
    y_binary = [results_binary[size] for size in sizes]

    # Обычный график
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, y_linear, 'ro-', label='Линейный поиск (O(n))')
    plt.plot(sizes, y_binary, 'bo-', label='Бинарный поиск (O(log n))')
    plt.xlabel('Размер массива (N)')
    plt.ylabel('Время (мс)')
    plt.title('Сравнение времени выполнения: линейный vs бинарный поиск')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig('./report/search_time_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Логарифмическая шкала
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, y_linear, 'ro-', label='Линейный поиск (O(n))')
    plt.plot(sizes, y_binary, 'bo-', label='Бинарный поиск (O(log n))')
    plt.xlabel('Размер массива (N)')
    plt.ylabel('Время (мс, логарифмическая шкала)')
    plt.yscale('log')
    plt.title('Сравнение времени (логарифмическая шкала)')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig('./report/search_time_plot_log.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_results(sizes, results_linear, results_binary)

# --- Итоговый анализ --- #
print("\nВыводы:")
print("1. Линейный поиск демонстрирует линейную зависимость времени от размера массива (O(n)).")
print("2. Бинарный поиск растёт логарифмически (O(log n)).")
print("3. Экспериментальные данные подтверждают теоретические оценки.")

<div>Линейный поиск (linear_search): теоретически O(n), время растет линейно с размером массива.
      Практически: время поиска первого элемента минимально, последнего/отсутствующего — максимально, график близок к прямой.
      Для последнего элемента требуется n сравнений.
<br>Бинарный поиск (binary_search): теоретически O(log n), время растет медленно, логарифмически.
      Практически: время почти не зависит от позиции элемента, график близок к логарифмической кривой.
      Для последнего элемента требуется log(n) сравнений.</div>
