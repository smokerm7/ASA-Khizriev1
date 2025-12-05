# Отчет по лабораторной работе 6
# Деревья. Бинарные деревья поиска

**Дата:** 2025-10-10
**Семестр:** 3 курс 5 семестр
**Группа:** ПИЖ-б-о-23-2(1)
**Дисциплина:** Анализ сложности алгоритмов
**Студент:** Хизриев Магомед-Салах Алиевич

## Цель работы
Изучить древовидные структуры данных, их свойства и применение. Освоить основные
 операции с бинарными деревьями поиска (BST). Получить практические навыки реализации BST на
 основе узлов (pointer-based), рекурсивных алгоритмов обхода и анализа их эффективности.
 Исследовать влияние сбалансированности дерева на производительность операций.
## Практическая часть

### Выполненные задачи
- [ ] Задача 1: Реализовать бинарное дерево поиска на основе узлов с основными операциями.
- [ ] Задача 2: Реализовать различные методы обхода дерева (рекурсивные и итеративные).
- [ ] Задача 3: Реализовать дополнительные методы для работы с BST.
- [ ] Задача 4: Провести анализ сложности операций для сбалансированного и вырожденного деревьев.
- [ ] Задача 5: Визуализировать структуру дерева.



### Ключевые фрагменты кода

```PYTHON
# analysis.py

import random
import time
from modules.binary_search_tree import BinarySearchTree


def build_random_tree(size):
    """
    Создаёт почти сбалансированное BST,
    вставляя элементы в случайном порядке.
    """
    values = list(range(size))
    random.shuffle(values)
    tree = BinarySearchTree()
    for v in values:
        tree.insert(v)
    return tree


def build_sorted_tree(size):
    """
    Создаёт вырожденное BST,
    вставляя элементы в отсортированном порядке.
    """
    values = list(range(size))
    tree = BinarySearchTree()
    for v in values:
        tree.insert(v)
    return tree


def measure_search_time(tree, size, trials=1000):
    """
    Замеряет время выполнения набора поисков в дереве.
    """
    keys = [random.randrange(size) for _ in range(trials)]
    start = time.perf_counter()
    for k in keys:
        tree.search(k)
    end = time.perf_counter()
    return end - start


def run_experiment(sizes, trials_per_size=1000, repeats=5):
    """
    Запускает сравнение поиска в сбалансированном и вырожденном BST.
    Возвращает средние результаты.
    """
    results = []
    for n in sizes:
        balanced_times = []
        degenerate_times = []

        for _ in range(repeats):
            t_bal = build_random_tree(n)
            balanced_times.append(
                measure_search_time(t_bal, n, trials=trials_per_size)
            )

            t_deg = build_sorted_tree(n)
            degenerate_times.append(
                measure_search_time(t_deg, n, trials=trials_per_size)
            )

        balanced_avg = sum(balanced_times) / repeats
        degenerate_avg = sum(degenerate_times) / repeats

        results.append((n, balanced_avg, degenerate_avg))
        print(
            f"n={n}: сбалансированное ~ {balanced_avg:.6f}s, "
            f"вырожденное ~ {degenerate_avg:.6f}s"
        )

    return results

```

```PYTHON
# binary_search_tree.py

class TreeNode:
    """
    Узел дерева: хранит значение и ссылки на двух детей.
    """

    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


class BinarySearchTree:
    """
    Бинарное дерево поиска.
    Умеет вставлять, искать, удалять и показывать дерево.
    """

    def __init__(self, root=None):
        self.root = root

    def insert(self, value):
        """
        Вставка нового значения в BST.
        """
        self.root = self._insert_rec(self.root, value)

    def _insert_rec(self, node, value):
        """
        Рекурсивная вставка в BST.
        """
        if node is None:
            return TreeNode(value)
        if value == node.value:
            return node
        if value < node.value:
            node.left = self._insert_rec(node.left, value)
        else:
            node.right = self._insert_rec(node.right, value)
        return node

    def search(self, value):
        """
        Ищет значение в дереве. Возвращает узел или None.
        """
        return self._search_rec(self.root, value)

    def _search_rec(self, node, value):
        """
        Рекурсивный поиск.
        """
        if node is None:
            return None
        if value == node.value:
            return node
        if value < node.value:
            return self._search_rec(node.left, value)
        return self._search_rec(node.right, value)

    def delete(self, value):
        """
        Удаляет значение из дерева. Возвращает True, если успешно.
        """
        self.root, deleted = self._delete_rec(self.root, value)
        return deleted

    def _delete_rec(self, node, value):
        """
        Рекурсивное удаление узла.
        """
        if node is None:
            return node, False

        deleted = False
        if value < node.value:
            node.left, deleted = self._delete_rec(node.left, value)
        elif value > node.value:
            node.right, deleted = self._delete_rec(node.right, value)
        else:
            deleted = True
            # лист
            if node.left is None and node.right is None:
                return None, True
            # один потомок
            if node.left is None:
                return node.right, True
            if node.right is None:
                return node.left, True
            # два потомка → ищем минимум справа
            successor = self.find_min(node.right)
            node.value = successor.value
            node.right, _ = self._delete_rec(node.right, successor.value)

        return node, deleted

    def find_min(self, node):
        """
        Находит минимальный элемент поддерева.
        """
        current = node
        while current and current.left:
            current = current.left
        return current

    def find_max(self, node):
        """
        Находит максимальный элемент поддерева.
        """
        current = node
        while current and current.right:
            current = current.right
        return current

    def visualize(self, node=None, level=0):
        """
        Показывает дерево в консоли «боком».
        """
        if node is None:
            node = self.root

        def _viz(n, lvl):
            if n is None:
                return
            _viz(n.right, lvl + 1)
            print("    " * lvl + str(n.value))
            _viz(n.left, lvl + 1)

        _viz(node, level)

    def is_valid_bst(self):
        """
        Проверяет, корректно ли построено BST.
        """

        def helper(node, low, high):
            if node is None:
                return True
            val = node.value
            if low is not None and val <= low:
                return False
            if high is not None and val >= high:
                return False
            return helper(node.left, low, val) and helper(node.right, val, high)

        return helper(self.root, None, None)

    def height(self, node):
        """
        Высота дерева — длина самого длинного пути до листа.
        """
        if node is None:
            return 0
        return 1 + max(self.height(node.left), self.height(node.right))

```

```PYTHON
# perfomance_analysis.py

import random
import timeit
import matplotlib.pyplot as plt
from modules.binary_search_tree import BinarySearchTree


def build_tree(n, balanced=True):
    """
    Создаёт дерево из n элементов.
    Если balanced=True → порядок случайный,
    иначе — по возрастанию.
    """
    values = list(range(n))
    if balanced:
        random.shuffle(values)

    tree = BinarySearchTree()
    for v in values:
        tree.insert(v)
    return tree


def time_insert(n, balanced=True):
    """
    Замеряет время вставки одного элемента
    в BST размера n.
    """
    tree = build_tree(n, balanced=balanced)
    new_value = n
    start = timeit.default_timer()
    tree.insert(new_value)
    end = timeit.default_timer()
    return (end - start) * 1000  # в мс


def measure_time(sizes):
    """
    Для каждого размера дерева измеряет время вставки
    в сбалансированное и вырожденное BST.
    """
    res = {'sizes': list(sizes), 'balanced': [], 'degenerate': []}

    for n in sizes:
        res['balanced'].append(time_insert(n, True))
        res['degenerate'].append(time_insert(n, False))

    return res


def visualisation(sizes, out_png=None):
    """
    Строит график времени вставки от размера BST.
    """
    data = measure_time(sizes)

    plt.plot(data['sizes'], data['balanced'],
             marker='o', label='Сбалансированное')
    plt.plot(data['sizes'], data['degenerate'],
             marker='o', label='Вырожденное')

    plt.xlabel('n')
    plt.ylabel('time (ms)')
    plt.title('BST insert: balanced vs degenerate')
    plt.legend()

    if out_png:
        plt.savefig(out_png, dpi=300, bbox_inches='tight')

    plt.show()

```

```PYTHON
# tree_traversal.py

def inorder_recursive(node, visit=print):
    """
    In-order обход: left → root → right.
    """
    if node is None:
        return
    inorder_recursive(node.left, visit)
    visit(node.value)
    inorder_recursive(node.right, visit)


def preorder_recursive(node, visit=print):
    """
    Pre-order обход: root → left → right.
    """
    if node is None:
        return
    visit(node.value)
    preorder_recursive(node.left, visit)
    preorder_recursive(node.right, visit)


def postorder_recursive(node, visit=print):
    """
    Post-order обход: left → right → root.
    """
    if node is None:
        return
    postorder_recursive(node.left, visit)
    postorder_recursive(node.right, visit)
    visit(node.value)


def inorder_iterative(node, visit=print):
    """
    Итеративный in-order обход через стек.
    """
    stack = []
    cur = node

    while stack or cur:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        visit(cur.value)
        cur = cur.right

```

```PYTHON
# main.py

from modules.binary_search_tree import BinarySearchTree
from modules.analysis import run_experiment
from modules.perfomance_analysis import visualisation
import sys

# Простая демонстрация дерева
tree = BinarySearchTree()
tree.insert(5)
tree.insert(3)
tree.insert(7)
tree.insert(2)
tree.insert(4)
tree.insert(6)
tree.insert(8)
tree.insert(1)

tree.visualize()

# Запуск экспериментов с поиском
sys.setrecursionlimit(40000)
sizes = [100, 1000, 5000, 10000]
res = run_experiment(sizes, trials_per_size=1000, repeats=3)

# Графики по времени вставки
sizes = [100, 1000, 5000, 10000, 25000]
visualisation(sizes, out_png="./report/insert.png")


# Характеристики вычислительной машины
pc_info = """
Характеристики ПК для тестирования:
- Процессор: Intel Core i5-12400 @ 4.00GHz
- Оперативная память: 32 GB DDR4
- Видеокарта: NVIDIA GeForce RTX 3070 Ti
- ОС: Windows 10 Pro
- Python: 3.12
"""
print(pc_info)

```

<image src="./report/insert.png" style="display:block; margin: auto;">


```bash
        8
    7
        6
5
        4
    3
        2
            1
n=100: Сбалансированное avg 0.000404s, Вырожденное avg 0.002483s
n=1000: Сбалансированное avg 0.000857s, Вырожденное avg 0.052115s
n=5000: Сбалансированное avg 0.001126s, Вырожденное avg 0.271026s
n=10000: Сбалансированное avg 0.001359s, Вырожденное avg 0.603855s
Характеристики ПК для тестирования:
- Процессор: Intel Core i5-12400 @ 4.00GHz
- Оперативная память: 32 GB DDR4
- Видеокарта: NVIDIA GeForce RTX 3070 Ti
- ОС: Windows 10 Pro
- Python: 3.12
```









## Сравнение практической и теоретической сложности операций
| Операция     | Теоретическая сложность (средний случай) | Худший случай (вырожденное дерево) | Практическое поведение |
|---------------|------------------------------------------|-------------------------------------|------------------------|
| Вставка       | O(log n)                                 | O(n)                                | Обычно близка к O(log n), если дерево сбалансировано или данные случайны |
| Поиск         | O(log n)                                 | O(n)                                | В большинстве случаев быстрее, чем линейный поиск, но сильно зависит от формы дерева |
| Удаление      | O(log n)                                 | O(n)                                | Зависит от реализации балансировки и структуры узлов |
| Обход дерева  | O(n)                                     | O(n)                                | Всегда линейный, так как каждый узел посещается ровно один раз |

**Вывод:**  
Структура дерева напрямую влияет на производительность. Если дерево сбалансировано — операции выполняются за O(log n). Если дерево вырождено (похоже на список) — операции деградируют до O(n).





## Ответы на контрольные вопросы


## 1. Основное свойство бинарного дерева поиска (BST)
Для любого узла дерева:
- Все значения в **левом поддереве** меньше значения узла.  
- Все значения в **правом поддереве** больше значения узла.  
- Оба поддерева также являются бинарными деревьями поиска.

---

## 2. Алгоритм вставки нового элемента в BST
**Пошагово:**
1. Начать с корня дерева.  
2. Если значение меньше значения текущего узла — перейти в левое поддерево.  
3. Если больше — перейти в правое поддерево.  
4. Когда достигнут `None` (пустое место), вставить новый узел туда.  
5. Рекурсивно вернуть обновлённое поддерево.

**Сложность:**
- В **сбалансированном дереве**: O(log n)  
- В **вырожденном дереве** (например, при вставке отсортированных данных): O(n)

---

## 3. Обход дерева в глубину (DFS) и в ширину (BFS)

**DFS (Depth-First Search)** — обход в глубину:
- Использует **стек** (рекурсия или структура данных).
- Обходит одну ветвь до конца, затем возвращается.
- Варианты:
  - **Pre-order** (корень → левое → правое) — используется для копирования дерева.
  - **In-order** (левое → корень → правое) — выдаёт значения в порядке возрастания.
  - **Post-order** (левое → правое → корень) — полезен при удалении дерева.

**BFS (Breadth-First Search)** — обход в ширину:
- Использует **очередь**.
- Обходит дерево **по уровням** (от корня к нижним узлам).
- Полезен для поиска кратчайшего пути или визуализации структуры дерева.

---

## 4. Почему в вырожденном BST сложность O(n)
Если элементы вставляются **в отсортированном порядке**, дерево превращается в **цепочку узлов**, где каждый элемент имеет только одного потомка.  
В результате глубина дерева равна `n`, и любая операция (вставка, поиск, удаление) требует обхода всех узлов.

---

## 5. Сбалансированное дерево и решение проблемы вырождения
**Сбалансированное дерево** — это BST, в котором разница высот левого и правого поддерева каждого узла **не превышает 1**.

**Пример: AVL-дерево**
- После каждой вставки или удаления выполняются **повороты**, чтобы восстановить баланс.
- Гарантирует высоту дерева O(log n).
- Благодаря этому, операции **вставки, удаления и поиска** всегда выполняются за O(log n), предотвращая вырождение структуры.