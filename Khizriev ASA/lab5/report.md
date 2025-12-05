# Отчет по лабораторной работе 5
# Хеш-функции и хеш-таблицы

**Семестр:** 3 курс 5 семестр
**Группа:** ПИЖ-б-о-23-2(1)
**Дисциплина:** Анализ сложности алгоритмов
**Студент:** Хизриев Магомед-Салах Алиевич

## Цель работы
Изучить принципы работы хеш-функций и хеш-таблиц. Освоить методы разрешения
 коллизий. Получить практические навыки реализации хеш-таблицы с различными стратегиями
 разрешения коллизий. Провести сравнительный анализ эффективности разных методов.
## Практическая часть

### Выполненные задачи
- [ ] Задача 1: Реализовать несколько хеш-функций для строковых ключей.
- [ ] Задача 2: Реализовать хеш-таблицу с методом цепочек.
- [ ] Задача 3: Реализовать хеш-таблицу с открытой адресацией (линейное пробирование и двойное
 хеширование).
- [ ] Задача 4: Провести сравнительный анализ эффективности разных методов разрешения коллизий.
- [ ] Задача 5: Исследовать влияние коэффициента заполнения на производительность.



### Ключевые фрагменты кода

```PYTHON
# hash_functions.py


def simple_hash(str):
    """
    Считает простой хеш строки как сумму кодов всех символов.

    Args:
        str: входная строка.

    Returns:
        Целое число — получившийся хеш.
    """
    total = 0
    for ch in str:
        total += ord(ch)
    return total
    # По времени: O(n), потому что проходим по каждому символу строки


def polynomial_hash(str, p=37, mod=10**9 + 7):
    """
    Полиномиальный хеш для строки.

    Args:
        str: строка, которую хешируем.
        p: основание хеша (обычно берём простое число).
        mod: модуль, чтобы числа не разрастались.

    Returns:
        Целое число — значение хеша.
    """
    hash_value = 0
    p_pow = 1
    for ch in str:
        code = ord(ch)
        hash_value = (hash_value + code * p_pow) % mod
        p_pow = (p_pow * p) % mod
    return hash_value
    # Время работы: O(n), один проход по символам строки


def djb2_hash(str):
    """
    Реализация классической хеш-функции DJB2.

    Args:
        str: строка для хеширования.

    Returns:
        Хеш, ограниченный 32-битным значением.
    """
    hash_value = 5381
    for ch in str:
        # hash * 33 + ord(ch)
        hash_value = ((hash_value << 5) + hash_value) + ord(ch)
    return hash_value & 0xFFFFFFFF  # оставляем только 32 бита
    # Время работы: O(n), так как проходим по строке один раз

```

```PYTHON
# hash_functions.py


def simple_hash(str):
    """
    Считает простой хеш строки как сумму кодов всех символов.

    Args:
        str: входная строка.

    Returns:
        Целое число — получившийся хеш.
    """
    total = 0
    for ch in str:
        total += ord(ch)
    return total
    # По времени: O(n), потому что проходим по каждому символу строки


def polynomial_hash(str, p=37, mod=10**9 + 7):
    """
    Полиномиальный хеш для строки.

    Args:
        str: строка, которую хешируем.
        p: основание хеша (обычно берём простое число).
        mod: модуль, чтобы числа не разрастались.

    Returns:
        Целое число — значение хеша.
    """
    hash_value = 0
    p_pow = 1
    for ch in str:
        code = ord(ch)
        hash_value = (hash_value + code * p_pow) % mod
        p_pow = (p_pow * p) % mod
    return hash_value
    # Время работы: O(n), один проход по символам строки


def djb2_hash(str):
    """
    Реализация классической хеш-функции DJB2.

    Args:
        str: строка для хеширования.

    Returns:
        Хеш, ограниченный 32-битным значением.
    """
    hash_value = 5381
    for ch in str:
        # hash * 33 + ord(ch)
        hash_value = ((hash_value << 5) + hash_value) + ord(ch)
    return hash_value & 0xFFFFFFFF  # оставляем только 32 бита
    # Время работы: O(n), так как проходим по строке один раз


```

```PYTHON
# hash_table_open_addressing.py

from modules.hash_functions import polynomial_hash, djb2_hash
# from src.modules.hash_functions import polynomial_hash, djb2_hash


class LinearHashTable:
    """
    Хеш-таблица с открытой адресацией и линейным пробированием.
    Поддерживает стандартные операции: вставка, поиск, удаление.
    """

    def __init__(self, size=8, load=0.7, hash_func=polynomial_hash):
        """
        Создаёт таблицу заданного размера.

        Args:
            size: начальный размер внутреннего массива.
            load: максимальный коэффициент загрузки перед расширением.
            hash_func: хеш-функция для ключей.
        """
        self.size = size
        self.table = [None] * size
        self.count = 0
        self.load = load
        self._hash = hash_func

    def _probe(self, key, for_insert=False):
        """
        Линейное пробирование: ищем индекс для ключа.

        Args:
            key: ключ.
            for_insert: если True — ищем свободную ячейку или ячейку с этим ключом;
                        если False — ищем только существующий ключ.

        Returns:
            Индекс найденной ячейки или None, если ключ не найден
            (когда for_insert=False).
        """
        index = self._hash(key) % self.size
        start_index = index

        while self.table[index] is not None:
            if self.table[index][0] == key:
                return index
            index = (index + 1) % self.size
            if index == start_index:
                break

        if for_insert:
            return index
        return None
        # В среднем O(1), в худшем O(n), когда приходится обходить почти всю таблицу

    def insert(self, key, value):
        """
        Вставляет или обновляет пару (key, value).

        Args:
            key: ключ.
            value: значение.
        """
        if self.count >= self.size * self.load:
            self._resize()

        index = self._probe(key, for_insert=True)
        if self.table[index] is None:
            self.count += 1
        self.table[index] = (key, value)
        # В среднем O(1), но при длинной цепочке пробирования — до O(n)

    def get(self, key):
        """
        Возвращает значение по ключу.

        Args:
            key: ключ для поиска.

        Returns:
            Значение, если ключ найден, иначе None.
        """
        index = self._probe(key)
        if index is not None:
            return self.table[index][1]
        return None
        # В среднем O(1), в худшем O(n), если много коллизий

    def remove(self, key):
        """
        Удаляет элемент по ключу и перехеширует затронутую цепочку.

        Args:
            key: ключ для удаления.

        Returns:
            True, если элемент был найден и удалён, иначе False.
        """
        index = self._probe(key)
        if index is not None:
            self.table[index] = None
            self.count -= 1
            self._rehash(index)
            return True
        return False
        # В среднем O(1), в худшем O(n)

    def _rehash(self, empty_index):
        """
        После удаления восстанавливает корректность пробирования:
        заново вставляет элементы, которые шли после удалённого.
        """
        index = (empty_index + 1) % self.size

        while self.table[index] is not None:
            key, value = self.table[index]
            self.table[index] = None
            self.count -= 1
            self.insert(key, value)
            index = (index + 1) % self.size
        # В худшем случае O(n), если большая цепочка пробирования

    def _resize(self):
        """
        Увеличивает таблицу вдвое и заново вставляет все элементы.
        """
        old_table = self.table
        self.size *= 2
        self.table = [None] * self.size
        self.count = 0

        for item in old_table:
            if item is not None:
                self.insert(*item)
        # Время: O(n) на перераспределение
        # Память: O(1) дополнительно, кроме нового массива

    def display(self):
        """
        Печатает содержимое таблицы: индекс и хранимое значение.
        """
        for i in range(self.size):
            if self.table[i] is None:
                print(f"{i}) {None}")
            else:
                print(f"{i}) {self.table[i][0]}: {self.table[i][1]}")
        # Время: O(n), так как проходим по всем ячейкам


def next_prime(n):
    """
    Возвращает первое простое число, не меньшее n.

    Args:
        n: исходное число.

    Returns:
        Простейшее число >= n.
    """
    def is_prime(num):
        if num < 2:
            return False
        if num == 2:
            return True
        if num % 2 == 0:
            return False
        i = 3
        while i * i <= num:
            if num % i == 0:
                return False
            i += 2
        return True

    while not is_prime(n):
        n += 1
    return n



class DoubleHashingHashTable:
    """
    Хеш-таблица с открытой адресацией и двойным хешированием.
    Здесь второй хеш определяет шаг пробирования.
    """

    def __init__(self, size=7, load=0.7,
                 hash_func1=polynomial_hash,
                 hash_func2=djb2_hash):
        """
        Инициализация таблицы.

        Args:
            size: желаемый размер (будет округлён до ближайшего простого).
            load: коэффициент загрузки, при превышении которого расширяемся.
            hash_func1: основная хеш-функция.
            hash_func2: хеш-функция для вычисления шага.
        """
        self.size = next_prime(size)
        self.table = [None] * self.size
        self.count = 0
        self.load = load
        self._hash1 = hash_func1
        self._hash2 = hash_func2

    def _probe(self, key, for_insert=False):
        """
        Двойное хеширование: ищем индекс или свободную ячейку.

        Args:
            key: ключ.
            for_insert: если True — можно возвращать индекс под вставку,
                        иначе ищем только существующий ключ.
        """
        index = self._hash1(key) % self.size

        step = self._hash2(key) % (self.size - 1) + 1
        if step % self.size == 0:
            step = 1

        start_index = index

        while self.table[index] is not None:
            if self.table[index][0] == key:
                return index
            index = (index + step) % self.size
            if index == start_index:
                break

        if for_insert:
            return index
        return None

    def insert(self, key, value):
        """
        Вставляет пару (key, value) или обновляет значение, если ключ уже есть.

        Args:
            key: ключ.
            value: значение.
        """
        if self.count >= self.size * self.load:
            self._resize()

        index = self._probe(key, for_insert=True)
        if self.table[index] is None:
            self.count += 1
        self.table[index] = (key, value)
        # Среднее O(1), в худшем O(n)

    def get(self, key):
        """
        Возвращает значение по ключу или None, если не нашли.

        Args:
            key: ключ для поиска.
        """
        index = self._probe(key)
        if index is not None:
            return self.table[index][1]
        return None

    def remove(self, key):
        """
        Удаляет элемент по ключу и перехеширует всю таблицу.

        Args:
            key: ключ, который нужно удалить.
        """
        index = self._probe(key)
        if index is not None:
            self.table[index] = None
            self.count -= 1
            self._rehash()
            return True
        return False

    def _rehash(self):
        """
        Полностью пересобирает таблицу после удаления.
        """
        old_table = self.table
        self.table = [None] * self.size
        self.count = 0

        for item in old_table:
            if item is not None:
                key, value = item
                self.insert(key, value)

    def _resize(self):
        """
        Расширяет таблицу до следующего простого числа
        и заново вставляет все элементы.
        """
        old_table = self.table
        self.size = next_prime(self.size * 2)
        self.table = [None] * self.size
        self.count = 0

        for item in old_table:
            if item is not None:
                self.insert(*item)

    def display(self):
        """
        Печатает текущее содержимое хеш-таблицы.
        """
        for i in range(self.size):
            if self.table[i] is None:
                print(f"{i}) {None}")
            else:
                print(f"{i}) {self.table[i][0]}: {self.table[i][1]}")

```

```PYTHON
# HistCollision.py

import random
import string
import matplotlib.pyplot as plt
from modules.hash_functions import djb2_hash


class ChainHashTableByCollision:
    """
    Упрощённая хеш-таблица с цепочками.
    Нужна только для сбора статистики по коллизиям.
    """

    def __init__(self, size=100, hash_func=djb2_hash):
        self.size = size
        self.table = [[] for _ in range(size)]
        self.collisions = []
        self._hash = hash_func

    def insert(self, key):
        """
        Добавляет ключ в таблицу.
        Если в бакете уже что-то есть, то это коллизия.

        Args:
            key: ключ для вставки.
        """
        index = self._hash(key) % self.size
        chain = self.table[index]

        if len(chain) > 0:
            # если бакет не пустой — считаем, что произошла коллизия
            self.collisions.append(len(chain))

        chain.append(key)

        if sum(len(c) for c in self.table) / self.size > 0.7:
            self._resize()

    def _resize(self):
        """
        Увеличивает таблицу и раскладывает все ключи заново.
        """
        old_table = self.table
        self.size *= 2
        self.table = [[] for _ in range(self.size)]

        for chain in old_table:
            for key in chain:
                self.insert(key)


class LinearProbingHashTableByCollision:
    """
    Упрощённая таблица с линейным пробированием.
    Используется для измерения числа шагов при коллизиях.
    """

    def __init__(self, size=100, hash_func=djb2_hash):
        self.size = size
        self.table = [None] * size
        self.collisions = []
        self._hash = hash_func

    def insert(self, key):
        """
        Вставляет ключ с линейным пробированием.
        При столкновении считает, сколько шагов потребовалось до свободной ячейки.

        Args:
            key: строковый ключ.
        """
        index = self._hash(key) % self.size
        start = index
        steps = 0

        while self.table[index] is not None:
            steps += 1
            index = (index + 1) % self.size
            if index == start:
                raise Exception("Таблица переполнена")

        if steps > 0:
            self.collisions.append(steps)

        self.table[index] = key



def next_prime(n):
    """
    Возвращает первое простое число >= n.

    Args:
        n: число, от которого начинаем поиск.

    Returns:
        Ближайшее простое число.
    """
    def is_prime(num):
        if num < 2:
            return False
        if num == 2:
            return True
        if num % 2 == 0:
            return False
        i = 3
        while i * i <= num:
            if num % i == 0:
                return False
            i += 2
        return True

    while not is_prime(n):
        n += 1
    return n



class DoubleHashingHashTableByCollision:
    """
    Упрощённая хеш-таблица с двойным хешированием.
    Нужна для анализа числа шагов при коллизиях.
    """

    def __init__(self, size=100, hash_func1=djb2_hash,
                 hash_func2=djb2_hash):
        self.size = next_prime(size)
        self.table = [None] * self.size
        self.collisions = []
        self._hash1 = hash_func1
        self._hash2 = hash_func2

    def insert(self, key):
        """
        Вставляет ключ с использованием двойного хеширования.
        Считаем, сколько шагов потребовалось из-за коллизий.

        Args:
            key: строковый ключ.
        """
        checked_ind = []
        index = self._hash1(key) % self.size

        step = self._hash2(key) % (self.size - 1) + 1
        if step % self.size == 0:
            step = 1

        steps = 0
        while self.table[index] is not None:
            steps += 1
            index = (index + step) % self.size
            if index not in checked_ind:
                checked_ind.append(index)
            if len(checked_ind) == self.size:
                raise Exception("Таблица переполнена")

        if steps > 0:
            self.collisions.append(steps)

        self.table[index] = key



def generate_random_string_loop(length):
    """
    Генерирует случайную строку указанной длины.

    Args:
        length: длина строки.

    Returns:
        random_string: сгенерированная строка.
    """
    characters = string.ascii_letters + string.digits
    random_string = ""
    for _ in range(length):
        random_string += random.choice(characters)
    return random_string



def visualisation(hash_func, N=2000, func_name="table"):
    """
    Сначала генерируем случайные ключи, затем вставляем их
    в три разные таблицы и собираем статистику по коллизиям.

    Args:
        hash_func: хеш-функция, которую тестируем.
        N: количество ключей и размер таблиц.
        func_name: подпись для файла с графиком.
    """
    keys = [generate_random_string_loop(10) for _ in range(N)]

    chain_ht = ChainHashTableByCollision(N // 10, hash_func=hash_func)
    linear_ht = LinearProbingHashTableByCollision(N, hash_func=hash_func)
    double_ht = DoubleHashingHashTableByCollision(
        N, hash_func1=hash_func, hash_func2=hash_func
    )

    for k in keys:
        chain_ht.insert(k)
        linear_ht.insert(k)
        double_ht.insert(k)

    data = [chain_ht, linear_ht, double_ht]

    create_plot(data, "./report/" + func_name + ".png")


def create_plot(data, path):
    """
    Рисует и сохраняет три гистограммы распределения коллизий
    для разных способов разрешения коллизий.

    Args:
        data: список из трёх таблиц с заполненным полем collisions.
        path: путь к PNG-файлу для сохранения.
    """
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.hist(data[0].collisions, bins=20, edgecolor='black')
    plt.title("Метод цепочек")
    plt.xlabel("Количество коллизий на вставку")
    plt.ylabel("Частота")

    plt.subplot(1, 3, 2)
    plt.hist(data[1].collisions, bins=20, edgecolor='black', color='orange')
    plt.title("Линейное пробирование")
    plt.xlabel("Количество коллизий на вставку")

    plt.subplot(1, 3, 3)
    plt.hist(data[2].collisions, bins=20, edgecolor='black', color='green')
    plt.title("Двойное хеширование")
    plt.xlabel("Количество коллизий на вставку")

    plt.tight_layout()
    plt.savefig(path)
    plt.show()

```

```PYTHON
# perfomance_analysis.py

from modules.hash_table_chaining import ChainingHashTable
from modules.hash_table_open_addressing import LinearHashTable
from modules.hash_table_open_addressing import DoubleHashingHashTable
import random
import string
import timeit
import matplotlib.pyplot as plt


def generate_random_string_loop(length):
    """
    Генерирует случайную строку указанной длины.

    Args:
        length: длина строки.

    Returns:
        random_string: полученная строка.
    """
    characters = string.ascii_letters + string.digits
    random_string = ""
    for _ in range(length):
        random_string += random.choice(characters)
    return random_string


def get_time_for_chained(load, size, strings):
    """
    Замеряет среднее время вставки в хеш-таблицу с цепочками.

    Args:
        load: коэффициент заполнения, с которым создаётся таблица.
        size: сколько элементов вставляем.
        strings: список строк-ключей длины >= size.

    Returns:
        Среднее время вставки всех элементов в миллисекундах
        (усреднено по нескольким прогонам).
    """
    measures = []
    for _ in range(20):
        table = ChainingHashTable(initial_size=size, load=load)
        start = timeit.default_timer()
        for i in range(size):
            table.insert(strings[i], i)
        end = timeit.default_timer()
        measures.append((end - start) * 1000)
    return sum(measures) / len(measures)


def get_time_for_linear(load, size, strings):
    """
    Замеряет среднее время вставки в таблицу с линейным пробированием.

    Args:
        load: коэффициент заполнения.
        size: количество вставляемых элементов.
        strings: список строк-ключей длины >= size.

    Returns:
        Среднее время вставки в миллисекундах.
    """
    measures = []
    for _ in range(20):
        table = LinearHashTable(size=size, load=load)
        start = timeit.default_timer()
        for i in range(size):
            table.insert(strings[i], i)
        end = timeit.default_timer()
        measures.append((end - start) * 1000)
    return sum(measures) / len(measures)


def get_time_for_double(load, size, strings):
    """
    Замеряет среднее время вставки в таблицу с двойным хешированием.

    Args:
        load: коэффициент заполнения.
        size: количество элементов.
        strings: список строк-ключей.

    Returns:
        Среднее время вставки в миллисекундах.
    """
    measures = []
    for _ in range(20):
        table = DoubleHashingHashTable(size=size, load=load)
        start = timeit.default_timer()
        for i in range(size):
            table.insert(strings[i], i)
        end = timeit.default_timer()
        measures.append((end - start) * 1000)
    return sum(measures) / len(measures)

def measure_time(loades=[0.1, 0.5, 0.7, 0.9], size=1000):
    """
    Собирает времена вставки для разных методов в один словарь.

    Args:
        loades: список коэффициентов загрузки для теста.
        size: сколько элементов вставляем в каждую таблицу.

    Returns:
        dict: {'chain': [...], 'linear': [...], 'double': [...]},
              где в списках — среднее время вставки в ms.
    """
    strings = []
    chained_list = []
    linear_list = []
    double_list = []

    for _ in range(size):
        strings.append(generate_random_string_loop(10))

    for load in loades:
        chained_list.append(get_time_for_chained(load, size, strings))
        linear_list.append(get_time_for_linear(load, size, strings))
        double_list.append(get_time_for_double(load, size, strings))

    result = {
        "chain": chained_list,
        "linear": linear_list,
        "double": double_list
    }
    return result

def visualisation(loads=[0.1, 0.5, 0.7, 0.9], size=1000):
    """
    Строит графики зависимости времени вставки
    от коэффициента заполнения для трёх реализаций.

    Args:
        loads: значения коэффициента загрузки по оси X.
        size: количество вставляемых элементов.
    """
    measures = measure_time(loades=loads, size=size)
    chained_list = measures["chain"]
    linear_list = measures["linear"]
    double_list = measures["double"]

    create_plot(chained_list, loads,
                "Зависимость времени от коэффициента заполнения",
                "./report/chained_hashtable.png", label="chain")
    create_plot(linear_list, loads,
                "Зависимость времени от коэффициента заполнения",
                "./report/linear_hashtable.png", label="linear")
    create_plot(double_list, loads,
                "Зависимость времени от коэффициента заполнения",
                "./report/double_hashtable.png", label="double")

def create_plot(data, sizes, title, path, label):
    """
    Строит и сохраняет график времени работы
    для одного из методов хеш-таблицы.

    Args:
        data: список измеренных времен (ms).
        sizes: значения коэффициента загрузки.
        title: заголовок графика.
        path: путь к файлу PNG.
        label: подпись на графике для легенды.
    """
    plt.plot(sizes, data, marker="o", color="red", label=label)

    plt.xlabel("Коэффициент заполнения")
    plt.ylabel("Время выполнения, ms")
    plt.title(title)
    plt.legend(loc="upper left", title="Метод")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

```

```PYTHON
# main.py

import modules.perfomance_analysis as perf_test
from modules.hash_functions import simple_hash, polynomial_hash, djb2_hash
import modules.HistCollision as hist

# Строим графики производительности при разном коэффициенте загрузки
perf_test.visualisation(size=100000)

# Строим гистограммы распределения коллизий для разных хеш-функций
hist.visualisation(simple_hash, func_name="Simple")
hist.visualisation(polynomial_hash, func_name="Polynomial")
hist.visualisation(djb2_hash, func_name="DJB2")

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

<image src="./report/chained_hashtable.png" style="display:block; margin: auto;">
<image src="./report/linear_hashtable.png" style="display:block; margin: auto; ">
<image src="./report/double_hashtable.png" style="display:block; margin: auto; ">

```bash
Спецификации тестовой системы:
- Процессор: Intel Core i5-12400
- Оперативная память: 32 GB
- Видеокарта: NVIDIA GeForce RTX 3070 Ti
- Операционная система: Windows 10 Pro
- Интерпретатор Python: 3.12
```



| Функция                  | Особенности                                                                                                                          | Качество распределения                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Сумма кодов символов** | Очень простая: суммирует коды всех символов строки. Легко вычисляется.                                                               | Плохое: строки с одинаковыми символами в разном порядке дают одинаковый хеш (коллизии часты).                                        |
| **Полиномиальная (p^i)** | Использует степени числа p для каждого символа. Можно брать модуль m, чтобы ограничить размер. Позволяет учитывать порядок символов. | Хорошее: порядок символов влияет на хеш, коллизий меньше, чем у простой суммы. Выбор простого p и большого m улучшает распределение. |
| **DJB2**                 | Стартовое число 5381, каждый символ умножается на 33 (через сдвиг) и добавляется. Легко вычисляется, широко используется.            | Очень хорошее: хорошо распределяет похожие строки, меньше коллизий на практике, стабильное и быстрое.                                |

| Тип хеш-таблицы                               | Операция | Средняя сложность | Худшая сложность |
| --------------------------------------------- | -------- | ----------------- | ---------------- |
| Метод цепочек с динамическим масштабированием | Вставка  | O(1)              | O(n)             |
|                                               | Поиск    | O(1 + α)          | O(n)             |
|                                               | Удаление | O(1 + α)          | O(n)             |
| Открытая адресация с линейным пробированием   | Вставка  | O(1)    | O(n)             |
|                                               | Поиск    | O(1)    | O(n)             |
|                                               | Удаление | O(1)    | O(n)             |
| Открытая адресация с двойным хешированием     | Вставка  | O(1)   | O(n)             |
|                                               | Поиск    | O(1)   | O(n)             |
|                                               | Удаление | O(1)    | O(n)             |

## Методы разрешения коллизий

1. Метод цепочек (с динамическим масштабированием)

- Каждый элемент корзины — список (цепочка). При коллизии элементы добавляются в конец списка.

- Сложность:

- Средний случай: O(1) для поиска, вставки.

- Худший случай: O(n) (все элементы в одной цепочке).

- Масштабирование (увеличение таблицы) уменьшает длину цепочек.

- Оптимальный коэффициент заполнения (α): 0.5–0.7.

- Преимущества: простая реализация, устойчивость к коллизиям.

- Недостатки: требует дополнительную память (списки).

2. Открытая адресация с линейным пробированием

- При коллизии ищется следующая свободная ячейка по формуле h + i mod m.

- Сложность:

- Средний случай: O(1).

- Худший случай: O(n) (при «скоплении» элементов).

- Оптимальный α: 0.5–0.7. При большем заполнении резко растет количество проб.

- Преимущества: компактность (всё в одном массиве).

- Недостатки: кластеризация (скопление занятых слотов снижает производительность).

3. Открытая адресация с двойным хешированием

- При коллизии используется вторая хеш-функция: h2(k) для шага пробирования.

- Сложность:

- Средний случай: O(1).

- Худший случай: O(n), но реже, чем при линейном пробировании.

- Оптимальный α: 0.5–0.7.

- Преимущества: лучшее распределение, меньше кластеризация.

- Недостатки: чуть более сложная реализация и вычислительная нагрузка (две функции).

## Влияние хеш-функции на производительность

- Качество распределения хеш-функции напрямую влияет на длину цепочек и количество проб.

- Плохая функция (например, простая сумма кодов) создаёт частые коллизии → производительность падает.

- Хорошая функция (DJB2, полиномиальная) обеспечивает равномерное распределение → операции выполняются почти за O(1).

- В таблицах с открытой адресацией качество хеша особенно важно, поскольку коллизии влияют на всю структуру массива.



## Ответы на контрольные вопросы


### 1. Каким требованиям должна удовлетворять "хорошая" хеш-функция?
- **Равномерность распределения:** значения хеша должны равномерно распределяться по всей таблице, чтобы избежать скоплений (кластеризации).  
- **Детерминированность:** для одного и того же ключа всегда возвращается одинаковое значение.  
- **Эффективность:** вычисление должно быть быстрым (O(n), где n — длина ключа).  
- **Минимум коллизий:** разные ключи должны как можно реже давать одинаковый хеш.  
- **Чувствительность к изменениям:** небольшое изменение входных данных должно сильно менять хеш.

---

### 2. Что такое коллизия в хеш-таблице? Опишите два основных метода разрешения коллизий.
**Коллизия** — это ситуация, когда два разных ключа имеют одинаковое значение хеш-функции и попадают в одну ячейку таблицы.  

**Основные методы разрешения:**
1. **Метод цепочек:** каждая ячейка хранит список (цепочку) всех элементов с одинаковым хешом.  
2. **Открытая адресация:** при коллизии ищется другая свободная ячейка по определённому правилу (линейное пробирование, двойное хеширование и т.д.).

---

### 3. В чем разница между методом цепочек и открытой адресацией с точки зрения использования памяти и сложности операций при высоком коэффициенте заполнения?
- **Использование памяти:**
  - Метод цепочек требует дополнительной памяти для хранения связанных списков.  
  - Открытая адресация хранит все элементы в одном массиве, что экономит память.
- **Сложность при большом коэффициенте заполнения:**
  - В цепочках длина списков растёт, операции могут стать ближе к O(n).  
  - В открытой адресации увеличивается количество проб, резко падает скорость вставки и поиска.  

---

### 4. Почему операции вставки, поиска и удаления в хеш-таблице в среднем выполняются за O(1)?
Потому что:
- Хеш-функция напрямую вычисляет позицию элемента.  
- При хорошем распределении коллизии редки, и большинство операций требует лишь одно обращение к ячейке.  
- Масштабирование таблицы поддерживает низкий коэффициент заполнения, сохраняя среднее время O(1).

---

### 5. Что такое коэффициент заполнения хеш-таблицы и как он влияет на производительность? Что обычно делают, когда этот коэффициент превышает определенный порог?
**Коэффициент заполнения (α)** — это отношение числа элементов в таблице к её размеру:  
\[
α = \frac{n}{m}
\]
где *n* — количество элементов, *m* — количество ячеек.  

**Влияние:**
- При низком α операции выполняются быстро (мало коллизий).  
- При высоком α увеличивается число коллизий → падает производительность.

**При превышении порога (обычно 0.7–0.8):**
- Таблицу **масштабируют (rehash)** — создают новую таблицу большего размера.  
- Все элементы пересчитываются новой хеш-функцией или с новым модулем.
