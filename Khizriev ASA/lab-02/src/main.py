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
