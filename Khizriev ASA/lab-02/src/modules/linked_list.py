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
