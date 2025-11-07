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
