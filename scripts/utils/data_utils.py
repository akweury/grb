# Created by jing at 26.02.25

import random

def not_all_true(n):
    if n < 1:
        return []

    # Generate a random boolean list
    bool_list = [random.choice([True, False]) for _ in range(n)]

    # Ensure at least one False exists
    if all(bool_list):
        bool_list[random.randint(0, n - 1)] = False

    return bool_list



def at_least_one_true(n):
    if n < 2:
        return [True] if n == 1 else []

    # Generate a random boolean list
    bool_list = [random.choice([True, False]) for _ in range(n)]

    # Ensure at least one True
    if not any(bool_list):
        bool_list[random.randint(0, n - 1)] = True

    # Ensure at least one False
    if all(bool_list):
        bool_list[random.randint(0, n - 1)] = False

    return bool_list
