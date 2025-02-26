# Created by jing at 25.02.25
import random
import math


def generate_points(center, radius, n, min_distance):
    points = []
    attempts = 0
    max_attempts = n * 300  # To prevent infinite loops

    while len(points) < n:
        # Generate random point in polar coordinates
        r = radius * math.sqrt(random.uniform(0, 1))  # sqrt for uniform distribution in the circle
        theta = random.uniform(0, 2 * math.pi)

        # Convert polar to Cartesian coordinates
        x = center[0] + r * math.cos(theta)
        y = center[1] + r * math.sin(theta)

        new_point = (x, y)

        # Check distance from all existing points
        if all(math.hypot(x - px, y - py) >= min_distance for px, py in points):
            points.append(new_point)

        attempts += 1

    return points


def euclidean_distance(anchor, existing):
    return math.sqrt((anchor[0] - existing[0]) ** 2 + (anchor[1] - existing[1]) ** 2)


def random_pop_elements(lst):
    num_to_pop = random.randint(0, len(lst))  # Random count of elements to remove
    for _ in range(num_to_pop):
        lst.pop(random.randint(0, len(lst) - 1))  # Randomly remove an element
    return lst