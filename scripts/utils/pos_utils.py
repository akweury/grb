# Created by jing at 25.02.25
import random
import math
import numpy as np
from scipy.interpolate import make_interp_spline, interp1d


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


def get_spline_points(points, n):
    # Separate the points into x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    # Generate a smooth spline curve (use k=3 for cubic spline interpolation)
    # Spline interpolation
    spl_x = make_interp_spline(np.linspace(0, 1, len(x)), x, k=2)
    spl_y = make_interp_spline(np.linspace(0, 1, len(y)), y, k=2)

    # Dense sampling to approximate arc-length
    dense_t = np.linspace(0, 1, 1000)
    dense_x, dense_y = spl_x(dense_t), spl_y(dense_t)

    # Calculate cumulative arc length
    arc_lengths = np.sqrt(np.diff(dense_x) ** 2 + np.diff(dense_y) ** 2)
    cum_arc_length = np.insert(np.cumsum(arc_lengths), 0, 0)

    # Interpolate to find points equally spaced by arc-length
    equal_distances = np.linspace(0, cum_arc_length[-1], n)
    interp_t = interp1d(cum_arc_length, dense_t)(equal_distances)

    # Get equally spaced points
    equal_x, equal_y = spl_x(interp_t), spl_y(interp_t)

    positions = np.stack([equal_x, equal_y], axis=-1)
    return positions

