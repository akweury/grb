# Created by jing at 27.02.25
import random
import numpy as np
import math
from scripts import config
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils import pos_utils, encode_utils


def get_circumference_points(cluster_num, x, y, radius):
    """
    Generate evenly spaced points on the circumference of a circle.
    """
    points = []
    for i in range(cluster_num):
        angle = (2 * math.pi / cluster_num) * i
        cx = x + radius * math.cos(angle)
        cy = y + radius * math.sin(angle)
        points.append((cx, cy))
    return points


def get_surrounding_positions(center, radius, num_points=2):
    """
    Generate multiple points near the given center, ensuring they remain on the circumference.
    """
    positions = []
    for _ in range(num_points):
        angle_offset = random.uniform(-0.2, 0.2)  # Small random variation
        angle = math.atan2(center[1] - 0.5, center[0] - 0.5) + angle_offset
        if random.random() < 0.5:
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
        else:
            x = 0.5 - radius * math.cos(angle)
            y = 0.5 - radius * math.sin(angle)
        positions.append((x, y))
    return positions

def get_symmetry_on_cir_positions(angle, radius, num_points=2):
    """
    Generate multiple points near the given center, ensuring they remain on the circumference.
    """
    positions = []
    for p_i in range(num_points):
        angle_offset = 0.3 * p_i
        shifted_angle = angle + angle_offset
        x = 0.5 + radius * math.cos(shifted_angle)
        y = 0.5 + radius * math.sin(shifted_angle)
        positions.append((x, y))
    return positions

def symmetry_solar_sys( is_positive, obj_size, cluster_num, fixed_props):
    objs = []

    shape = "circle"
    color = random.choice(config.color_large_exclude_gray)
    cir_so = 0.3 + random.random() * 0.5

    objs.append(encode_utils.encode_objs(
        x=0.5,
        y=0.5,
        size=cir_so, color=color, shape=shape,
        line_width=-1, solid=True,
    ))
    # Generate evenly distributed group centers on the circumference
    group_centers = get_circumference_points(cluster_num, 0.5, 0.5, cir_so)

    for a_i in range(cluster_num):
        shape = random.choice(config.bk_shapes[1:])
        if is_positive:
            group_size = random.randint(1, 3)
            positions = get_symmetry_on_cir_positions(group_centers[a_i], cir_so, group_size)
        else:
            group_size = random.randint(2, 4)
            # Get multiple nearby positions along the circumference
            positions = get_surrounding_positions(group_centers[a_i], cir_so, group_size)

        for x, y in positions:
            objs.append(encode_utils.encode_objs(
                x=x,
                y=y,
                size=obj_size, color=color, shape=shape,
                line_width=-1, solid=True,
            ))
    return objs


def non_overlap_red_triangle(obj_size, is_positive, cluster_num, fixed_props):
    objs = symmetry_solar_sys(is_positive, obj_size, cluster_num, fixed_props)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = symmetry_solar_sys(is_positive, obj_size, cluster_num, fixed_props)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
