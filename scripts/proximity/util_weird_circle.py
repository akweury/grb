# Created by jing at 25.02.25

import random
import math

from scripts import config
from scripts.utils import encode_utils


def get_circumference_angles(cluster_num):
    """
    Generate evenly spaced points on the circumference of a circle.
    """
    angles = []
    shift = random.random() * math.pi
    for i in range(cluster_num):
        angle = (2 * math.pi / cluster_num) * i + shift
        angles.append(angle)
    return angles


def get_circumference_positions(angle, radius, num_points=2, obj_dist_factor=1):
    """
    Generate multiple points near the given center, ensuring they remain on the circumference.
    """
    positions = []
    for p_i in range(num_points):
        angle_offset = 0.3 * p_i
        shifted_angle = angle + angle_offset
        x = 0.5 + radius * math.cos(shifted_angle)*obj_dist_factor
        y = 0.5 + radius * math.sin(shifted_angle)*obj_dist_factor
        positions.append((x, y))
    return positions


def overlap_circle_features(obj_size, is_positive, cluster_num, fixed_props, obj_dist_factor):
    objs = []

    shape = "circle"
    color = random.choice(config.color_large_exclude_gray)
    cir_so = 0.3 + random.random() * 0.5
    obj_size = cir_so * 0.07
    obj = encode_utils.encode_objs(x=0.5, y=0.5, size=cir_so, color=color, shape=shape, line_width=-1,
                                   solid=True)
    objs.append(obj)

    if not is_positive:
        new_cluster_num = random.randint(1, cluster_num + 2)
        while new_cluster_num == cluster_num:
            new_cluster_num = random.randint(1, cluster_num + 2)
        cluster_num = new_cluster_num

    # Generate evenly distributed group centers on the circumference
    angles = get_circumference_angles(cluster_num)

    for a_i in range(cluster_num):
        group_size = random.randint(1, 3)
        shape = random.choice(config.bk_shapes[1:])
        # Get multiple nearby positions along the circumference
        positions = get_circumference_positions(angles[a_i], cir_so / 2, group_size, obj_dist_factor)
        for x, y in positions:
            if "shape" not in fixed_props:
                shape = random.choice(config.bk_shapes[1:])
            if "color" not in fixed_props:
                color = random.choice(config.color_large_exclude_gray)

            obj = encode_utils.encode_objs(x=x, y=y, size=obj_size, color=color, shape=shape, line_width=-1,
                                           solid=True)
            objs.append(obj)

    return objs
