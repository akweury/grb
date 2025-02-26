# Created by jing at 26.02.25


import random

from scripts import config
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils import pos_utils, encode_utils


def proximity_big_small(is_positive, given_size, cluster_num, fixed_props):
    cluster_dist = 0.3  # Increased to ensure clear separation
    neighbour_dist = 0.05
    group_sizes = [2, 3]
    group_radius = 0.05
    fixed_props = fixed_props.split("_")
    if not is_positive:
        fixed_props = pos_utils.random_pop_elements(fixed_props)

    if not is_positive and "count" in fixed_props:
        new_cluster_num = random.randint(1, cluster_num + 1)
        while new_cluster_num == cluster_num:
            new_cluster_num = random.randint(1, cluster_num + 1)
        cluster_num = new_cluster_num

    def generate_random_anchor(existing_anchors):
        while True:
            anchor = [random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)]
            if all(pos_utils.euclidean_distance(anchor, existing) > cluster_dist for existing in existing_anchors):
                return anchor

    # Generate random anchors for clusters ensuring proper distance
    group_anchors = []
    for _ in range(cluster_num):
        group_anchors.append(generate_random_anchor(group_anchors))

    # group_anchors = [generate_random_anchor([]) for _ in range(cluster_num)]
    objs = []
    big_size = random.random() * 0.05 + 0.1

    # Determine how many clusters will contain a red triangle (0 to cluster_num - 1)
    big_clusters = random.randint(0, cluster_num - 1)
    big_indices = range(cluster_num)
    fixed_shape = random.choice(config.bk_shapes[1:])
    fixed_color = random.choice(config.color_large_exclude_gray)
    for a_i in range(cluster_num):
        group_size = random.choice(group_sizes)
        neighbour_points = pos_utils.generate_points(group_anchors[a_i], group_radius, group_size, neighbour_dist)
        has_big = a_i in big_indices if not is_positive else True

        for i in range(group_size):
            if i == 0:
                obj_size = big_size if has_big else given_size
                shape = fixed_shape if "shape" in fixed_props and is_positive else random.choice(config.bk_shapes[1:])
                color = fixed_color if "color" in fixed_props and is_positive else random.choice(
                    config.color_large_exclude_gray)
                if "shape" in fixed_props and not is_positive:
                    if "count" in fixed_props:
                        shape = fixed_shape
                    else:
                        while shape == fixed_shape:
                            shape = random.choice(config.bk_shapes[1:])
                if "color" in fixed_props and not is_positive:
                    if "count" in fixed_props:
                        color = fixed_color
                    else:
                        while color == fixed_color:
                            color = random.choice(config.color_large_exclude_gray)
            else:
                obj_size = given_size
                shape = random.choice(config.bk_shapes[1:])
                while "shape" in fixed_props and shape == fixed_shape:
                    shape = random.choice(config.bk_shapes[1:])
                color = random.choice(config.color_large_exclude_gray)
                while "color" in fixed_props and color == fixed_color:
                    color = random.choice(config.color_large_exclude_gray)

            x, y = neighbour_points[i]
            obj = encode_utils.encode_objs(x=x, y=y, size=obj_size, color=color, shape=shape, line_width=-1, solid=True)
            objs.append(obj)

    return objs


def proximity_big_small_2(is_positive, given_size, cluster_num, fixed_props):
    cluster_dist, neighbour_dist, group_radius = 0.3, 0.05, 0.05
    group_sizes = [2, 3]
    fixed_props = fixed_props.split("_")

    if not is_positive:
        fixed_props = pos_utils.random_pop_elements(fixed_props)

        if "count" in fixed_props:
            cluster_num = random.choice([n for n in range(1, cluster_num + 2) if n != cluster_num])

    def generate_random_anchor(existing_anchors):
        while True:
            anchor = [random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)]
            if all(pos_utils.euclidean_distance(anchor, existing) > cluster_dist for existing in existing_anchors):
                return anchor

    group_anchors = []
    for _ in range(cluster_num):
        group_anchors.append(generate_random_anchor(group_anchors))

    objs, big_size = [], random.uniform(0.1, 0.15)
    fixed_shape, fixed_color = random.choice(config.bk_shapes[1:]), random.choice(config.color_large_exclude_gray)

    for a_i in range(cluster_num):
        group_size = random.choice(group_sizes)
        neighbour_points = pos_utils.generate_points(group_anchors[a_i], group_radius, group_size, neighbour_dist)
        has_big = is_positive or a_i in range(cluster_num)

        for i, (x, y) in enumerate(neighbour_points):
            obj_size = big_size if i == 0 and has_big else given_size

            shape = fixed_shape if "shape" in fixed_props and is_positive else random.choice(config.bk_shapes[1:])
            color = fixed_color if "color" in fixed_props and is_positive else random.choice(
                config.color_large_exclude_gray)

            if "shape" in fixed_props and not is_positive:
                shape = fixed_shape if "count" in fixed_props else random.choice(
                    [s for s in config.bk_shapes[1:] if s != fixed_shape])
            if "color" in fixed_props and not is_positive:
                color = fixed_color if "count" in fixed_props else random.choice(
                    [c for c in config.color_large_exclude_gray if c != fixed_color])

            objs.append(
                encode_utils.encode_objs(x=x, y=y, size=obj_size, color=color, shape=shape, line_width=-1, solid=True))

    return objs


def non_overlap_big_small(obj_size, is_positive, cluster_num, fixed_props):
    objs = proximity_big_small(is_positive, obj_size, cluster_num, fixed_props)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = proximity_big_small(is_positive, obj_size, cluster_num, fixed_props)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs


def non_overlap_big_small_2(obj_size, is_positive, cluster_num, fixed_props):
    objs = proximity_big_small_2(is_positive, obj_size, cluster_num, fixed_props)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = proximity_big_small_2(is_positive, obj_size, cluster_num, fixed_props)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
