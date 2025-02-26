# Created by jing at 25.02.25


import random

from scripts import config
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils import pos_utils, encode_utils, data_utils


def proximity_red_triangle(is_positive, obj_size, cluster_num, fixed_props):
    cluster_dist = 0.3  # Increased to ensure clear separation
    neighbour_dist = 0.05
    group_sizes = [2, 3]
    group_radius = 0.05

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

    # Determine how many clusters will contain a red triangle (0 to cluster_num - 1)

    if "all" in fixed_props:
        # red_triangle_clusters = random.randint(0, cluster_num - 1)
        red_triangle_neg_indices = data_utils.not_all_true(cluster_num)
        red_triangle_pos_indices = [True] * cluster_num
    elif "exist" in fixed_props:
        # red_triangle_clusters = random.randint(1, cluster_num)
        red_triangle_neg_indices = [False] * cluster_num
        red_triangle_pos_indices = data_utils.at_least_one_true(cluster_num)
    else:
        raise ValueError("fixed_props must either be 'exist' or 'all'")
    fixed_shape = "triangle"
    fixed_color = "red"

    for a_i in range(cluster_num):
        group_size = random.choice(group_sizes)
        neighbour_points = pos_utils.generate_points(group_anchors[a_i], group_radius, group_size, neighbour_dist)
        if not is_positive:
            has_red_triangle = red_triangle_neg_indices[a_i]
        else:
            has_red_triangle = red_triangle_pos_indices[a_i]

        for i in range(group_size):
            if i == 0:
                if has_red_triangle:
                    if "shape" not in fixed_props:
                        shape = random.choice(config.bk_shapes[1:])
                    else:
                        shape = fixed_shape
                    if "color" not in fixed_props:
                        color = random.choice(config.color_large_exclude_gray)
                    else:
                        color = fixed_color
                else:
                    shape = random.choice(["triangle", "square", "circle"])
                    color = random.choice(config.color_large_exclude_gray)
                    if "color" in fixed_props:
                        while color == fixed_color:
                            color = random.choice(config.color_large_exclude_gray)
                    if "shape" in fixed_props:
                        while shape == fixed_shape:
                            shape = random.choice(config.bk_shapes[1:])
            else:

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


def non_overlap_red_triangle(obj_size, is_positive, cluster_num, fixed_props):
    objs = proximity_red_triangle(is_positive, obj_size, cluster_num, fixed_props)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = proximity_red_triangle(is_positive, obj_size, cluster_num, fixed_props)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
