# Created by jing at 01.03.25

import random
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline, interp1d

from scripts import config
from scripts.utils import encode_utils, data_utils, pos_utils
from scripts.utils.shape_utils import overlaps, overflow



def closure_big_circle(obj_size, is_positive, clu_num, params, obj_quantity):
    objs = []
    positions = []

    # Generate random anchors for clusters ensuring proper distance
    group_anchors = []
    for _ in range(clu_num):
        group_anchors.append(pos_utils.generate_random_anchor(group_anchors))

    for i in range(clu_num):
        x = group_anchors[i][0]
        y = group_anchors[i][1]
        positions += pos_utils.get_circle_positions(obj_quantity, x, y)
    obj_num = len(positions)


    if is_positive:
        if "shape" in params or random.random() < 0.5:
            shapes = random.choices(["triangle", "square"], k=obj_num)
        else:
            shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], obj_num)

        if "color" in params or random.random() < 0.5:
            colors = random.choices(["blue", "yellow"], k=obj_num)
        else:
            colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, obj_num)

        if "size" in params or random.random() < 0.5:
            sizes = [obj_size] * obj_num
        else:
            sizes = [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in range(obj_num)]
    else:
        if "shape" in params or random.random() < 0.5:
            shapes = data_utils.random_select_unique_mix(config.bk_shapes[1:], obj_num)
        else:
            shapes = [random.choice(config.bk_shapes[1:])] * obj_num
        if "color" in params or random.random() < 0.5:
            colors = data_utils.random_select_unique_mix(config.color_large_exclude_gray, obj_num)
        else:
            colors = [random.choice(config.color_large_exclude_gray)] * obj_num
        if "size" in params or random.random() < 0.5:
            sizes = [random.uniform(obj_size * 0.6, obj_size * 1.5) for _ in range(obj_num)]

        else:
            sizes = [obj_size] * obj_num
    try:
        for i in range(len(positions)):
            objs.append(encode_utils.encode_objs(
                x=positions[i][0],
                y=positions[i][1],
                size=sizes[i],
                color=colors[i],
                shape=shapes[i],
                line_width=-1,
                solid=True
            ))
    except Exception as e:
        raise e


    return objs

def non_overlap_big_circle(params, is_positive, clu_num, obj_quantity):
    obj_size = 0.05
    objs = closure_big_circle(obj_size, is_positive, clu_num, params, obj_quantity)
    t = 0
    tt = 0
    max_try = 2000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = closure_big_circle(obj_size, is_positive, clu_num, params, obj_quantity)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs