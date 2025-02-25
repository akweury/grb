# Created by jing at 25.02.25

from scripts.proximity import util_grid_objs


def gen_objs(is_positive, obj_size):
    objs = util_grid_objs.proximity_grid(obj_size, is_positive, cluster_num=2)
    return objs
