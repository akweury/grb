# Created by jing at 25.02.25


from scripts.proximity import util_red_triangle


def gen_objs(is_positive, obj_size):
    objs = util_red_triangle.proximity_red_triangle(obj_size, is_positive, cluster_num=2)
    return objs
