# Created by jing at 25.02.25

from scripts.proximity.util_grid_objs import non_overlap_grid
from scripts.proximity.util_red_triangle import non_overlap_red_triangle
from scripts.proximity.util_weird_circle import overlap_circle_features
from scripts.proximity.util_fixed_props import non_overlap_fixed_props
from scripts.proximity.util_big_small import non_overlap_big_small, non_overlap_big_small_2

""" 
p: positive
s: size

"""


def create_tasks(func, obj_size, task_sizes, *args):
    return {f"{func.__name__}_{'_'.join(map(str, args))}_{s}": (lambda p, s=s, args=args: func(obj_size, p, s, *args))
            for s in task_sizes}


# Define task functions dynamically
tasks = {}

# symbolic features

# color, all
tasks.update(create_tasks(non_overlap_red_triangle, 0.05, range(1, 5), "color_all"))
# color, exist
tasks.update(create_tasks(non_overlap_red_triangle, 0.05, range(2, 5), "color_exist"))

# shape, all
tasks.update(create_tasks(non_overlap_red_triangle, 0.05, range(1, 5), "shape_all"))
# shape, exist
tasks.update(create_tasks(non_overlap_red_triangle, 0.05, range(2, 5), "shape_exist"))

# color, shape, all
tasks.update(create_tasks(non_overlap_red_triangle, 0.05, range(1, 5), "shape_color_all"))
# color, shape, exist
tasks.update(create_tasks(non_overlap_red_triangle, 0.05, range(2, 5), "shape_color_exist"))

# shape
tasks.update(create_tasks(non_overlap_grid, 0.05, range(2, 5), "shape"))
tasks.update(create_tasks(non_overlap_grid, 0.05, range(2, 5), "shape"))

# color
tasks.update(create_tasks(non_overlap_grid, 0.05, range(2, 5), "color"))
tasks.update(create_tasks(non_overlap_grid, 0.05, range(2, 5), "color"))

# shape, color
tasks.update(create_tasks(non_overlap_grid, 0.05, range(2, 5), "shape_color"))
tasks.update(create_tasks(non_overlap_grid, 0.05, range(2, 5), "shape_color"))

# color, shape,
tasks.update(create_tasks(non_overlap_fixed_props, 0.05, range(2, 5), "color_shape"))
tasks.update(create_tasks(non_overlap_fixed_props, 0.05, range(2, 5), "shape"))
tasks.update(create_tasks(non_overlap_fixed_props, 0.05, range(2, 5), "color"))

# size, color, shape
tasks.update(create_tasks(non_overlap_big_small, 0.05, range(2, 5), "color"))
tasks.update(create_tasks(non_overlap_big_small, 0.05, range(2, 5), "shape"))
tasks.update(create_tasks(non_overlap_big_small, 0.05, range(2, 5), "color_shape"))

# size, color, shape, count
tasks.update(create_tasks(non_overlap_big_small, 0.05, range(2, 5), "count_color"))
tasks.update(create_tasks(non_overlap_big_small, 0.05, range(2, 5), "count_shape"))
tasks.update(create_tasks(non_overlap_big_small, 0.05, range(2, 5), "count_color_shape"))

tasks.update(create_tasks(non_overlap_big_small_2, 0.05, range(2, 5), "count_color"))
tasks.update(create_tasks(non_overlap_big_small_2, 0.05, range(2, 5), "count_shape"))
tasks.update(create_tasks(non_overlap_big_small_2, 0.05, range(2, 5), "count_color_shape"))

# neural features
# color, shape, count
# tasks.update(create_tasks(overlap_circle_features, 0.05, range(2, 5), "color", 0.8))
tasks.update(create_tasks(overlap_circle_features, 0.05, range(2, 5), "color", 1))
tasks.update(create_tasks(overlap_circle_features, 0.05, range(2, 5), "color", 1.2))
# tasks.update(create_tasks(overlap_circle_features, 0.05, range(2, 5), "shape", 0.8))
tasks.update(create_tasks(overlap_circle_features, 0.05, range(2, 5), "shape", 1))
tasks.update(create_tasks(overlap_circle_features, 0.05, range(2, 5), "shape", 1.2))
# tasks.update(create_tasks(overlap_circle_features, 0.05, range(2, 5), "color_shape", 0.8))
tasks.update(create_tasks(overlap_circle_features, 0.05, range(2, 5), "color_shape", 1))
tasks.update(create_tasks(overlap_circle_features, 0.05, range(2, 5), "color_shape", 1.2))

# Convert tasks to pattern dictionary
pattern_dicts = [{"name": key, "module": task} for key, task in tasks.items()]
