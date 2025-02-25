# Created by jing at 25.02.25

from scripts.proximity.util_grid_objs import non_overlap_grid
from scripts.proximity.util_red_triangle import non_overlap_red_triange

""" 
p: positive
s: size

"""


def task01(p): return non_overlap_red_triange(0.05, p, 1)


def task02(p): return non_overlap_red_triange(0.05, p, 2)


def task03(p): return non_overlap_red_triange(0.05, p, 3)


def task04(p): return non_overlap_grid(0.05, p, 2)
