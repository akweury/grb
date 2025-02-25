# Created by jing at 25.02.25

from scripts import config


def encode_objs(x, y, size, color, shape, line_width, solid):
    data = {"x": x,
            "y": y,
            "size": size,
            "color_name": config.color_large.index(color),
            "color_r": config.color_matplotlib[color][0],
            "color_g": config.color_matplotlib[color][1],
            "color_b": config.color_matplotlib[color][2],
            "shape": shape,
            "line_width": line_width,
            "solid": solid
            }
    return data
