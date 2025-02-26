# Created by jing at 25.02.25


import sys
import os
import numpy as np
import cv2

from scripts import config
from scripts.utils import file_utils, visual_utils
from scripts.proximity import prox_patterns

# pattern_principle = "proximity"
#
# # List of proximity patterns with their associated modules
# pattern_dicts = [
#     {"name": "red_triangle", "module": prox_patterns.task01},
#     {"name": "red_triangle_two", "module": prox_patterns.task02},
#     {"name": "red_triangle_three", "module": prox_patterns.task03},
#     {"name": "grid_2", "module": prox_patterns.task04},
#
#     # Add more patterns here
# ]

#
# def gen_image(objs):
#     """
#     Generate an image from a list of objects.
#     :param objects: List of objects, each defined by a dict with keys x, y, size, rgb_color, shape.
#     :param img_size: Size of the output image (img_size x img_size).
#     :return: Generated image as a NumPy array.
#     """
#     img_size = config.img_width
#     image = np.zeros((img_size, img_size, 3), dtype=np.uint8) + config.color_matplotlib["lightgray"]
#     image = image.astype(np.uint8)
#     for obj in objs:
#         x = int(obj["x"] * img_size)
#         y = int(obj["y"] * img_size)
#         size = int(obj["size"] * img_size)
#         color = (obj["color_r"], obj["color_g"], obj["color_b"])
#
#         if obj["shape"] == "circle":
#             cv2.circle(image, (x, y), size // 2, color, -1)
#         elif obj["shape"] == "square":
#             top_left = (x - size // 2, y - size // 2)
#             bottom_right = (x + size // 2, y + size // 2)
#             cv2.rectangle(image, top_left, bottom_right, color, -1)
#         elif obj["shape"] == "triangle":
#             half_size = size // 2
#             points = np.array([
#                 [x, y - half_size],
#                 [x - half_size, y + half_size],
#                 [x + half_size, y + half_size]
#             ])
#             cv2.fillPoly(image, [points], color)
#     visual_utils.van(image, "test.png")
#     return image
#
#
#
# def save_patterns(pattern, save_path, num_samples, is_positive):
#     for example_i in range(num_samples):
#         img_path = save_path / f"{example_i:05d}.png"
#         data_path = save_path / f"{example_i:05d}.json"
#         objs = pattern["module"](is_positive)
#         # encode symbolic object tensors
#         image = gen_image(objs)
#         file_utils.save_img(img_path, data_path, pattern_principle, objs, image)


# pattern_counter = 0
# for pattern in pattern_dicts:
#     pattern_counter += 1
#     module = pattern["module"]
#     pattern_name = f"{pattern_counter:03d}_" + pattern["name"]
#     # Run the save_patterns function if it exists in the script
#
#     print(f"{pattern_counter}/{len(pattern_dicts)} Generating patterns using {pattern_name}...")
#     os.makedirs(config.proximity_patterns_train / pattern_name, exist_ok=True)
#     os.makedirs(config.proximity_patterns_train / pattern_name / "positive", exist_ok=True)
#     os.makedirs(config.proximity_patterns_train / pattern_name / "negative", exist_ok=True)
#
#     os.makedirs(config.proximity_patterns_test / pattern_name, exist_ok=True)
#     os.makedirs(config.proximity_patterns_test / pattern_name / "positive", exist_ok=True)
#     os.makedirs(config.proximity_patterns_test / pattern_name / "negative", exist_ok=True)
#
#     save_patterns(pattern, config.proximity_patterns_train / pattern_name / "positive",
#                   num_samples=config.num_samples, is_positive=True)
#     save_patterns(pattern, config.proximity_patterns_train / pattern_name / "negative",
#                   num_samples=config.num_samples, is_positive=False)
#     save_patterns(pattern, config.proximity_patterns_test / pattern_name / "positive",
#                   num_samples=config.num_samples, is_positive=True)
#     save_patterns(pattern, config.proximity_patterns_test / pattern_name / "negative",
#                   num_samples=config.num_samples, is_positive=False)
#
# print("Proximity pattern generation complete.")
