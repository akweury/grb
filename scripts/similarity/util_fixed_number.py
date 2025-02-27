# Created by jing at 26.02.25

import random
import numpy as np

from scripts import config
from scripts.utils.shape_utils import overlaps, overflow
from scripts.utils import pos_utils, encode_utils


def generate_group_objects(color, target_count, cluster_centers, image_size, diameter,
                           so, min_circles, max_circles, used_centers, fixed_props):
    """Generate objects for one group (of a given color) until target_count is reached."""
    group_objs = []
    count = 0
    shape = random.choice(config.bk_shapes[1:])
    while count < target_count:
        # Pick a random cluster center to start a new cluster.
        cluster_x, cluster_y = random.choice(cluster_centers)
        cluster_points = [(cluster_x, cluster_y)]
        used_centers.add((cluster_x, cluster_y))

        # Decide how many circles to generate in this cluster.
        num_circles = np.random.randint(min_circles, max_circles + 1)

        # Create additional circles in the cluster.
        for _ in range(num_circles - 1):
            if count >= target_count:
                break
            directions = [
                (diameter, 0), (-diameter, 0), (0, diameter), (0, -diameter),
                (diameter, diameter), (-diameter, -diameter), (diameter, -diameter), (-diameter, diameter)
            ]
            random.shuffle(directions)  # Try different directions at random
            for dx, dy in directions:
                new_x = cluster_points[-1][0] + dx
                new_y = cluster_points[-1][1] + dy
                # Check that the new circle is within bounds and not overlapping existing ones.
                if (0.05 < new_x < image_size[0] and 0.05 < new_y < image_size[1] and
                        all((new_x - cx) ** 2 + (new_y - cy) ** 2 >= diameter ** 2 for cx, cy in used_centers)):
                    cluster_points.append((new_x, new_y))
                    used_centers.add((new_x, new_y))


                    if "shape" in fixed_props:
                        shape = random.choice(config.bk_shapes[1:])

                    obj = encode_utils.encode_objs(x=new_x, y=new_y, size=so, color=color, shape=shape,
                                                   line_width=-1, solid=True)
                    group_objs.append(obj)
                    count += 1
                    break
    return group_objs


# def similarity_fixed_number(is_positive, obj_size, cluster_num, fixed_props, grid_size=3, min_circles=3, max_circles=5,
#                             diameter=0.08, image_size=(1, 1)):
#     """
#     Generate a scene by creating multiple groups.
#
#     Parameters:
#       - is_positive  : determines whether groups are equal-sized.
#       - obj_size     : size of the objects.
#       - cluster_num  : number of clusters (2, 3, or 4).
#       - fixed_props  : fixed properties for objects.
#       - grid_size    : grid dimensions for cluster centers.
#       - min_circles, max_circles: range for objects per cluster.
#       - diameter     : spacing for circle placement.
#       - image_size   : dimensions of the image.
#
#     Returns:
#       - List of kandinskyShape objects.
#     """
#     if cluster_num not in {2, 3, 4}:
#         raise ValueError("cluster_num must be 2, 3, or 4.")
#
#     base_count = random.randint({2: (10, 20), 3: (6, 12), 4: (5, 10)}[cluster_num][0],
#                                 {2: (10, 20), 3: (6, 12), 4: (5, 10)}[cluster_num][1])
#
#     def adjust_count(base):
#         return base if is_positive else max(1, base + random.randint(-5, 5))
#
#     colors = ["yellow", "blue", "red", "green"][:cluster_num]
#
#
#     if is_positive:
#         cluster_sizes = [base_count] * cluster_num
#     else:
#         if random.random() < 0.5:
#             cluster_num = random.choice([n for n in {2, 3, 4} if n != cluster_num])
#             cluster_sizes = [base_count] * cluster_num
#         else:
#             cluster_sizes = [random.randint(1, base_count + 5) for _ in range(cluster_num)]
#
#     group_configs = list(zip(colors, cluster_sizes))
#
#     grid_spacing = image_size[0] / (grid_size + 1)
#     cluster_centers = [(grid_spacing * (i + 1), grid_spacing * (j + 1))
#                        for i in range(grid_size) for j in range(grid_size)]
#     random.shuffle(cluster_centers)
#
#     used_centers = set()
#     objs = [
#         obj for color, count in group_configs
#         for obj in generate_group_objects(color, count, cluster_centers, image_size,
#                                           diameter, obj_size, min_circles, max_circles, used_centers, fixed_props)
#     ]
#
#     return objs



def similarity_fixed_number(is_positive, obj_size, cluster_num, fixed_props, quantity, grid_size=3, min_circles=3,
                            max_circles=5, diameter=0.08, image_size=(1, 1)):
    """
    Generate a scene by creating multiple groups.

    Parameters:
    - is_positive: Determines whether groups are equal-sized.
    - obj_size: Size of the objects.
    - cluster_num: Number of clusters (2, 3, or 4).
    - fixed_props: Fixed properties for objects.
    - quantity: Defines the total object quantity in the image ('s' for small, 'm' for medium, 'l' for large).
    - grid_size: Defines the number of cluster centers (grid_size x grid_size).
    - min_circles, max_circles: Controls the number of circles per cluster.
    - diameter: Spacing used for circle placement.
    - image_size: Tuple defining the dimensions of the image.

    Returns:
    - List of kandinskyShape objects.
    """
    if cluster_num not in {2, 3, 4}:
        raise ValueError("cluster_num must be 2, 3, or 4.")

    # Define object count ranges based on quantity parameter
    quantity_map = {"s": (2, 4), "m": (4, 10), "l": (20, 30)}
    base_count_range = quantity_map.get(quantity, (10, 20))  # Default to 'm' if invalid input
    base_count = random.randint(*base_count_range)

    def adjust_count(base):
        """Adjust the count of objects if the pattern is negative."""
        return base if is_positive else max(1, base + random.randint(-5, 5))

    colors = ["yellow", "blue", "red", "green"][:cluster_num]

    # Determine cluster sizes based on the pattern type
    if is_positive:
        cluster_sizes = [base_count] * cluster_num
    else:
        if random.random() < 0.5:
            cluster_num = random.choice([n for n in {2, 3, 4} if n != cluster_num])
            cluster_sizes = [base_count] * cluster_num
        else:
            cluster_sizes = [random.randint(1, base_count + 5) for _ in range(cluster_num)]

    group_configs = list(zip(colors, cluster_sizes))

    # Generate evenly spaced cluster centers within the grid
    grid_spacing = image_size[0] / (grid_size + 1)
    cluster_centers = [(grid_spacing * (i + 1), grid_spacing * (j + 1))
                       for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(cluster_centers)

    used_centers = set()
    objs = [
        obj for color, count in group_configs
        for obj in generate_group_objects(color, count, cluster_centers, image_size,
                                          diameter, obj_size, min_circles, max_circles, used_centers, fixed_props)
    ]

    return objs


#
# def similarity_fixed_number(is_positive, obj_size, cluster_num, fixed_props, grid_size=3, min_circles=3, max_circles=5,
#                             diameter=0.08, image_size=(1, 1)):
#     """
#     Generate a scene by creating multiple groups.
#
#     Parameters:
#       grid_size    : defines the number of cluster centers (grid_size x grid_size).
#       min_circles, max_circles: control the number of circles per cluster.
#       diameter     : the spacing used for circle placement.
#       image_size   : tuple defining the dimensions of the image.
#
#     Returns:
#       List of kandinskyShape objects.
#     """
#     if cluster_num == 2:
#         base_count = random.randint(10, 20)
#         if is_positive:
#             target_yellow = base_count
#             target_blue = base_count
#         else:
#             # When dtype is False, adjust one group’s count to be different.
#             if random.random() < 0.5:
#                 target_yellow = max(1, base_count + random.randint(-5, -1))
#             else:
#                 target_yellow = max(1, base_count + random.randint(1, 5))
#             target_blue = base_count
#         # Define two groups: yellow and blue.
#         group_configs = [("yellow", target_yellow), ("blue", target_blue)]
#     elif cluster_num == 3:
#         base_count = random.randint(6, 12)
#         if is_positive:
#             target_yellow = base_count
#             target_blue = base_count
#             target_red = base_count
#         else:
#             # When dtype is False, adjust one group’s count to be different.
#             if random.random() < 0.5:
#                 target_yellow = max(1, base_count + random.randint(-5, -1))
#             else:
#                 target_yellow = max(1, base_count + random.randint(1, 5))
#             target_blue = base_count
#             target_red = max(1, base_count + random.randint(-5, 5))
#         # Define two groups: yellow and blue.
#         group_configs = [("yellow", target_yellow), ("blue", target_blue), ("red", target_red)]
#     elif cluster_num == 4:
#         base_count = random.randint(5, 10)
#         if is_positive:
#             target_yellow = base_count
#             target_blue = base_count
#             target_red = base_count
#             target_green = base_count
#         else:
#             # When dtype is False, adjust one group’s count to be different.
#             if random.random() < 0.5:
#                 target_yellow = max(1, base_count + random.randint(-3, -1))
#             else:
#                 target_yellow = max(1, base_count + random.randint(1, 3))
#             target_blue = base_count
#             target_red = max(1, base_count + random.randint(-3, 3))
#             target_green = max(1, base_count + random.randint(1, 2))
#         # Define two groups: yellow and blue.
#         group_configs = [("yellow", target_yellow), ("blue", target_blue), ("red", target_red), ("green", target_green)]
#
#     else:
#         raise ValueError("g_num must be 2 or 3 or 4.")
#
#     used_centers = set()
#     objs = []
#     # Define evenly spaced cluster centers over the image.
#     grid_spacing = image_size[0] / (grid_size + 1)
#     cluster_centers = [(grid_spacing * (i + 1), grid_spacing * (j + 1))
#                        for i in range(grid_size) for j in range(grid_size)]
#     random.shuffle(cluster_centers)
#
#     # Generate objects for each group.
#     for color, target_count in group_configs:
#         objs.extend(generate_group_objects(color, target_count, cluster_centers,
#                                            image_size, diameter, obj_size,
#                                            min_circles, max_circles, used_centers))
#
#     return objs


def non_overlap_fixed_number(params, is_positive, cluster_num, quantity):
    obj_size = 0.05
    objs = similarity_fixed_number(is_positive, obj_size, cluster_num, params,quantity)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = similarity_fixed_number(is_positive, obj_size, cluster_num, params, quantity)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
