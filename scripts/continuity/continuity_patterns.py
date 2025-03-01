# Created by jing at 01.03.25

from scripts.utils.encode_utils import create_tasks_v2, create_tasks_v3

from scripts.continuity.util_one_split_n import non_overlap_one_split_n
from scripts.continuity.util_two_splines import non_overlap_two_splines
from scripts.continuity.util_a_splines import non_overlap_a_splines
from scripts.continuity.util_u_splines import non_overlap_u_splines
from scripts.continuity.util_x_feature_splines import feature_continuity_x_splines
size_list = ["s", "m", "l"]
# Define task functions dynamically
tasks = {}

tasks.update(create_tasks_v3(non_overlap_one_split_n, ["shape", "color", "size", "count"], range(2,3), size_list))
tasks.update(create_tasks_v3(non_overlap_two_splines, ["shape", "color", "size", "count"], range(2, 3), size_list))
tasks.update(create_tasks_v3(non_overlap_a_splines, ["shape", "color", "size", "count"], range(2, 3), size_list))
tasks.update(create_tasks_v3(non_overlap_u_splines, ["shape", "color", "size", "count"], range(2, 3), size_list))
tasks.update(create_tasks_v3(feature_continuity_x_splines, ["shape", "color", "size", "count"], range(2, 3), size_list))

# Convert tasks to pattern dictionary
pattern_dicts = [{"name": key, "module": task} for key, task in tasks.items()]
