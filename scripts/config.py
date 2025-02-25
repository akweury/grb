# Created by jing at 25.02.25
import os
from pathlib import Path
import matplotlib

root = Path(__file__).parents[0]

# settings
num_samples = 10
img_width = 1024

# -------------------- shape settings --------------------
bk_shapes = ["none", "triangle", "square", "circle"]

# ---------------------- color settings ------------------
color_matplotlib = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in
                    list(matplotlib.colors.cnames.items())}
color_matplotlib.pop("darkslategray")
color_matplotlib.pop("lightslategray")
color_matplotlib.pop("black")
color_matplotlib.pop("darkgray")

color_dict_rgb2name = {value: key for key, value in color_matplotlib.items()}
color_large = [k for k, v in list(color_matplotlib.items())]
color_large_exclude_gray = [item for item in color_large if item != "lightgray" and item != "lightgrey"]

# -------------- data path -----------------------
data = root / 'data'
if not os.path.exists(data):
    os.makedirs(data)
raw_patterns = data / 'raw_patterns'
if not os.path.exists(raw_patterns):
    os.makedirs(raw_patterns)

proximity_patterns = raw_patterns / 'proximity'
if not os.path.exists(proximity_patterns):
    os.makedirs(proximity_patterns)
proximity_patterns_train = proximity_patterns / "train"
if not os.path.exists(proximity_patterns_train):
    os.makedirs(proximity_patterns_train)
proximity_patterns_test = proximity_patterns / "test"
if not os.path.exists(proximity_patterns_test):
    os.makedirs(proximity_patterns_test)

# -------------- scripts path -----------------------
scripts = root / 'scripts'
if not os.path.exists(scripts):
    os.mkdir(scripts)

proximity_script_path = scripts / "proximity"
if not os.path.exists(proximity_script_path):
    os.mkdir(proximity_script_path)
