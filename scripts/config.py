# Created by jing at 25.02.25
import os
from pathlib import Path
import matplotlib

root = Path(__file__).parents[1]

# settings
num_samples = 5
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

# -------------- llm path -----------------------
llm_path = data/"llm_pretrained"
if not os.path.exists(llm_path):
    os.makedirs(llm_path)
# -------------- scripts path -----------------------
scripts = root / 'scripts'
if not os.path.exists(scripts):
    os.mkdir(scripts)
