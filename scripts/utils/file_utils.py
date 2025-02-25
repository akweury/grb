# Created by jing at 25.02.25
import os
import json
from PIL import Image
import numpy as np
import torch


def img_padding(img, pad_width=2):
    if img.ndim == 3:
        pad_img = np.pad(img, pad_width=(
            (pad_width, pad_width), (pad_width, pad_width), (0, 0)),
                         constant_values=255)
    elif img.ndim == 2:
        pad_img = np.pad(img, pad_width=(
            (pad_width, pad_width), (pad_width, pad_width)),
                         constant_values=255)

    else:
        raise ValueError()

    return pad_img


def hconcat_imgs(img_list):
    padding_imgs = []
    for img in img_list:
        padding_imgs.append(img_padding(img))
    img = np.hstack(padding_imgs).astype(np.uint8)

    return img


def save_img(img_path, data_path, principle, img_data, image):

    # save image
    Image.fromarray(image).save(img_path)

    # save data
    data = {"principle": principle, "img_data": img_data}
    with open(data_path, 'w') as f:
        json.dump(data, f)
