import os
import json
import time

import numba as nb
from numba import typed, types
import numpy as np

from io_func import load_image, save_image

print(os.getcwd())

with open("./config_my.json", "r") as f:
    labels = json.load(f)

colors = {tuple(label['color']): i for i, label in enumerate(labels)}

labels_path = '.\\training\\v2.0\\labels'
out_path = './labels_encoded'
os.makedirs(out_path, exist_ok=True)

labels_all = os.listdir(labels_path)
labels_processed = os.listdir(out_path)

colors_nb = typed.Dict.empty(key_type=types.UniTuple(types.int64, 3), value_type=types.int64)
for key, value in colors.items():
    colors_nb[key] = value

@nb.njit(parallel=True)
def encode_image(colors, image):
    encoded_image = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

    for i in nb.prange(image.shape[0]):
        for j in nb.prange(image.shape[1]):
            color = (image[i, j][0], image[i, j][1], image[i, j][2])
            encoded_image[i, j] = colors[color]
  
    return encoded_image
    
start = time.time()
batch_time = time.time()
for i, im in enumerate([label for label in labels_all if label not in labels_processed]):
    im_path = os.path.join(labels_path, im)
    out_im_path = os.path.join(out_path, im)
    
    im = load_image(im_path)
    im = encode_image(colors_nb, im)
    save_image(out_im_path, im)
    
    if i % 50 == 0:
        print(f"Processed {i} images. Batch time: {batch_time - time.time()}. Time elapsed: {time.time() - start} seconds.")
        batch_time = time.time()