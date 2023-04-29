import os
import json
import time

from numba import typed, types

from io_func import load_image, save_image
from read_config import read_config
from encoder_func import encode_image

cfg = read_config()
label_config_path = cfg['LABEL_CONFIG_PATH']
labels_path = cfg['LABELS_PATH']
out_path = cfg['OUTPUT_PATH']

os.makedirs(out_path, exist_ok=True)

labels_all = os.listdir(labels_path)
labels_processed = os.listdir(out_path)

with open(label_config_path, "r") as f:
    labels = json.load(f)

# In my case, the colors are stored in the color field of the label dictionary
colors = {tuple(label['color']): i for i, label in enumerate(labels)}
# Numba did not accept the dictionary as it is, so I had to convert it to a typed dictionary
colors_nb = typed.Dict.empty(key_type=types.UniTuple(types.int64, 3), value_type=types.int64)
for key, value in colors.items():
    colors_nb[key] = value

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