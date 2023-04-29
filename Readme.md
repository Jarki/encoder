# Semantic segmentation label encoder

## Overview

I had a dataset for a semantic segmentation task, but the masks were not encoded in any way, just simple rgb values, which is not optimal for trying to predict to which class a pixel belongs. So, i wrote this simple script that encodes all the pixels in an image with the corresponding label. As Python is pretty slow, I used numba library to make the script faster.

## Usage

1. Clone the repository.
2. Install the requirements.txt
3. Change settings in config.json
4. Tweak the code if needed - the encoder function expects an array of dictionaries of type {tuple: class_index}, so make sure to convert the colors accordingly

## config.json

LABEL_CONFIG_PATH - path to a json file which contains color indices (which color corresponds to which class)
LABELS_PATH - folder from which the labels are taken
OUTPUT_PATH - folder, to which processed labels are output