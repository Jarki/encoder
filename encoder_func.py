import numba as nb
from numba import typed, types
import numpy as np


# numba function that converts colors into labels
@nb.njit(parallel=True)
def encode_image(colors, image):
    encoded_image = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

    for i in nb.prange(image.shape[0]):
        for j in nb.prange(image.shape[1]):
            color = (image[i, j][0], image[i, j][1], image[i, j][2])
            encoded_image[i, j] = colors[color]
  
    return encoded_image