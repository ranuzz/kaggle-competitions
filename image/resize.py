import numpy as np
from PIL import Image


def resize_np2np(img_arr, size_tuple):
    img=Image.fromarray(img_arr)
    out_img = img.resize(size_tuple)
    return np.array(out_img)
