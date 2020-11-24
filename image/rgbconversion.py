import numpy as np
import matplotlib.image as mpimg
import imghdr
from PIL import Image
import os


def rgb_path2nparray(path):
    if not os.path.isfile(path):
        return None
    try:
        hdr = imghdr.what(path)
        if hdr == "jpeg":
            img = mpimg.imread(path)
            return img
        pil_img = Image.open(path)
        rgb_img = pil_img.convert('RGB')
        return np.array(rgb_img)
    except OSError:
        return None
    except Exception as e:
        print(e)
        print("Unknown exception during RGB convert")
        return None


def rgb_path2path(srcpath, dstpath, overwrite=False):
    if not os.path.isfile(srcpath):
        return 1
    if os.path.isfile(dstpath) and not overwrite:
        return 0
    try:
        hdr = imghdr.what(srcpath)
        if hdr == "jpeg":
            img = Image.open(srcpath)
            img.save(dstpath)
            return 0
        pil_img = Image.open(srcpath)
        rgb_img = pil_img.convert('RGB')
        rgb_img.save(dstpath)
        return 0
    except OSError:
        return 1
    except Exception as e:
        print(e)
        print("Unknown exception during RGB convert")
        return 1
