import numpy as np
import cv2

def get_depth_scale(depth):
    if isinstance(depth, np.ndarray):
        dtype = depth.dtype
    else:
        dtype = depth

    if dtype == float or dtype == np.float32:
        scale = 1
    elif dtype == np.uint16:
        scale = 0.001
    else:
        raise RuntimeError(f"Unknown depth type {dtype}")

    return scale