import cv2
import numpy as np


def extract_connected_region(mask, x, y):
    h, w = mask.shape
    class_id = mask[y, x]

    class_mask = (mask == class_id).astype(np.uint8)
    flood = class_mask.copy() * 255
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, flood_mask, seedPoint=(x, y), newVal=128)
    region_mask = (flood == 128).astype(np.uint8)
    return region_mask
