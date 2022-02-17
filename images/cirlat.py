#!/usr/bin/env python

import cv2
import numpy as np

def cirlat(size, radius, spacing, padding, color):
    img = np.zeros(size, dtype=np.uint8)
    for yy in range(padding[0]+radius, size[0], spacing):
        for xx in range(padding[1]+radius, size[1], spacing):        
            cv2.circle(img, (xx,yy), radius, color, thickness=-1)

    return img
