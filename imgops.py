import numpy as np
from cv2 import cv2

def threshold(img,threshhold):
    copy = img.copy()
    wi,le = img.shape
    for i in range(0,wi):
        for j in range(0,le):
            if copy[i,j] > threshhold:
                copy[i,j] = 255
            else:
                copy[i,j] = 0
    return copy
