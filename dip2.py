import imgops as ops
import numpy as np
from cv2 import cv2
import random as rd

img = cv2.imread("saber.jpg",cv2.IMREAD_GRAYSCALE)

while True:
    cv2.imshow("thresh",ops.threshold(img,rd.randint(0,255)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()