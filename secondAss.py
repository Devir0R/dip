import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

img =  cv2.imread("saber.jpeg",cv2.IMREAD_UNCHANGED)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dst = []
ret, thresh = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)
#hist = cv2.calcHist([imgray],[0],None,[256],[0,256])

fig,ax = plt.subplots(1,2)
ax[0].imshow(img,'gray')
ax[1].imshow(thresh,'gray')
plt.show() 
#cv2.imshow("saber")

#cv2.waitKey(0)
#cv2.destroyAllWindows() 

