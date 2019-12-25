import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from functools import reduce

def calcHistogram(image):
    hist = [0]*256
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            hist[image[i,j]] += 1
    return hist

def findThresh(hist):
    sum = reduce((lambda x, y: x+y),hist)



img =  cv2.imread("saber.jpeg",cv2.IMREAD_UNCHANGED)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

threshhold = 180
ret, thresh = cv2.threshold(imgray,threshhold, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

final = thresh

fig,ax = plt.subplots(1,2)
ax[0].imshow(img,'gray')
ax[1].imshow(final,'gray')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows() 

#plt.plot(range(0,256),calcHistogram(imgray))
#cv2.imshow("saber")
#sum = reduce((lambda x, y: x+y),calcHistogram(imgray))
#print(contours)

