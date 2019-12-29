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


def mapFunc(f,arr):
    ret =  list(range(0,len(arr)))
    for i in range(0,len(arr)):
        ret[i] = f(arr[i])
    return ret

def filterFunc(f,arr):
    ret =  []
    for i in range(0,len(arr)):
        if f(arr[i]):
            ret.append(arr[i])
    return ret

def maxIndex(arr):
    max = arr[0]
    index = 0
    for i in range(1,len(arr)):
        if max < arr[i]:
            max = arr[i]
            index = i
    return index
    


img =  cv2.imread("saber.jpeg",cv2.IMREAD_UNCHANGED)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

threshhold = 180
ret, thresh = cv2.threshold(imgray,threshhold, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

final = thresh

bounding_boxes = mapFunc(lambda cnt: cv2.boundingRect(cnt),contours)
boxes_sizes =  mapFunc(lambda box: box[2]*box[3],bounding_boxes)
maxI = maxIndex(boxes_sizes)
del bounding_boxes[maxI]
del boxes_sizes[maxI]
del contours[maxI]


print("sizes")
print(list(boxes_sizes))
print("\nbounding boxes")
print(list(bounding_boxes))

bb=dict(zip(list(boxes_sizes),[list(boxes_sizes).count(i) for i in list(boxes_sizes)]))
print(sorted(list(bb)))

fig,ax = plt.subplots(1,2)
ax[0].imshow(img,'gray')
ax[1].imshow(final,'gray')
plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows() 

#plt.plot(range(0,256),calcHistogram(imgray))
#cv2.imshow("saber")
#sum = reduce((lambda x, y: x+y),calcHistogram(imgray))
#print(contours)

