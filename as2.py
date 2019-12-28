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


def getFirst(arr):
    return arr[0]
    

cnt_area_thresh = 29/227.5
imgray =  cv2.imread("saber.jpg",cv2.IMREAD_GRAYSCALE)
edited_img = imgray.copy()
#imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

threshhold = 180
ret, thresh = cv2.threshold(imgray,threshhold, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

final = thresh

bounding_boxes = list(mapFunc(lambda cnt: cv2.boundingRect(cnt),contours))
bounding_boxes.sort(key=getFirst)
boxes_sizes =  list(mapFunc(lambda box: box[2]*box[3],bounding_boxes))
wholePicIndex = maxIndex(boxes_sizes)
del bounding_boxes[wholePicIndex]
del boxes_sizes[wholePicIndex]
del contours[wholePicIndex]


sizes = mapFunc(lambda cnt: (cv2.contourArea(cnt),cv2.boundingRect(cnt)[0],cv2.boundingRect(cnt)[1]),contours)
sizes.sort(key=getFirst)
biggest_letter_size = sizes[len(sizes)-1][0]

flag_small_contours = mapFunc(lambda cnt: cv2.contourArea(cnt)/biggest_letter_size<cnt_area_thresh,contours)

for cnt_i in range(0,len(contours)):  
    if(flag_small_contours[cnt_i]):
        curr_cnt = contours[cnt_i]
        x,y,w,h = cv2.boundingRect(curr_cnt)
        for i in range(0,w):
            for j in range(0,h):
                if(cv2.pointPolygonTest(curr_cnt,(x+i,y+j),False)>=0):
                    edited_img[y+j,x+i] = 255
        continue

#print("boxes sizes")
#print(boxes_sizes)
#print("\nbounding boxes")
#print(bounding_boxes)
#print("\nsizes")
#print(list(sizes))
#print(cnt_area_thresh)
#bb=dict(zip(list(boxes_sizes),[list(boxes_sizes).count(i) for i in list(boxes_sizes)]))
#print(sorted(list(bb)))




#plt.plot(range(0,256),calcHistogram(imgray))
cv2.imshow("saber",edited_img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
#sum = reduce((lambda x, y: x+y),calcHistogram(imgray))
#print(contours)