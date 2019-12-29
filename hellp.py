import numpy as np
import math
from cv2 import cv2


def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy

def getAngle(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    x = x2-x1
    y = y2-y1
    ans = np.arctan(y/x)
    if(x2<x1):
        if (y2<y1):
            ans = -np.pi+ans
        else:
            ans = np.pi + ans
    return ans

def rotate_on_pivot(pivot,xy):
    xp,yp = pivot
    x,y = xy
    rotated = rotate_origin_only((x-xp,y-yp),getAngle(pivot,xy))
    return xp+rotated[0],yp+rotated[1]

p = np.array([1,5])
print(np.array([0,1])+(2,3)+(1,3))

p1 = (2,2)
p2 = (-1,-2)
print(rotate_on_pivot(p1,p2))
p1_=np.array([0,0])
p2_=np.array([1,1])
p3_=np.array([0,10])
d=np.cross(p2_-p1_,p3_-p1_)/np.linalg.norm(p2_-p1_)
print(d)
c1 = (0,0)
c2 = (3,4)
if (np.sqrt((c1[0]-c2[0])**2 + (c1[1]+c2[1])**2)>5):
    print("bad")
else:
    print("its fine")

img = cv2.imread("saber.jpg",cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

new_img = img.copy()
for i in range (0,img.shape[0]):
    for j in range (0,img.shape[1]):
        if(i==30):
            new_img[j,i] = 0
cv2.imshow("name", new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

