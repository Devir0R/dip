import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
import math

###IMAGE READ###
img = cv2.imread("saber.jpg",cv2.IMREAD_GRAYSCALE)



###utilities###
def angle_between_vector(a,b,c):
    a = np.array([a[0],a[1]])
    b = np.array([b[0],b[1]])
    c = np.array([c[0],c[1]])
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

def distance(start,finish,point):
    s_x,s_y = start
    p1_=np.array([s_x,s_y])

    f_x,f_y = finish
    p2_=np.array([f_x,f_y])

    p_x,p_y = point
    p3_=np.array([p_x,p_y])
    d=np.cross(p2_-p1_,p3_-p1_)/np.linalg.norm(p2_-p1_)
    return d

def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy

def getAngleRadians(p1,p2):
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
    rotated = rotate_origin_only((x-xp,y-yp),getAngleRadians(pivot,xy))
    return xp+rotated[0],yp+rotated[1]

M = np.array([
    [1 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0 , 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-3, 3, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 2,-2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-2,-1, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 1, 1, 0, 0],
    [-3, 0, 3, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0,-2, 0,-1, 0],
    [ 9,-9,-9, 9, 6, 3,-6,-3, 6,-6, 3,-3, 4, 2, 2, 1],
    [-6, 6, 6,-6,-3,-3, 3, 3,-4, 4,-2, 2,-2,-2,-1,-1],
    [ 2, 0,-2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [-6, 6, 6,-6,-4,-2, 4, 2,-3, 3,-3, 3,-2,-1,-2,-1],
    [ 4,-4,-4, 4, 2, 2,-2,-2, 2,-2, 2,-2, 1, 1, 1, 1]
])

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
    return 360*( ans/(2*np.pi))-90

def rotate_bound(image, angle,pivot):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    cX, cY = pivot
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

#######INTERPOLATIONS######
def nearest_neighbor(point_2d,img):
    x,y = point_2d
    x = int(np.round(x))
    y = int(np.round(y))
    if x>0 and y>0 and x< img.shape[1] and y<img.shape[0]:
        return img[x,y]
    else:
        return 0

def bilinear(point_2d,img):#[(0,0),(0,1),(1,0),(1,1)]
    x,y = point_2d
    x = int(np.round(x))
    y = int(np.round(y))
    p00 = (x,y)
    ##p01 = (x,y+1)
    #p10 = (x+1,y)
    p11 = (x+1,y+1)
    p00_x,p00_y = p00
    p11_x,p11_y = p11

    ##delta_x=1   #p11_x-p00_x
    ##delta_y=1   #p11_y-p00_y

    I_x0 = (p11_x-x)*img[x,y] + (x-p00_x)*img[x,y+1] ##/delta_x
    I_x1 = (p11_x-x)*img[x+1,y] + (x-p00_x)*img[x+1,y+1]##delta_x

    I_y = (p11_y-y)*I_x0 + (y-p00_y)*I_x1##delta_y
    return I_y

#### bicubic interpolation stuff #######
I = img

x,y = I.shape
j = x
k = y
temp_image = np.zeros((x,y))
 
Ix = np.zeros((j,k))
for count1 in range(0,j-1):
    for count2 in range(0,k-1):
        if( (count2==1) or (count2==k) ):
            Ix[count1,count2]=0
        else:
            Ix[count1,count2]=int(0.5*(I[count1,count2+1]-I[count1,count2-1]))
 
Iy = np.zeros((j,k))
for count1 in range(0,j-1):
    for count2  in range(0,k-1):
            if( (count1==1) or (count1==j) ):
                Iy[count1,count2]=0
            else:
                Iy[count1,count2]=int(0.5*(I[count1+1,count2]-I[count1-1,count2]))

Ixy = np.zeros((j,k))
for count1 in range(0,j-1):
    for count2  in range(0,k-1):
        if( (count1==1) or (count1==j) or (count2==1) or (count2==k) ):                
            Ixy[count1,count2]=0
        else:
            Ixy[count1,count2]=int(0.25*((I[count1,count2]+I[count1,count2]) - (I[count1,count2]+I[count1,count2])))

def bicubic(point_2d,img):
    count1,count2 = point_2d
    I=img
    I11_index = [int(1+np.floor(count1)),int(1+np.floor(count2))]
    I21_index = [int(1+np.floor(count1)),int(1+np.ceil(count2))]
    I12_index = [int(1+np.ceil(count1)),int(1+np.floor(count2))]
    I22_index = [int(1+np.ceil(count1)),int(1+np.ceil(count2))]
    #%Calculate the four nearest function values
    I11 = I[I11_index[0],I11_index[1]]
    I21 = I[I21_index[0],I21_index[1]]
    I12 = I[I12_index[0],I12_index[1]]
    I22 = I[I22_index[0],I22_index[1]]
    #%Calculate the four nearest horizontal derivatives
    Ix11 = Ix[I11_index[0],I11_index[1]]
    Ix21 = Ix[I21_index[0],I21_index[1]]
    Ix12 = Ix[I12_index[0],I12_index[1]]
    Ix22 = Ix[I22_index[0],I22_index[1]]
    #%Calculate the four nearest vertical derivatives
    Iy11 = Iy[I11_index[0],I11_index[1]]
    Iy21 = Iy[I21_index[0],I21_index[1]]
    Iy12 = Iy[I12_index[0],I12_index[1]]
    Iy22 = Iy[I22_index[0],I22_index[1]]
    #%Calculate the four nearest cross derivatives
    Ixy11 = Ixy[I11_index[0],I11_index[1]]
    Ixy21 = Ixy[I21_index[0],I21_index[1]]
    Ixy12 = Ixy[I12_index[0],I12_index[1]]
    Ixy22 = Ixy[I22_index[0],I22_index[1]]
    #%Create our beta-vector
    beta = np.array([I11,I21,I12,I22,Ix11,Ix21,Ix12,Ix22,Iy11,Iy21,Iy12, Iy22, Ixy11, Ixy21, Ixy12, Ixy22])
    # beta2 = np.array(
    #     [
    #         [I11,I12,Iy11,Iy12],
    #         [I21,I22,Iy21,Iy22],
    #         [Ix11,Ix12,Ixy11, Ixy12],
    #         [Ix21, Ix22, Ixy21, Ixy22]
    #     ])
    coef = np.matmul(M,np.transpose(beta))
    a00,a10,a20,a30,a01,a11,a21,a31,a02,a12,a22,a32,a03,a13,a23,a33 = np.transpose(coef)
    coef_mat = np.transpose(np.array([
        [a00,a10,a20,a30],
        [a01,a11,a21,a31],
        [a02,a12,a22,a32],
        [a03,a13,a23,a33]
    ]))
    ans = np.polynomial.polynomial.polyval2d(count1,count2,coef_mat)
    if ans > 255:
        return 255
    elif ans<0:
        return 0
    return ans
        

###### end of bicubic #########
def interpolation(kind,img):
    if(kind=='nearest_neighbor'):
        return lambda point_2d : nearest_neighbor(point_2d,img)
    elif(kind=='bilinear'):
        return lambda point_2d :  bilinear(point_2d,img)
    elif(kind=='bicubic'):
        return lambda point_2d : bicubic(point_2d,img)



###IMAGE CIRCLE POSITION###
y,x = img.shape
x=int(x/2)
y=int(y/2)
point = (x,y)
radius = 5

###IMAGE PLOT###
fig,ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(img)
circle = plt.Circle(point, radius=radius, color='r',fill=False)
circle2 = plt.Circle(point, radius=radius, color='b',fill=False)
ax[0].add_artist(circle)
ax[1].add_artist(circle2)
ax[0].set_title('Click to move the circle')

def move_circle(event):
    global fig,circle,img,circle2
    if event.inaxes is None:
        return
    if event.inaxes==ax[0]:
        if event.button==3 or event.button==2:
            px,py=circle.center
            circle.radius = int(np.sqrt((event.xdata-px)**2+(event.ydata-py)**2))
            circle2.radius = circle.radius
        elif event.button==1:
            circle.center = event.xdata, event.ydata
    elif event.inaxes==ax[1]:
        circle2.center = event.xdata, event.ydata
        ax[1].imshow(change_circle_in_img_2(event,'bicubic'))
    fig.canvas.draw()

def change_circle_in_img_2(event,method):
    global circle,circle2,img
    c1 = circle.center
    c2 = circle2.center
    pow1 =  (c1[0]-c2[0])**2
    pow2 = (c1[1]-c2[1])**2
    sq = np.sqrt(pow1 + pow2)
    if (sq>2*circle.radius):
        print("bad")
        raise Exception("too small")
    interpolate = interpolation(method,img)
    new_img = img.copy()
    vect = (c2[0]-c1[0],c2[1]-c1[0])
    vect_norm = (vect/np.linalg.norm(vect))
    A = - circle.radius * vect_norm + circle.center
    B = np.array( [circle.center[0],circle.center[1]])
    C = np.array( [circle2.center[0],circle2.center[1]])
    D = circle.radius * vect_norm + circle2.center
    AB_to_AC = (A-C/np.linalg.norm(A-C))/(A-B/np.linalg.norm(A-B))
    BD_to_CD = (C-D/np.linalg.norm(C-D))/(B-D/np.linalg.norm(B-D))
    for i in range (0,img.shape[0]):
        for j in range (0,img.shape[1]):
            if ((np.sqrt((c1[0]-i)**2 + (c1[1]-j)**2)<circle.radius) or (np.sqrt((c2[0]-i)**2 + (c2[1]-j)**2)<circle2.radius)):
                d = distance(circle.center,circle2.center,(i,j))
                rate = 1-(d/circle.radius)
                target_point = np.array([i,j]) 
                if(np.abs( angle_between_vector((i,j),B,C))<np.pi/2):
                    target_point = target_point + (B-A)*AB_to_AC
                else:
                    target_point = target_point + (D-B)*BD_to_CD
                x = interpolate(target_point)
                new_img[i,j] = x
    img2 = cv2.imwrite("saber-bic.jpg",new_img)
    return new_img



 

#print(bilinear((14.5,20.2),img))

##https://stackoverflow.com/questions/52365190/blur-a-specific-part-of-an-image
#https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

fig.canvas.mpl_connect('button_press_event', move_circle)

plt.show()
