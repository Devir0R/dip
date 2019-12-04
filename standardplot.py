import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2


###IMAGE READ###
img = cv2.imread("saber.jpg",cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("saber.jpg",cv2.IMREAD_UNCHANGED)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


###utilities###

M_dim_left = np.array([[1,0,0,0],[0,0,1,0],[-3,3,-2,1],[2,-2,1,1]])
M_dim_right = np.array([[1,0,-3,2],[0,0,3,-2],[0,1,-2 ,1],[0,0,-1,1]])

def getAngle(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    x = x2-x1
    y = y2-y1
    add_degrees = 0
    if(x2<x1):
        if(y2<y1):
            add_degrees = 90
        else:
            add_degrees = 180
    print(np.arctan(y/x))
    return 360*( np.arctan(y/x)/(2*np.pi)) + add_degrees

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
    x = np.round(x)
    y = np.round(y)
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
            Ix[count1,count2]=0.5*(I[count1,count2+1]-I[count1,count2-1])
 
Iy = np.zeros((j,k))
for count1 in range(0,j-1):
    for count2  in range(0,k-1):
            if( (count1==1) or (count1==j) ):
                Iy[count1,count2]=0
            else:
                Iy[count1,count2]=0.5*(I[count1+1,count2]-I[count1-1,count2])

Ixy = np.zeros((j,k))
for count1 in range(0,j-1):
    for count2  in range(0,k-1):
        if( (count1==1) or (count1==j) or (count2==1) or (count2==k) ):                
            Ixy[count1,count2]=0
        else:
            Ixy[count1,count2]=0.25*((I[count1,count2]+I[count1,count2]) - (I[count1,count2]+I[count1,count2]))

def bicubic(point_2d,img):
    count1,count2 = point_2d
    I=img
    I11_index = [1+np.floor(count1),1+np.floor(count2)]
    I21_index = [1+np.floor(count1),1+np.ceil(count2)]
    I12_index = [1+np.ceil(count1),1+np.floor(count2)]
    I22_index = [1+np.ceil(count1),1+np.ceil(count2)]
    #%Calculate the four nearest function values
    I11 = I[I11_index[1],I11_index[2]]
    I21 = I[I21_index[1],I21_index[2]]
    I12 = I[I12_index[1],I12_index[2]]
    I22 = I[I22_index[1],I22_index[2]]
    #%Calculate the four nearest horizontal derivatives
    Ix11 = Ix[I11_index[1],I11_index[2]]
    Ix21 = Ix[I21_index[1],I21_index[2]]
    Ix12 = Ix[I12_index[1],I12_index[2]]
    Ix22 = Ix[I22_index[1],I22_index[2]]
    #%Calculate the four nearest vertical derivatives
    Iy11 = Iy[I11_index[1],I11_index[2]]
    Iy21 = Iy[I21_index[1],I21_index[2]]
    Iy12 = Iy[I12_index[1],I12_index[2]]
    Iy22 = Iy[I22_index[1],I22_index[2]]
    #%Calculate the four nearest cross derivatives
    Ixy11 = Ixy[I11_index[1],I11_index[2]]
    Ixy21 = Ixy[I21_index[1],I21_index[2]]
    Ixy12 = Ixy[I12_index[1],I12_index[2]]
    Ixy22 = Ixy[I22_index[1],I22_index[2]]
    #%Create our beta-vector
    #beta = np.array([I11,I21,I12,I22,Ix11,Ix21,Ix12,Ix22,Iy11,Iy21,Iy12, Iy22, Ixy11, Ixy21, Ixy12, Ixy22])
    beta2 = np.array(
        [
            [I11,I12,Iy11,Iy12],
            [I21,I22,Iy21,Iy22],
            [Ix11,Ix12,Ixy11, Ixy12],
            [Ix21, Ix22, Ixy21, Ixy22]
        ])
    coef = np.matmul(np.matmul(M_dim_left,beta2),M_dim_right)
    x_array = np.array([1,count1,count1*count1,count1*count1*count1])
    y_array = np.array([1,count2,count2*count2,count2*count2*count2])
    ans = np.matmul(np.matmul(x_array,coef),y_array)
    return ans
        

###### end of bicubic #########
def interpolation(kind,point_2d,img):
    if(kind=='nearest_neighbor'):
        return nearest_neighbor(point_2d,img)
    elif(kind=='bilinear'):
        return bilinear(point_2d,img)
    elif(kind=='bicubic'):
        return bicubic(point_2d,img)



###IMAGE CIRCLE POSITION###
y,x = img.shape
x=int(x/2)
y=int(y/2)
point = (x,y)
radius = 5

###IMAGE PLOT###
fig,ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(img2)
circle = plt.Circle(point, radius=radius, color='r',fill=False)
circle2 = plt.Circle(point, radius=radius, color='r',fill=False)
ax[0].add_artist(circle)
ax[1].add_artist(circle2)
ax[0].set_title('Click to move the circle')

def move_circle(event):
    global fig,circle,img,circle2
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        ('double' if event.dblclick else 'single', event.button,
        event.x, event.y, event.xdata, event.ydata))
    print(event.inaxes)
    print(event.inaxes==ax[0])
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
        ax[1].imshow(rotate_bound(img,getAngle(circle.center,circle2.center),circle.center))
    #change_circle_in_img_2(event)
    fig.canvas.draw()

def change_circle_in_img_2(event):
    global img2,circle
 

print(bilinear((14.5,20.2),img))

##https://stackoverflow.com/questions/52365190/blur-a-specific-part-of-an-image
#https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

fig.canvas.mpl_connect('button_press_event', move_circle)

plt.show()



def getBicPixel(img,x,y):
  if (x < img.shape[1]) and (y < img.shape[0]):
    return img[y,x]
  return 0


def Bicubic(img, rate):
  new_w = int(np.ceil(float(img.shape[1]) * rate))
  new_h = int(np.ceil(float(img.shape[0]) * rate))

  new_img = np.zeros((new_w, new_h, 3))

  x_rate = float(img.shape[1]) / img.shape[1]
  y_rate = float(img.shape[0]) / img.shape[0]

  C = np.zeros(5)

  for hi in range(img.shape[0]):
    for wi in range(img.shape[1]):
        x_int = int(wi * x_rate)
        y_int = int(hi * y_rate)
        dx = x_rate * wi - x_int
        dy = y_rate * hi - y_int
        for jj in range(0,4):
            o_y = y_int - 1 + jj
            a0 = getBicPixel(img,x_int,o_y)
            d0 = getBicPixel(img,x_int - 1,o_y) - a0
            d2 = getBicPixel(img,x_int + 1,o_y) - a0
            d3 = getBicPixel(img,x_int + 2,o_y) - a0
            a1 = -1./3 * d0 + d2 - 1./6 * d3
            a2 = 1./2 * d0 + 1./2 * d2
            a3 = -1./6 * d0 - 1./2 * d2 + 1./6 * d3
            C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx
        d0 = C[0] - C[1]
        d2 = C[2] - C[1]
        d3 = C[3] - C[1]
        a0 = C[1]
        a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
        a2 = 1. / 2 * d0 + 1. / 2 * d2
        a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
        new_img[hi, wi] = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy
  return new_img
