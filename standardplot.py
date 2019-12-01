import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2

#######INTERPOLATIONS######
def nearest_neighbor(points_3d,point_2d):
    x,y = point_2d
    for p in points_3d:
        px,py,p_color = p
        if np.round(x)==px and np.round(y)==py:
            return p_color
    raise Exception("I know python!")

def bilinear(points_3d,point_2d):#[(0,0),(0,1),(1,0),(1,1)]
    p00 = points_3d[0]
    p01 = points_3d[1]
    p10 = points_3d[2]
    p11 = points_3d[3]
    p00_x,p00_y,p00_color = p00
    p01_color = p01[2]
    p10_color = p10[2]
    p11_x,p11_y,p11_color = p11

    x,y=point_2d

    ##delta_x=1   #p11_x-p00_x
    ##delta_y=1   #p11_y-p00_y

    I_x0 = (p11_x-x)*p00_color + (x-p00_x)*p01_color ##/delta_x
    I_x1 = (p11_x-x)*p10_color + (x-p00_x)*p11_color##delta_x

    I_y = (p11_y-y)*I_x0 + (y-p00_y)*I_x1##delta_y
    return I_y



def bicubic(points_3d,point_2d):
    print()

def interpolation(kind,points_3d,point_2d):
    if(kind=='nearest_neighbor'):
        return nearest_neighbor(points_3d,point_2d)
    elif(kind=='bilinear'):
        return bilinear(points_3d,point_2d)
    elif(kind=='bicubic'):
        return bicubic(points_3d,point_2d)



###IMAGE READ###
img = cv2.imread("saber.jpg",cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("saber.jpg",cv2.IMREAD_UNCHANGED)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

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
    #change_circle_in_img_2(event)
    fig.canvas.draw()

def change_circle_in_img_2(event):
    global img2,circle
 

print(bilinear([(14,20,91),(14,21,210),(15,20,162),(15,21,95)],(14.5,20.2)))

##https://stackoverflow.com/questions/52365190/blur-a-specific-part-of-an-image

fig.canvas.mpl_connect('button_press_event', move_circle)

plt.show()



# def getBicPixelChannel(img,x,y,channel):
# if (x < img.shape[1]) and (y < img.shape[0]):
#     return img[y,x,channel]

# return 0


# def Bicubic(img, rate):
# new_w = int(math.ceil(float(img.shape[1]) * rate))
# new_h = int(math.ceil(float(img.shape[0]) * rate))

# new_img = np.zeros((new_w, new_h, 3))

# x_rate = float(img.shape[1]) / new_img.shape[1]
# y_rate = float(img.shape[0]) / new_img.shape[0]

# C = np.zeros(5)

# for hi in range(new_img.shape[0]):
#     for wi in range(new_img.shape[1]):

#         x_int = int(wi * x_rate)
#         y_int = int(hi * y_rate)

#         dx = x_rate * wi - x_int
#         dy = y_rate * hi - y_int

#         for channel in range(new_img.shape[2]):
#             for jj in range(0,4):
#                 o_y = y_int - 1 + jj
#                 a0 = getBicPixelChannel(img,x_int,o_y, channel)
#                 d0 = getBicPixelChannel(img,x_int - 1,o_y, channel) - a0
#                 d2 = getBicPixelChannel(img,x_int + 1,o_y, channel) - a0
#                 d3 = getBicPixelChannel(img,x_int + 2,o_y, channel) - a0

#                 a1 = -1./3 * d0 + d2 - 1./6 * d3
#                 a2 = 1./2 * d0 + 1./2 * d2
#                 a3 = -1./6 * d0 - 1./2 * d2 + 1./6 * d3
#                 C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx

#             d0 = C[0] - C[1]
#             d2 = C[2] - C[1]
#             d3 = C[3] - C[1]
#             a0 = C[1]
#             a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
#             a2 = 1. / 2 * d0 + 1. / 2 * d2
#             a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
#             new_img[hi, wi, channel] = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy

# return new_img