import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2


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
ax[0].add_artist(circle)
ax[0].set_title('Click to move the circle')

def move_circle(event):
    global fig,circle
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        ('double' if event.dblclick else 'single', event.button,
        event.x, event.y, event.xdata, event.ydata))
    if event.inaxes is None:
        return
    if event.button==3 or event.button==2:
        px,py=circle.center
        circle.radius = int(np.sqrt((event.xdata-px)**2+(event.ydata-py)**2))
    elif event.button==1:
        circle.center = event.xdata, event.ydata
    #change_circle_in_img_2(event)
    fig.canvas.draw()

def change_circle_in_img_2(event):
    global img2,circle
    

##https://stackoverflow.com/questions/52365190/blur-a-specific-part-of-an-image

fig.canvas.mpl_connect('button_press_event', move_circle)
plt.show()

