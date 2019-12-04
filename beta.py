import numpy as np
from cv2 import cv2

I =  cv2.imread("saber.jpg",cv2.IMREAD_UNCHANGED)
x,y = I.shape
j = x
k = y
temp_image = np.zeros((x,y))
 
Ix = np.zeros((j,k))
for count1 in range(0,j):
    for count2 in range(0,k):
        if( (count2==1) or (count2==k) ):
            Ix[count1,count2]=0
        else:
            Ix[count1,count2]=0.5*(I[count1,count2+1]-I[count1,count2-1])
 
Iy = np.zeros((j,k))
for count1 in range(0,j):
    for count2  in range(0,k):
            if( (count1==1) or (count1==j) ):
                Iy[count1,count2]=0
            else:
                Iy[count1,count2]=0.5*(I[count1+1,count2]-I[count1-1,count2])

Ixy = np.zeros((j,k))
for count1 in range(0,j):
    for count2  in range(0,k):
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
    beta = np.array([I11,I21,I12,I22,Ix11,Ix21,Ix12,Ix22,Iy11,Iy21,Iy12, Iy22, Ixy11, Ixy21, Ixy12, Ixy22])
        