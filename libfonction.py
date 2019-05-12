import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from math import exp, pi, sqrt
from cmath import phase


j = complex(0,1)

#to reverse a 2D signal
def reverse2D(u):
    v = np.zeros(u.shape, dtype=complex)
    for y in range(u.shape[1]):
        for x in range(u.shape[0]):
            v[x][y] = u[(u.shape[0]-1-x)][(u.shape[1]-1-y)]
    return v
    
    
#to grey out an image by averaging its 3 color
def grey(arr):
    if len(arr.shape) == 2:
        return arr
    (X,Y) = arr.shape[:2]
    arr = arr.astype(int)
    a = np.zeros((X,Y), dtype=complex)
    for x in range(X):
        for y in range(Y):
            a[x][y] = (arr[x][y][0] + arr[x][y][1] + arr[x][y][2])/3
    return a
    
#in order to plot the fourier transform 
def logOperator(fourier):
    highest = np.nanmax(fourier)
    print("highest:", highest)
    fourier = 255/np.log10(1+highest) * np.log10(1+fourier)
    return fourier.astype(int)

def linearNormalization(u):
    highest = np.nanmax(u)
    print("highest:", highest)
    u = (255/highest)*u
    return u
    
def linearNormalizationFilter(u):
    s = u.sum()
    return u/s
    
#shift 2D signal
def shift2D(arr):
    (X,Y) = arr.shape[:2]
    a = np.zeros((X,Y), dtype=complex)
    for x in range(X):
        for y in range(Y):
            a[(X//2 + x)%X][(Y//2 + y)%Y] = arr[x][y]
    return a

#image of shape "shape" of a normalised circle centered at (0,0) of radius radius 
def circle(shape, r): 
    (X,Y) = shape
    v = np.zeros((X,Y))
    c = 0
    for x in range(-X//2, X//2):
        for y in range(-Y//2, Y//2):
            if x**2+y**2 < r**2:
                c+=1
                v[x][y] = 1
    for x in range(-X//2, X//2):
        for y in range(-Y//2, Y//2):
            v[x][y] /= c
            #v[x][y] *= 255
    return v

#plot image in grey and its DFT
def plot(img):
    u = np.asarray(img)
    if not len(u.shape) == 2: #not grey
        u = grey(u)
    t = logOperator(abs(shift2D(np.fft.fft2(u))))
    img_grey = Image.fromarray(u.astype(np.uint8))
    img_dft = Image.fromarray(t.astype(np.uint8))
    
    (m,n) = u.shape
    # Display
    dpi = 20    
    
    plt.figure(figsize=(m/float(dpi),n/float(dpi)))
    plt.subplot(221)
    plt.imshow(img_grey)
    plt.title('Original')
    plt.axis('off');
    plt.subplot(222)
    plt.imshow(img_dft)
    plt.title('DFT')
    plt.axis('off');
    
#take a numpy array u and expand it with zeros in a array of shape (X,Y)
def addZeros(u, shp):
    (X,Y) = shp
    (x,y) = u.shape
    z1 = np.zeros((x,Y-y))
    z2 = np.zeros((X-x,Y))
    u = np.concatenate((u,z1), axis=1)
    u = np.concatenate((u,z2), axis=0)
    return u
    
    
    
def plotArray(u):
    im = Image.fromarray(u.astype(np.uint8))
    plt.imshow(im)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
