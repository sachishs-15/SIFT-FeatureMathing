
import numpy as np
import cv2
import scipy.io
import glob
from scipy import misc
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from skimage.color import rgb2gray
import matplotlib.cm as cm
import pylab as pl
import math

# load that file
fname = '/Users/sachish/Desktop/AGV/TaskRound/Task2/Files/Video_search_code/twoFrameData.mat'
mat = scipy.io.loadmat(fname)

numfeats1 = mat['descriptors1'].shape[0]
numfeats2 = mat['descriptors2'].shape[0]
 
    
#read the associated images

im1 = mat['im1']
im2 = mat['im2']


# Select a subset of the features in first image using polygon drawing. 
print ('Use the mouse to draw a polygon, right click to end it')
pl.imshow(im1)
MyROI = roipoly(roicolor='r')

# Getting polygon Coordinates
xc = MyROI.allxpoints
yc = MyROI.allypoints

xc.append(xc[0])
yc.append(yc[0])

Ind = MyROI.getIdx(im1, mat['positions1'])

# Ind contains the indices of the SIFT features whose centers fall
# within the selected region of interest.
# Note that these indices apply to the *rows* of 'descriptors' and
# 'positions', as well as the entries of 'scales' and 'orients'

fig, (ax, bx) = plt.subplots(1, 2)
ax.imshow(im1)
ax.set_title("Region selected")
ax.plot(xc, yc, ls='-', linewidth=2, color='red')

desc2 = mat['descriptors2']
desc1 = mat['descriptors1']
ls = []

# RATIO TEST FOR FEATURE MATCHING ________________________________

# setting a threshold for matching
thresh = 0.6

matches = []

for i in range(numfeats2):
    ls = []
    for j in range(len(Ind)):
        sum = 0
        for x in range(128):
            sum += (desc1[j][x] - desc2[i][x])**2
        
        ls.append(math.sqrt(sum))

    ls.sort()

    if(ls[0]/ls[1] < thresh):
        matches.append(i);   
 
#____________________________________________________________________
bx.imshow(im2)

print(len(matches))

coners2 = displaySIFTPatches(mat['positions2'][matches,:], mat['scales2'][matches,:], mat['orients2'][matches,:])

# SIFT descriptor matching using euclidean distance
for j in range(len(matches)):

    #diplaying SIFT patch
    bx.plot([coners2[j][0][1], coners2[j][1][1]], [coners2[j][0][0], coners2[j][1][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners2[j][1][1], coners2[j][2][1]], [coners2[j][1][0], coners2[j][2][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners2[j][2][1], coners2[j][3][1]], [coners2[j][2][0], coners2[j][3][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners2[j][3][1], coners2[j][0][1]], [coners2[j][3][0], coners2[j][0][0]], color='g', linestyle='-', linewidth=1)


bx.set_title("Matched SIFT Features")
bx.set_xlim(0, im2.shape[1])
bx.set_ylim(0, im2.shape[0]) 

bx.invert_yaxis()

plt.show()    