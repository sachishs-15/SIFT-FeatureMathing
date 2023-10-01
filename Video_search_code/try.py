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
from sklearn.cluster import KMeans
import pickle as pkl
import statistics
from statistics import mode
from scipy.cluster.vq import vq
from numpy.linalg import norm
import random
from findIndex import find_index


# specific frame dir and siftdir
framesdir = '/Users/sachish/Desktop/AGV/TaskRound/Task2/Files/frames/'
siftdir = '/Users/sachish/Desktop/AGV/TaskRound/Task2/Files/sift/'

fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

for i in range(0, 300):
    #print(i)
    fname = siftdir + fnames[i]
        
    #read the associated image
    imname = framesdir + fnames[i][:-4]
    print(imname)
    im = plt.imread(imname)

    # display the image
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.imshow(im)
    plt.show()