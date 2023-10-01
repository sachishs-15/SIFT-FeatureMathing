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

# specific frame dir and siftdir
framesdir = '/Users/sachish/Desktop/AGV/TaskRound/Task2/Files/frames/'
siftdir = '/Users/sachish/Desktop/AGV/TaskRound/Task2/Files/sift/'

fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

fname = siftdir + fnames[0]
mat = scipy.io.loadmat(fname)
    
feats = mat['descriptors']

#building a stack for training the kmeans model
stackx = feats
for i in range(1, 100):
    fname = siftdir + fnames[i]
    
    mat = scipy.io.loadmat(fname)
    
    feats = mat['descriptors']
    stackx = np.vstack((stackx, feats))

# number of cluster centres
k = 1500
kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(stackx)

pkl.dump(KMeans, open('Model.pkl', 'wb'))

# kmeans = pkl.load(open("Model.pkl", "rb"))

# getting the centres of clusters of the kmeans model
clus_centres = kmeans.cluster_centers_

ls = []
for i in range(0, 15):
    fname = siftdir + fnames[i]

    mat = scipy.io.loadmat(fname)
    
    feats = mat['descriptors']
    
    c = kmeans.predict(feats)
    
    # list containing cluster centres corresponding to all descriptors in an image
    l = c.tolist()
    ls = ls + l

freqvw = [x for x in ls if ls.count(x) >= 30]

#taking two clusters as reference for visual words
vw1 = freqvw[0]
vw2 = freqvw[20] 

fw = [] #first word
sw = [] #second word

for i in range(0, 15):
    fname = siftdir + fnames[i]

    mat = scipy.io.loadmat(fname)
    numfeats = mat['descriptors'].shape[0]
    
    feats = mat['descriptors']
    coners = displaySIFTPatches(mat['positions'], mat['scales'], mat['orients'])
    
    c = kmeans.predict(feats)

    imname = framesdir + fnames[i][:-4]
    im = plt.imread(imname)

    # display the image and its SIFT features drawn as squares
    fig=plt.figure()
    bx=fig.add_subplot(111)
    bx.imshow(im)
    coners = displaySIFTPatches(mat['positions'], mat['scales'], mat['orients'])


    for j in range(len(c)) :
        if c[j] == vw1:
            fw.append(getPatchFromSIFTParameters(mat['positions'][j,:], mat['scales'][j], mat['orients'][j], rgb2gray(im)))
            bx.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g', linestyle='-', linewidth=1)
            bx.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g', linestyle='-', linewidth=1)
            bx.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g', linestyle='-', linewidth=1)
            bx.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g', linestyle='-', linewidth=1)

        elif c[j] == vw2:
            sw.append(getPatchFromSIFTParameters(mat['positions'][j,:], mat['scales'][j], mat['orients'][j], rgb2gray(im)))
            bx.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='r', linestyle='-', linewidth=1)
            bx.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='r', linestyle='-', linewidth=1)
            bx.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='r', linestyle='-', linewidth=1)
            bx.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='r', linestyle='-', linewidth=1)
        
    bx.set_xlim(0, im.shape[1])
    bx.set_ylim(0, im.shape[0])
    plt.gca().invert_yaxis()
    plt.show()  

fig, ax = plt.subplots(nrows=5, ncols=5)   
x = 0
    
#displaying 25 sift fatches correspoing to 1st word
for row in ax:
    for col in row:
        col.imshow(fw[x],  cmap = cm.Greys_r)
        x+=1

plt.suptitle("Visual Word 1")

plt.show()
        
fig, ax = plt.subplots(nrows=5, ncols=5)   
plt.suptitle("Visual Word 2")
x = 0


#displaying 25 sift fatches correspoing to 2nd word
for row in ax:
    for col in row:
        if(x>=len(sw)):
            break
        col.imshow(sw[x],  cmap = cm.Greys_r)
        x+=1

plt.show()
        


