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
k = 1000
kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(stackx)
pkl.dump(kmeans, open('Model.pkl', 'wb'))

#kmeans = pkl.load(open("Model.pkl", "rb"))

# getting the centres of clusters of the kmeans model
clus_centres = kmeans.cluster_centers_

visual_words= []

for i in range(0, 100):
    fname = siftdir + fnames[i]
    mat = scipy.io.loadmat(fname)
    
    feats = mat['descriptors']
    
    #to exclude on file which has no descriptors
    if(feats.shape[0] == 0):
        l = [0]
    
    else:
        c = kmeans.predict(feats)

        # list containing cluster centres corresponding to all descriptors in an image
        l = c.tolist()
    visual_words.append(l)

frequency_vectors = []


for img_visual_words in visual_words:

# create a frequency vector for each image
    img_frequency_vector = np.zeros(k)
    for word in img_visual_words:
        img_frequency_vector[word] += 1
    frequency_vectors.append(img_frequency_vector)

# selecting query images
query_img = ["friends_0000005042.jpeg", "friends_0000004821.jpeg", "friends_0000005986.jpeg"]

# finding the index of query image chosen in the sift directory
query_ind = find_index(query_img)

for j in range(len(query_ind)):

    print("Query ", j+1)
    x = query_ind[j]

    fname = siftdir + fnames[x]
        
    #read the query image
    imname = framesdir + fnames[x][:-4]
    im = plt.imread(imname)

    # display the image
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.imshow(im)
    ax.set_title("Query Image")
    plt.show()

    l = frequency_vectors[x]

    l_norm = norm(l)
    s = list(range(len(frequency_vectors))) #similarity scores

    print("Showing 5 most similar images: ")
    for i in range(len(frequency_vectors)):
        if norm(frequency_vectors[i]) == 1:
            s[i] = 0
        else:
            s[i] = np.dot(l, frequency_vectors[i])/(norm(frequency_vectors[i])*l_norm)

    # Getting indices of M = 6 maximum values where most similar image is the image itself
    M = 5
    xx = np.argsort(s)[::-1][:M+1]

    fig, axs = plt.subplots(1, M)

    for j in range(1, len(xx)):
        i = xx[j]
        fname = siftdir + fnames[i]
        
        #read the associated image
        imname = framesdir + fnames[i][:-4]
        im = plt.imread(imname)

        print("Similarity score ", s[i])

        axs[j-1].imshow(im)
        
    plt.suptitle("Most similar images")
    plt.show()

