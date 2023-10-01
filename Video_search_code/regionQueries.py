
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
from findIndex import find_index
import pickle as pkl
import math
from sklearn.cluster import KMeans


def l2_norm(lis, ind):
    sum = 0
    for i in ind:
        sum = sum + lis[i] ** 2
    return math.sqrt(sum)

def dot(lis1, lis2, ind):
    sum = 0
    for i in ind:
        sum = sum +  lis1[i]*lis2[i]
    
    return sum

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

queries = find_index(["friends_0000003478.jpeg","friends_0000006178.jpeg", "friends_0000005771.jpeg", "friends_0000000378.jpeg"])


#kmeans = pkl.load(open("Model.pkl", "rb"))
clus_centres = kmeans.cluster_centers_

for x in queries:

    imname = framesdir + fnames[x][:-4]

    fname = siftdir + fnames[x]
        
    mat = scipy.io.loadmat(fname)

    im = plt.imread(imname)
    plt.title("Select a region", color = "red", fontsize = "20", fontfamily = "serif")

    pl.imshow(im)
    MyROI = roipoly(roicolor='r')
    Ind = MyROI.getIdx(im, mat['positions'])

    visual_words= []

    for i in range(0, 300):
    
        fname = siftdir + fnames[i]

        mat = scipy.io.loadmat(fname)
        
        if i == x:
            feats = mat['descriptors'][Ind, :]
        else:
            feats = mat['descriptors']
        
        if(feats.shape[0] == 0):
            l = [0 for i in range(k)]
        else: 
            c = kmeans.predict(feats)
            l = c.tolist()
        visual_words.append(l)
        # visual_words = np.vstack((visual_words, c))

    frequency_vectors = []
    
    for img_visual_words in visual_words:
    # create a frequency vector for each image
        img_frequency_vector = np.zeros(k)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)

    l = frequency_vectors[x]

    #storng the cluster centres for which at least one feature centre is present in query region
    nonzerocls = [m for m in range(k) if l[m] > 0]

    lnorm = l2_norm(l, nonzerocls)
    s = list(range(len(frequency_vectors))) #similarity scores

    print("Showing 5 most similar images: ")
    for i in range(len(frequency_vectors)):
        if l2_norm(frequency_vectors[i], nonzerocls) == 0:
            s[i]= 0
        else:
            s[i] = dot(l, frequency_vectors[i], nonzerocls)/(l2_norm(frequency_vectors[i], nonzerocls)*lnorm)

    # Getting indices of N = 6 maximum values
    x = np.argsort(s)[::-1][:6]

    for j in range(1, len(x)):
        i = x[j]
        fname = siftdir + fnames[i]
            
        #read the associated image
        imname = framesdir + fnames[i][:-4]
       
        im = plt.imread(imname)
        print("Similarity score ", s[i])

        # display the image
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.imshow(im)
        plt.show()

