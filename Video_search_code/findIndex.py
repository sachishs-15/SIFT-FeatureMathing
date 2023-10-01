import glob

framesdir = '/Users/sachish/Desktop/AGV/TaskRound/Task2/Files/frames/'
siftdir = '/Users/sachish/Desktop/AGV/TaskRound/Task2/Files/sift/'

fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

imnames = []
for i in range(len(fnames)):

    imname = fnames[i][:-4]
    imnames.append(imname)

def find_index(arg):
    indexes = []
    for el in arg:
        ind = imnames.index(el)
        indexes.append(ind)
    
    return indexes

#print (find_index(["friends_0000000060.jpeg"]))
