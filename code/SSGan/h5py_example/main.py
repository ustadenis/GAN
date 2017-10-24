import numpy as np
import h5py
from scipy import ndimage as nd
from os import listdir
from os.path import isfile, join
from scipy.misc import imresize

h5pyFilename = 'data.hy'
labelsFilename = 'train_perfect_preds.txt'
idsFilename = 'id.txt'
imagesPath = 'cars'
matFilename = 'cars_train_annos.mat'

#labels_count = 8144
labels_count = 196
labels_pattern = np.zeros(labels_count)


def get_label_one_hot_vector(label): 
    one_hot_vector = labels_pattern.copy() 
    one_hot_vector[label - 1] = 1 
    return one_hot_vector

def get_filenames(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


hf = h5py.File(h5pyFilename, 'w')

with open(labelsFilename) as f:
    labels = f.readlines()

filenames = get_filenames(imagesPath)

for index, file in enumerate(filenames):
    #if index == 500:
        #break
    if index % 100 == 0: 
        print(str(index) + " images have been processed") 
    im = nd.imread(imagesPath + "/" + file)
    resized = imresize(im, (32, 32), interp='bicubic') 
    if resized.shape == (32,32,3):
        g = hf.create_group(str(index))
        g['image'] = resized
        g['label'] = get_label_one_hot_vector(int(labels[index]))
    else: 
        print ("wrong data!" + str(index))

hf.close()
