import numpy as np
import h5py
from scipy import ndimage as nd
from os import listdir
from os.path import isfile, join

h5pyFilename = 'data.hy'
labelsFilename = 'train_perfect_preds.txt'
idsFilename = 'id.txt'
imagesPath = 'cars'

labels_count = 8144
labels_pattern = np.zeros(labels_count)


def get_label_one_hot_vector(label):
    one_hot_vector = labels_pattern.copy()
    one_hot_vector[label] = 1
    return one_hot_vector

def get_filenames(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


hf = h5py.File(h5pyFilename, 'w')
with open(labelsFilename) as f:
    labels = f.readlines()

filenames = get_filenames(imagesPath)

for index, file in enumerate(filenames):
    im = nd.imread(imagesPath + "/" + file)
    g = hf.create_group(str(index))
    g['image'] = im
    # g.create_dataset('label', data=get_label_one_hot_vector(int(labels[index])))
    g['label'] = get_label_one_hot_vector(index)

    if index == 10:
        break

hf.close()
