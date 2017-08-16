#!/usr/bin/env python3
import os
import numpy as np
from skimage.io import imread_collection


labels, sequences = list(), list()
count = '/' + str(len(os.listdir('train/'))) + ' samples'
for j, directory in enumerate(sorted(os.listdir('train/'))):
    path = 'train/' + directory + '/'
    names = [path + name for name in sorted(os.listdir(path))]
    data = imread_collection(names).concatenate()[:, :, :, (0, 1)]
    data = data.astype('float64')
    data -= np.mean(data); data /= np.std(data)
    data = np.gradient(data, axis=0)
    data = np.sum(data, axis=(1, 2))
    data = np.diff(data, axis=1)
    sequences.append(list(data.ravel()))
    name = 'akn.' + directory + '.left.avi'
    label = str(np.argmax(data)).zfill(5)
    labels.append((name, label))
    print(str(j + 1) + count)
np.savetxt('submit.txt', labels, fmt='%s')
np.save('train.npy', sequences)
