#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from skimage.io import imread_collection, imshow
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.transform import hough_circle, hough_circle_peaks


def make_sequence(directory, color):
    path = 'train/' + directory + '/'
    names = [path + name for name in sorted(os.listdir(path))]
    data = imread_collection(names).concatenate()
    data = data[:, :, :, color].astype('float64')
    data -= np.mean(data)
    data /= np.std(data)
    data = np.gradient(data, axis=0)
    data = np.sum(data, axis=(1, 2)).ravel()
    return data


def draw_images(directory, label):
    path1 = 'train/' + directory + '/'
    path2 = '.' + directory + '.png'
    images = imread_collection([
        path1 + str(label).zfill(3) + path2,
        path1 + str(label + 1).zfill(3) + path2,
        path1 + str(label + 2).zfill(3) + path2
    ]).concatenate().astype('float64')
    [(plt.imshow(image.astype('uint8')), plt.show()) for image in images]


def draw_sequence(data, label, title):
    plt.plot(data)
    plt.title(title)
    patch = Rectangle((label - 6, data.min()), 12,
        data.max() - data.min(), fill=False, color='black')
    plt.axes().add_patch(patch)
    plt.show()


def draw_circles(directory):
    path = 'train/' + directory + '/'
    names = [path + name for name in sorted(os.listdir(path))]
    data = imread_collection(names).concatenate()

    for j, frame in enumerate(data[115:]):
        edge = canny(rgb2gray(frame), sigma=2, low_threshold=0.2)
        edge = (edge*255).astype('uint8')
        imshow(edge, cmap='gray'); plt.show()

        imshow(frame)
        hspace = hough_circle(edge, range(10, 30))
        accums, cx, cy, radii = hough_circle_peaks(
                hspace, range(10, 30), total_num_peaks=40)
        for x, y, r in zip(cx, cy, radii):
            patch = Circle((x, y), r, fill=True, color='black')
            plt.axes().add_patch(patch)
        plt.show()
        print(j)


if __name__=='__main__':
    labels, sequences = list(), list()
    count = '/' + str(len(os.listdir('train/'))) + ' samples'
    for j, directory in enumerate(sorted(os.listdir('train/'))):
        path = 'train/' + directory + '/'
        red = make_sequence(directory, 0)
        green = make_sequence(directory, 1)
        blue = make_sequence(directory, 2)
        sequences.append(np.vstack((red, green, blue)))
        print(str(j + 1) + count)
    np.save('train.npy', sequences)