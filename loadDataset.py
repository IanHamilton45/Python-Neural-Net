import urllib
import gzip
import os
import numpy as np
import urllib.request

TRAINING_IMAGE_LIMIT = 2500
TESTING_IMAGE_LIMIT = 9000


def load_dataset():

    # Downloads the specified file from image database
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading ", filename)
        urllib.request.urlretrieve(source + filename, filename)

    def loadImages(filename):
        # Check if the file exists
        if not os.path.exists(filename):
            download(filename)

        # open the zip file containing images
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

            # data contains integer bytes in a 1 dimensional array
            # Needs to be an array of images - Since all the images are monochrome, they
            # use 1 channel and are 28 x 28 pixels each.
            # number of images (unknown), number of channels, width, height
            data = data.reshape(-1, 1, 28, 28)

            # e.g
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 1st row of pixels
            # [0,0,0,0,0,0,0,0,0,0,0,0.9,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0] 2st row of pixels
            # 28 x 28 pixels, each element representing 0 - 1 black value, 0 being white, 1 being black
            # will convert the byte value to a float32 in the range of 0 - 1
            return data / np.float32(256)

    def loadLabels(filename):
        # Check if the file exists
        if not os.path.exists(filename):
            download(filename)

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
            # data contains an array of integers corresponding to each image label

        return data

    trainImages = loadImages('train-images-idx3-ubyte.gz')
    trainLabels = loadLabels('train-labels-idx1-ubyte.gz')
    testImages = loadImages('t10k-images-idx3-ubyte.gz')
    testLabels = loadLabels('t10k-labels-idx1-ubyte.gz')

    trainImages = np.delete(trainImages, np.s_[TRAINING_IMAGE_LIMIT:], 0)
    trainLabels = np.delete(trainLabels, np.s_[TRAINING_IMAGE_LIMIT:], 0)

    testImages = np.delete(testImages, np.s_[TESTING_IMAGE_LIMIT:], 0)
    testLabels = np.delete(testLabels, np.s_[TESTING_IMAGE_LIMIT:], 0)

    return trainImages, trainLabels, testImages, testLabels
