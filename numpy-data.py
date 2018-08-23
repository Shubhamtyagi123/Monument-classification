from imutils import paths
import numpy as np
import argparse
import random
import cv2 as cv
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to imput dataset")
args = vars(ap.parse_args())

IMAGE_DIMS = (96, 96, 3)

data = []
labels = []

print ("[+] loading dataset images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	image = cv.imread(imagePath)
	image = cv.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image = gray.flatten()
	data.append(image)

	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

data = np.array(data, dtype="float")/255.0
np.save('info.npy', data)
print(len(data))

labels = np.array(labels)
np.save('lb.npy', labels)
print(len(labels))