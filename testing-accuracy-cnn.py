import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import pickle
import cv2 as cv
import imutils
import os
import random
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
ap.add_argument("-l", "--labelbin", required=True, help="path to label binarizer")
ap.add_argument("-i", "--images", required=True, help="path to the input image")
ap.add_argument("-p", "--plot", type=str, default="acc_cnn1.png", help="path to output accuracy plot")
args = vars(ap.parse_args())

imageFolder = list(paths.list_images(args["images"]))
random.shuffle(imageFolder)
indices = []
x = []

for images in imageFolder:

	image = cv.imread(images)
	output = image.copy()

	image = cv.resize(image, (96,96))
	image = image.astype("float")/255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	print ("[+] loading network model...")
	model = load_model(args["model"])
	lb = pickle.loads(open(args["labelbin"], "rb").read())

	print ("[+] classifying input image...")
	pred = model.predict(image)[0]
	idx = np.argmax(pred)
	indices.append(pred[idx]*100)

avg = 0
for i in range(0,len(indices)):
	avg += indices[i]
	x.append(i)

avg = avg/len(indices)
print avg

plt.style.use("ggplot")
plt.figure()

plt.plot(x, indices, label="Average accuracy : {:.2f}%".format(avg))
plt.xlabel("No. of Steps")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.savefig(args["plot"])

