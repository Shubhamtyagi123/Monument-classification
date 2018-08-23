from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import pickle
import cv2 as cv
import imutils
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
ap.add_argument("-l", "--labelbin", required=True, help="path to label binarizer")
ap.add_argument("-i", "--images", required=True, help="path to the input image")
args = vars(ap.parse_args())

imageFolder = sorted(list(paths.list_images(args["images"])))
random.shuffle(imageFolder)

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
	label = lb.classes_[idx]

	filename = images[images.rfind(os.path.sep) + 1:]
	correct = "Incorrect" if filename.rfind(label) != -1 else "Correct"

	label = "{}: {:.2f}% ({})".format(label, pred[idx] * 100, correct)
	output = imutils.resize(output, width=500)
	cv.putText(output, label, (10,25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

	print ("[INFO] {}".format(label))

	cv.imshow("Output", output)
	cv.waitKey(1000)

