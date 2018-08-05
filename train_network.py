# USAGE
# python train_network.py --dataset images --model santa_not_santa.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from tempfile import TemporaryFile

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
# ap.add_argument("-m", "--model", required=True, help="path to output model")
# ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
# args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
# print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them

imagePaths = sorted(list(paths.list_images("images")))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
count = 0
for imagePath in imagePaths:
	count = count+1
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)

	try:
		image = cv2.resize(image, (256, 256))
	except:
		continue

	image = img_to_array(image)
	data.append(image)
	# extract the class label from the image path and update the
	# labels list

	if par1 == "jackal":
		label = 0
	if par1 == "Barking_deer":
		label = 1
	if par1 == "Porcupine":
		label = 2
	if par1 == "Gaur":
		label = 3
	if par1 == "Sambar":
		label = 4
	if par1 == "Elephant":
		label = 5
	if par1 == "Chital":
		label = 6
	if par1 == "Wild_pig":
		label = 7
	if par1 == "Jungle_Cat":
		label = 8
	if par1 == "Hare":
		label = 9
	if par1 == "sloth_bear":
		label = 10
	
	labels.append(label)
	print("image count:", count, par1, label)

	# if count == 500:
	# 	break
	
	labels.append(label)
	print("image count:", count, imagePath, label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("train_test_split...")
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, random_state=42)


# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes = 11)
testY = to_categorical(testY, num_classes = 11)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=256, height=256, depth=3, classes=11)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save('animal.model')

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('Animal_plot')

