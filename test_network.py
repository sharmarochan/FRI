# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()

# ap.add_argument("-m", "--model", required=True, help="path to trained model model")
# ap.add_argument("-i", "--image", required=True, help="path to input image")
# args = vars(ap.parse_args())

# load the image
image = cv2.imread("/Users/rochansharma/Desktop/extracted_files/model/test_images/Elephant_agu_0_0_115.jpg")
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (256, 256))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model('animal.model')

# classify the input image
# (Barking_deer, Chital, Elephant) = model.predict(image)[0]
proba = model.predict(image)
max_proba = np.max(proba)

# label, = np.where(proba == max_proba)
label = str(np.argmax(proba))

print ("proba , label: ",max_proba)


# build the label
'''
if (prediction > Chital and  prediction > Elephant):
	label = "Barking_deer" 
	proba = Barking_deer

elif (prediction > Barking_deer and  prediction > Elephant):
	label = "Chital" 
	proba = Chital

elif (prediction > Barking_deer and  prediction > Chital):
	label = "Elephant" 
	proba = Elephant

else:
	print("None of Barking_deer, Elephant,Chital")
'''
print("proba", proba)
# label = "{}: {:.2f}%".format(label, proba * 100)



# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)



