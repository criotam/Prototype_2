# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:09:15 2018

@author: Anuj
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), padding = "same", input_shape = (64, 64, 3), activation = "relu"))
classifier.add(Convolution2D(32, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))

classifier.add(Convolution2D(32, (3,3), padding = "same", activation = "relu"))
classifier.add(Convolution2D(32, (3,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 1, activation = "sigmoid"))

classifier.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'binary_crossentropy')

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/Users/Anuj/Desktop/Python codes/Dataset_Faces/training_set',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = "binary")

test_set = test_datagen.flow_from_directory('/Users/Anuj/Desktop/Python codes/Dataset_Faces/test_set',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = "binary")

classifier.fit_generator(training_set,
                         samples_per_epoch = 3850,
                         nb_epoch = 1,
                         validation_data = test_set,
                         nb_val_samples = 1023)

import numpy as np
import cv2
from keras.preprocessing import image
import time

# initialize the camera
cam = cv2.VideoCapture(0)   # 0 -> index of camera
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)
time.sleep(0.5)
s, img = cam.read()
if s:    # frame captured without any error
    cv2.namedWindow("Camera ON", 1)
    cv2.imshow("Camera ON",img)
    cv2.waitKey(0)
    t1 = time.time()
    cv2.imwrite("/Users/Anuj/Desktop/Python codes/Dataset_faces/Captured_images/Image_test.png",img) #save image

test_image = image.load_img('/Users/Anuj/Desktop/Python codes/Dataset_faces/Captured_images/Image_test.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Anuj'
else:
    prediction = 'Ananth'

cam.release()
cv2.destroyAllWindows()    
    
