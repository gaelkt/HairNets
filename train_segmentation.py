#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:50:26 2019

@author: gael
"""


"""
Created on Fri May 10 13:24:22 2019

@author: g84120421
"""
import gc
gc.collect()

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%matplotlib inline


from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape

from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

print('Packet imported successfully')

import sys
sys.path.insert(0, 'libs/')
from curliqfunctions import loading_training_faces_masks, visualize_face_mask, plot_sample
from curliqnet import get_unet
# Set some parameters
im_width = 224
im_height = 224
border = 5
number_channel = 1
train_images_folder = "datasets/224/hair_training/hair_training/"
mask_images_folder = "datasets/224/hair_segment/hair_segment/"


#We load images and mask from the dataset LBW
#The data have been already preprocessed by another script to have 128x128 gray image
#And to extract only hair segments
X, y = loading_training_faces_masks(train_images_folder, mask_images_folder)

# Split train and valid (90% and 10%)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
print('Shape training set X_train is ', np.shape(X_train))
print('Shape training set label y_train is ', np.shape(y_train))
print('Shape validation set X_valid is ', np.shape(X_valid))
print('Shape validation set y_valid is ', np.shape(y_valid))


# Visualize any randome image along with the mask   
visualize_face_mask(X_train, y_train)
visualize_face_mask(X_train, y_train)
visualize_face_mask(X_train, y_train)


print('starting with Unet')
##### Convolutional Neural Network For Hair Segmentation
input_img = Input((im_height, im_width, 3), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])


# Summary
#model.summary()


callbacks = [
    EarlyStopping(patience=4, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('weights/weights_224.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


results = model.fit(X_train, y_train, batch_size=32, epochs=25, callbacks=callbacks,\
                    validation_data=(X_valid, y_valid))	





model.load_weights('weights/weights_224.h5')

# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate(X_valid, y_valid, verbose=1)
	


# Predict on train, val and test
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)


# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

# We plot the results

plot_sample(X_valid, y_valid, preds_val, preds_val_t)

plot_sample(X_valid, y_valid, preds_val, preds_val_t)

plot_sample(X_valid, y_valid, preds_val, preds_val_t)




