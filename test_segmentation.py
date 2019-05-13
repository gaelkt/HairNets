# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:41:51 2019

@author: Gael
"""

import gc
gc.collect()

import os
import random
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%matplotlib inline


from skimage.transform import resize
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
from curliqfunctions import loading_training_faces_masks, visualize_face_mask, plot_sample_curl, load_type_images, hair_extract
from curliqfunctions import save_hair_segment
from curliqnet import get_unet
# Set some parameters
im_width = 128
im_height = 128
border = 5
number_channel = 1


type_3a_rgb = "datasets/segmentation/Type_RGB/type_a/"
type_3a_gray = "datasets/segmentation/Type_Gray/type_a/"

type_3b_rgb = "datasets/segmentation/Type_RGB/type_b/"
type_3b_gray = "datasets/segmentation/Type_Gray/type_b/"

type_3c_rgb = "datasets/segmentation/Type_RGB/type_c/"
type_3c_gray = "datasets/segmentation/Type_Gray/type_c/"

# Reading the images gray and rgb
X_gray_a, X_rgb_a, X_name_a = load_type_images(type_3a_gray, type_3a_rgb)
X_gray_b, X_rgb_b, X_name_b = load_type_images(type_3b_gray, type_3b_rgb)
X_gray_c, X_rgb_c, X_name_c = load_type_images(type_3c_gray, type_3c_rgb)

print('starting with Unet')
##### Convolutional Neural Network For Hair Segmentation
input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)




model.load_weights('weights/weights.h5')

# Predict hair segment
preds_hair_segment_a = model.predict(X_gray_a, verbose=1)
preds_hair_segment_b = model.predict(X_gray_b, verbose=1)
preds_hair_segment_c = model.predict(X_gray_c, verbose=1)

# Threshold for binary hair segment
preds_hair_segment_binay_a = (preds_hair_segment_a > 0.25).astype(np.uint8)
preds_hair_segment_binay_b = (preds_hair_segment_b > 0.25).astype(np.uint8)
preds_hair_segment_binay_c = (preds_hair_segment_c > 0.25).astype(np.uint8)


print("Prediction Finished")

X_rgb_segment_a = hair_extract(X_rgb_a, preds_hair_segment_binay_a)
X_rgb_segment_b = hair_extract(X_rgb_b, preds_hair_segment_binay_b)
X_rgb_segment_c = hair_extract(X_rgb_c, preds_hair_segment_binay_c)


# Save hair segment
folder_save_a = "datasets/segmentation/results/type_a/"
folder_save_b = "datasets/segmentation/results/type_b/"
folder_save_c = "datasets/segmentation/results/type_c/"

save_hair_segment(X_rgb_segment_a, X_name_a, folder_save_a)
save_hair_segment(X_rgb_segment_b, X_name_b, folder_save_b)
save_hair_segment(X_rgb_segment_c, X_name_c, folder_save_c)

plot_sample_curl(X_rgb_a/255, preds_hair_segment_a, X_rgb_segment_a/255)

plot_sample_curl(X_rgb_b/255, preds_hair_segment_b, X_rgb_segment_b/255)

plot_sample_curl(X_rgb_c/255, preds_hair_segment_c, X_rgb_segment_c/255)













