# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:41:51 2019

@author: Gael
"""

import gc
gc.collect()

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%matplotlib inline

from keras.layers import Input
import sys
sys.path.insert(0, 'libs/')
from curliqfunctions import plot_sample_curl, load_type_images, hair_extract
from curliqfunctions import save_hair_segment
from curliqnet import get_unet

print('Packet imported successfully')


# Set some parameters
im_width = 224
im_height = 224
number_channel = 3
threshold_hair = 0.10 #Threshold for binarization

#Location of images
type_3a_rgb = "datasets/224/Type_RGB/type_a/"
type_3b_rgb = "datasets/224/Type_RGB/type_b/"
type_3c_rgb = "datasets/224/Type_RGB/type_c/"

# Reading the images rgb
X_rgb_a, X_name_a = load_type_images(type_3a_rgb)
X_rgb_b, X_name_b = load_type_images(type_3b_rgb)
X_rgb_c, X_name_c = load_type_images(type_3c_rgb)

print('Type A is ', np.shape(X_rgb_a))
print('Type B is ', np.shape(X_rgb_b))
print('Type C is ', np.shape(X_rgb_c))

print('starting with Unet')
##### Convolutional Neural Network For Hair Segmentation
input_img = Input((im_height, im_width, number_channel), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.load_weights('weights/weights_224.h5')

# Predict hair segment
preds_hair_segment_a = model.predict(X_rgb_a, verbose=1)
preds_hair_segment_b = model.predict(X_rgb_b, verbose=1)
preds_hair_segment_c = model.predict(X_rgb_c, verbose=1)

# Threshold for binary hair segment
preds_hair_segment_binay_a = (preds_hair_segment_a > threshold_hair).astype(np.uint8)
preds_hair_segment_binay_b = (preds_hair_segment_b > threshold_hair).astype(np.uint8)
preds_hair_segment_binay_c = (preds_hair_segment_c > threshold_hair).astype(np.uint8)

print("Prediction Finished")

# Extraction of pixels corresponding to hao
X_rgb_segment_a = hair_extract(X_rgb_a, preds_hair_segment_binay_a)
X_rgb_segment_b = hair_extract(X_rgb_b, preds_hair_segment_binay_b)
X_rgb_segment_c = hair_extract(X_rgb_c, preds_hair_segment_binay_c)

# Save hair segment
folder_save_a = "datasets/224/results/type_a/"
folder_save_b = "datasets/224/results/type_b/"
folder_save_c = "datasets/224/results/type_c/"

save_hair_segment(255*X_rgb_segment_a, X_name_a, folder_save_a)
save_hair_segment(255*X_rgb_segment_b, X_name_b, folder_save_b)
save_hair_segment(255*X_rgb_segment_c, X_name_c, folder_save_c)

plot_sample_curl(X_rgb_a, preds_hair_segment_a, X_rgb_segment_a)

plot_sample_curl(X_rgb_b, preds_hair_segment_b, X_rgb_segment_b)

plot_sample_curl(X_rgb_c, preds_hair_segment_c, X_rgb_segment_c)













