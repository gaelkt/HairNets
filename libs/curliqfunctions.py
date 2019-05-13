# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:35:49 2019

@author: Gael
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.style.use("ggplot")
#%matplotlib inline
from skimage.transform import resize
from keras.preprocessing.image import img_to_array, load_img


number_channel = 3
im_width = 224
im_height = 224
border = 5

def loading_training_faces_masks(train_images_folder, mask_images_folder):
    #This function load training data (faces and mask, 128x128)
    ids = next(os.walk(train_images_folder))[2] # list of names all images in the given path
    print("No. of train images = ", len(ids))
    X = np.zeros((len(ids), im_height, im_width, number_channel), dtype=np.float32)
    y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    
    for n in range(len(ids)):
        img = load_img(train_images_folder+ids[n], grayscale=False)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_height, im_width, number_channel), mode = 'constant', preserve_range = True)
        # Load masks
        mask = img_to_array(load_img(mask_images_folder+ids[n], grayscale=True))
        mask = resize(mask, (im_width, im_width, 1), mode = 'constant', preserve_range = True)
        # Save images
        X[n] = x_img/255.0
        y[n] = mask/255.0
        
    print('Data Loaded')
    
    return X, y



# Visualize any random image along with the mask
def visualize_face_mask(X, y, ix=None):
    
    if ix is None:
        ix = random.randint(0, len(X))
    
    has_mask = y[ix].max() > 0 # hair segment if there or not


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 15))

    ax1.imshow(X[ix], interpolation = 'bilinear', vmin=0, vmax=1)
    if has_mask: # Hair is detected 
        # draw a boundary(contour) in the original image separating hair from non hair areas
        ax1.contour(y[ix].squeeze(), colors = 'r', linewidths = 8, levels = [0.5])
        
    ax1.set_title('ORIGINAL FACE')

    ax2.imshow(y[ix].squeeze(), cmap = 'gray', interpolation = 'bilinear')
    ax2.set_title('GROUND TRUTH HAIR MASK')
    
def plot_sample(X, y, preds, binary_preds, ix=None):
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('GRAY FACE')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('GT MASK')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Mask predicted')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Mask Predicted Binary');
    
    
def load_type_images(path_gray, path_rgb):
    #Load images from the assigments in rgb and gray
    #A previous script has already resized the images to 128x128 and create gray version of images
    
    ids_gray = next(os.walk(path_gray))[2] # list of names of all images
    print("No. of curl images = ", len(ids_gray))
    
    X_gray = np.zeros((len(ids_gray), im_height, im_width, 1), dtype=np.float32)
    X_rgb = np.zeros((len(ids_gray), im_height, im_width, number_channel), dtype=np.float32)
    X_name = []
    
    for n in range(len(ids_gray)):
        if (ids_gray[n] == '._Pete_Beaudrault_0001.jpg'):
            print('error')
        else:
            img = load_img(path_gray+ids_gray[n], grayscale=True)
            img_rgb = load_img(path_rgb+ids_gray[n], grayscale=False)
        
            x_img = img_to_array(img)
            x_img_rgb = img_to_array(img_rgb)
        
            x_img = resize(x_img, (im_height, im_width, number_channel), mode = 'constant', preserve_range = True)
            x_img_rgb = resize(x_img_rgb, (im_height, im_width, number_channel), mode = 'constant', preserve_range = True)
        
            X_gray[n] = x_img/255.0
            X_rgb[n] = x_img_rgb/1.0
            X_name.append(ids_gray[n])

    return X_gray, X_rgb, X_name

def plot_sample_curl(Ximage, preds, binary_preds, ix=None):
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(Ximage))

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(Ximage[ix], vmin=0, vmax=1)
    ax[0].set_title('ORIGINAL IMAGE')

    ax[1].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    ax[1].set_title('HAIR SEGMENT PREDICTED')
    
    ax[2].imshow(binary_preds[ix], vmin=0, vmax=1)
    ax[2].set_title('HAIR SEGMENT BINARY');
    
def hair_extract(X, binary_mask):
    X_ = 1.0*X
    X_rgb_segment = X_
    for image in range(len(X)):
        for channel in range(3):
            mask = binary_mask[image, :, :,0]
            sample = X_[image, :, :, channel]
            X_rgb_segment[image, :, :, channel] = np.multiply(sample, mask)
    return X_rgb_segment

def save_hair_segment(Ximage, Xname, folder_final):
    
    for i in range(len(Ximage)):
        cv2.imwrite(folder_final+Xname[i], Ximage[i])
    