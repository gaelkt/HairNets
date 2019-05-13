# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:25:49 2019

@author: Gael
"""
import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def augmentation(folder_reduced, folder_augmented):
    
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    
    allimages = os.listdir(folder_reduced)
    print('Total images', len(allimages))

    for j in len(allimages):
    
        img = load_img(folder_reduced+allimages[j]) # This is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=folder_augmented, save_prefix=allimages[j], save_format='jpeg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely

