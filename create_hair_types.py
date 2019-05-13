# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:39:52 2019

@author: Gael
"""
import cv2
import os

def create_rgb_hair_resize(folder_initial, folder_final_rgb):
    # This function read the images for the assigment and create 224x224x3 

    allimages = os.listdir(folder_initial)
    print('Total images', len(allimages))

    width,height = 224, 224
    dim = (width, height)

    for image in allimages:
        #RGB image
        img_rgb = cv2.imread(folder_initial+image)
        resized = cv2.resize(img_rgb, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(folder_final_rgb+image,resized)
    
    
# Folder where images for the assigments are located
folder_initial_a = "datasets/224/Type_Intial/Type_3/Type_3a/Front_View_3a/"
folder_final_rgb_a = "datasets/224/Type_RGB/type_a/"

folder_initial_b = "datasets/224/Type_Intial/Type_3/Type_3b/"
folder_final_rgb_b = "datasets/224/Type_RGB/type_b/"

folder_initial_c = "datasets/224/Type_Intial/Type_3/Type_3c/"
folder_final_rgb_c = "datasets/224/Type_RGB/type_c/"


create_rgb_hair_resize(folder_initial_a, folder_final_rgb_a)
create_rgb_hair_resize(folder_initial_b, folder_final_rgb_b)
create_rgb_hair_resize(folder_initial_c, folder_final_rgb_c)