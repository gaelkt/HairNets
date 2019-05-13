# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:52:15 2019

@author: g84120421
"""

import os
import numpy as np
import cv2


# WE SELECT THE FACES THAT WE ARE GOING TO USE
folder_gt_images = "datasets/parts_lfw_funneled_gt/parts_lfw_funneled_gt/"
folder_images = "datasets/lfw_funneled/lfw_funneled"
destination = "datasets/faces/"


hair_segment = "datasets/hair_segment/"
hair_training = "datasets/hair_training/"
folder_mask = "datasets/parts_lfw_funneled_gt_images/parts_lfw_funneled_gt_images/"


def select_faces(folder_gt_images, folder_images, destination):
#This function selects only images which ground truth masks labels are provided in the LBW dataset
    i = 0
    #Path for person names
    person_name = [f.name for f in os.scandir(folder_gt_images) if f.is_dir() ] #name of person

    for person in person_name:
        #path for images on .dat for that person
        path_dat =  folder_gt_images + person + "/"
        #list of all dat images for that person
        images_dat = os.listdir(path_dat)
    
        for images in images_dat:
            image_jpg = images.replace(".dat", ".jpg")
            img = cv2.imread(folder_images + person + "/" + image_jpg)
            cv2.imwrite(destination+image_jpg,img)
            i = i+1

    print("Total number of label faces in the wild with ground truth ", i)
    
def convert_mask_gray(folder_mask, hair_segment, hair_training):
    
    # MASK ARE CONVERTED TO binary mask
    #We only take the red channel that correspons to hair segment that is number 2
    #Images and Masks are resized to 128x128

    allmasks = os.listdir(folder_mask)

    width,height = 224, 224
    dim = (width, height)

    for mask in allmasks:
        img = cv2.imread(folder_mask+mask)
        image_name = mask.replace(".ppm", "")
        image_name = mask.replace("._", "")
        image_jpg = image_name + ".jpg"
    
    # Face image in grayscale
        img_gray_face = cv2.imread(destination+image_jpg, 2)
        if (img is not None) and (img_gray_face is not None):
        #Hair segment
            segment = img[:, :, 2]   # the hair segment is the red channel
            resized = cv2.resize(segment, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(hair_segment+image_jpg,resized)
        
        #Face in grayscale
            resized_gray = cv2.resize(img_gray_face, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(hair_training+image_jpg,resized_gray)

