# -*- coding: utf-8 -*-
"""
Created on Thu May 16 01:37:32 2019

@author: Gael
"""
import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import time
import sys
sys.path.insert(0, 'lib/')
from GoogleNetwork import GoogLeNet as DNN

from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, precision_score

# This file is used for testing classification algorithm


###############                Parameters
number_channels = 3

###############              Folder locations
# Specify the location of the hair segment obtained from data augmentation
folder_data_a = 'datasets/224/validation/type_a/'   
folder_data_b = 'datasets/224/validation/type_b/'
folder_data_c = 'datasets/224/validation/type_c/'

#############            Function to read the training data
# The file dataset_train.txt is read and parsed
def reading_testing_data(folder_data, hair_type):
    # y_train = hair type: [p_a, p_b, p_c] softmax
    # X_train: training images nx224x224x3
    # hair_type is either 'a' for type_a, 'b' for type_b and 'c' for type_c
    allimages = os.listdir(folder_data)
    number_images = len(allimages)
    print('Images are ', number_images)
    y_train = np.zeros((number_images, 3))
    X_train = np.zeros((number_images, 224, 224, number_channels))
    name_images = []
    
    if hair_type == 'a':
        label = np.array([1, 0, 0])
    elif hair_type == 'b':
        label = np.array([0, 1, 0])
    else:
        label = np.array([0, 0, 1])
    
    print('Total images for training for this type', len(allimages))

    for i in range(len(allimages)):
        y_train[i] = label
        img = load_img(folder_data+allimages[i]) # This is a PIL image
        X_train[i] = img_to_array(img)  # this is a Numpy array with shape (3, 224, 224)
        name_images.append(allimages[i])
    return X_train, y_train, name_images

#############            Normalization
    #We normalize the data by substracting the mean and scaling 
def normalization_test(X_test, MEAN):
    # # Forcing the pixels and the poses to be coded as floats
    for i in range(len(X_test)):
        X_test[i] = X_test[i] - MEAN[0]
    return X_test

   
##############################################################################
##############################################################################
##############################################################################

# Reset the graph
tf.reset_default_graph()

#Placeholder input data
image = tf.placeholder(tf.float32, [1, 224, 224, number_channels], name="image_data")

#Architecture
net = DNN({'data': image})

#Output of DNN
# Last softmax transformed to FC
type_pred_3 = net.layers['type_3']


# Initializer variable
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#Getting data
print('Readingand aggregating the testing data')
X_test_a, y_test_a, name_images_a= reading_testing_data(folder_data_a, 'a')
X_test_b, y_test_b, name_images_b= reading_testing_data(folder_data_b, 'b')
X_test_c, y_test_c, name_images_c= reading_testing_data(folder_data_c, 'c')

# We aggregate data from all hair types
X_test = np.append(X_test_a, X_test_b, axis=0)
y_test = np.append(y_test_a, y_test_b, axis=0)
name_images = np.append(name_images_a, name_images_b)

X_test = np.append(X_test, X_test_c, axis=0)
y_test = np.append(y_test, y_test_c, axis=0)
name_images = np.append(name_images, name_images_c)

#Getting mean of training images
path_save = "Save"
filename = 'MEAN.mat'
x = sio.loadmat(os.path.join(path_save, filename))
MEAN = x['MEAN']

#Normalizing
X_test = normalization_test(X_test, MEAN)
print("Shape of y_test ", np.shape(y_test))
print("Shape of Dataset X_test ", np.shape(X_test))

# Data to save 
duration = np.zeros((len(name_images))) 
type_pred_result = np.zeros((len(name_images)))


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8833)
print('Starting training')
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    
    # Loading the model 
    #model_last
    saver = tf.train.import_meta_graph('Save2/model_3.ckpt.meta')
    saver.restore(sess, "Save2/model_3.ckpt")
    
    for i in range(len(X_test)):
        # Timer starts
        start = time.time()
        # Input image
        image_test = X_test[i, :, :, :]
        image_test_tensor = np.reshape(image_test, [1, 224, 224, number_channels])
        feed = {image: image_test_tensor}
        
        #Estimated parameters
        output = sess.run([type_pred_3], feed_dict=feed)
        type_pred_result[i] = np.argmax(output)
        print('Predicted Hair Type',  type_pred_result[i], 'Actual Hair Type ', np.argmax(y_test[i]), 'Name ', name_images[i])
        
        # Timer ends
        end = time. time()
        duration[i] = end-start 

print('Finish')
confusion_matrix(type_pred_result, np.argmax(y_test, axis=1))
accuracy = accuracy_score(np.argmax(y_test, axis=1), type_pred_result)
precision_recall_fscore_support(np.argmax(y_test, axis=1), type_pred_result)
precision = precision_score(np.argmax(y_test, axis=1), type_pred_result, average='macro')
print('Accuracy is ', accuracy)
print('Precision is ', precision)
        
