# HairNets
Hair Segmentation and Classification 
Hair Segmentation uses a U-Net architecture to segment hair pixels from images and Hair Classification uses GoogleNet architecture to classifiy hair segment.

Requirements:
- Tensorflow 1.12
- Keras
- Skimage
- Opencv

This work uses the Part Label Database as dataset for Hair Segmentation. The dataset can be downladed here:

-You need to create a folder datasets and insert three folders for the 'funneled images', 'Ground Truth Images' and 'Ground  Truth Labels' that you will download from the above link. 

Then run the file create_dataset.py to create process the training data 

- Run train.segmentation to train the network for segmentation
- Run test_segmentation to test the segmentation on test images
- Run train_classification to classify hair type
