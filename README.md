# Project Description

This is the final project of the Applied AI course, with Guo Zhi Wang as my project collaborator. The objective of this project was to use the "Cambridge Labeled Objects in Video" dataset for image semantic segmentation. The dataset comprises 90 training images and 11 testing images, each accompanied by a corresponding semantic segmentation map that represents 32 different classes of objects in driving environments. We implemented three different models for image semantic segmentation: Fully Convolutional Networks (FCN), DeepLabV3, and LRASPP_MobileNet_V3_Large. During the training process, we employed data augmentation and split the training data into a training set (90%) and a validation set (10%).

# Project Aims

1. Understand the application of Computer Vision (CV).
2. Demonstrate proficiency in using PyTorch in deep learning.

# Result

## 1. Fully Convolutional Networks (FCN)

After cross-validation, we selected the following hyperparameters: not pretrained weights, a ResNet 50 backbone, SGD with momentum optimizer, and pixel-wise cross-entropy loss function. The final model was trained for 80 epochs. The pixel-wise accuracy for training is 0.938, and for testing, it is 0.939.

## 2. DeepLabV3

Following cross-validation, we opted for the hyperparameters: not pretrained weights, a ResNet 50 backbone, SGD with momentum optimizer, and pixel-wise cross-entropy loss function. The final model was trained for 80 epochs. The pixel-wise accuracy for training is 0.936, and for testing, it is 0.935.

## 3. LRASPP_MobileNet_V3_Large

After cross-validation, we chose the hyperparameters: not pretrained weights, SGD with momentum optimizer, and pixel-wise cross-entropy loss function. The final model was trained for 80 epochs. The pixel-wise accuracy for training is 0.912, and for testing, it is 0.915.
