# Deep Learning With Pytorch

## Description 💡

This README contains the following course:

- [Deep Learning With Pytorch](https://www.udemy.com/course/formacao-deep-learning-pytorch-python/)

## Table of Contents 📚

I divided the course into projects, and each project has it own porpuse. The topics are:

### Binary Classification 🎯

- [Project 1 - Breast Cancer classification](/Courses/Deep%20Learning%20With%20Pytorch/Project_1.ipynb)

- [Project 2 - Binary classification brest cancer with cross validation and dropout](/Courses/Deep%20Learning%20With%20Pytorch/Project_2.ipynb)

- [Project 3 - Binary classification for brest cancer with tuning](/Courses/Deep%20Learning%20With%20Pytorch/Project_3.ipynb)

- [Project 4 - Based on Grid Search best params train with all data](/Courses/Deep%20Learning%20With%20Pytorch/Project_4.ipynb)

- [Project 5 - Loading classifier](/Courses/Deep%20Learning%20With%20Pytorch/Project_5.ipynb)

### Multiclass Classification 🎯

- [Project 6 - Multiclass classifier iris dataset](/Courses/Deep%20Learning%20With%20Pytorch/Project_6.ipynb)

- [Project 7 - Grid Search + Parameter Tuning + DropOut in Iris dataset](/Courses/Deep%20Learning%20With%20Pytorch/Project_7.ipynb)

### Regression 🎯

- [Project 8 - Regression with car price dataset](/Courses/Deep%20Learning%20With%20Pytorch/Project_8.ipynb)

- [Project 9 - Cross Validation to car price dataset](/Courses/Deep%20Learning%20With%20Pytorch/Project_9.ipynb)

- [Project 10 - Grid Search + Parameter Tuning in car price dataset](/Courses/Deep%20Learning%20With%20Pytorch/Project_10.ipynb)

---

### Multiclass Regression 🎯

- [Project 11 - Multiclass regression for video game sales](/Courses/Deep%20Learning%20With%20Pytorch/Project_11.ipynb)

### Convolutional Neural Networks 🎯

Used for computer vision. Also is the reason why we can build self-driving cars. They are better than SVM.

To better train this type o neural network we should extract the features from the images. Example: We want to predict if is or not homer simpson. We should extract the format of the face, the eyes, the mouth, etc. However in this step we should take care of extract the features that are general and not specific to the image.

Dense neural netwoorks are not good for images because the time to train is too high. Imagine a 32x32 image. If we build a dense neural network we would have 3072 input neurons. This would be a lot of weights to train.

#### Convolutional operator

Process of add each image element to the respective neighboor. Weights are based on kernel. 

The image is a matrix and the kernel is the other matrix.

$$ (f*g)[n] = \sum_{m=-\infty}^{\infty} f[m]g[n-m] $$

The kernel is the result of pre processing the image. The kernel is the filter.

#### Pooling

Pooling is the process of highlight features in the image. This process also reduces the image size.

So basic the steps of a convolutional neural network are:

Image -> feature detector -> feature map -> activation function -> Max pooling -> Flattening -> Full connection

#### Flattening

Flattening is the process of convert the matrix into a vector.


### Convolutional Neural Networks 🎯

- [Project 12 - Convolutional Neural Network MNIST](/Courses/Deep%20Learning%20With%20Pytorch/Project_12.ipynb)

- [Project 13 - Cross validation for MNIST dataset](/Courses/Deep%20Learning%20With%20Pytorch/Project_13.ipynb) 

- [Project 14 - Image Augmentation](/Courses/Deep%20Learning%20With%20Pytorch/Project_14.ipynb)

- [Project 15 - CIFAR-10 Image classification](/Courses/Deep%20Learning%20With%20Pytorch/Project_15.ipynb)

- [Project 16 - Convolutional Neural Network for dog and cat classification](/Courses/Deep%20Learning%20With%20Pytorch/Project_16.ipynb)

- [Project 17 - Classify a homer or bart simpsons with a dense neural network](/Courses/Deep%20Learning%20With%20Pytorch/Project_17.ipynb)

- [Project 18 - Convolutional Neural Network for homer and bart simpsons classification](/Courses/Deep%20Learning%20With%20Pytorch/Project_18.ipynb)

### Autoencoders 🎯

- [Project 19 - Image compaction with autoencoders](/Courses/Deep%20Learning%20With%20Pytorch/Project_19.ipynb)

- [Project 20: Classification with autoencoders](/Courses/Deep%20Learning%20With%20Pytorch/Project_20.ipynb)

- [Project 21: Image compression with deep autoencoders](/Courses/Deep%20Learning%20With%20Pytorch/Project_21.ipynb)

- [Project 22: Image compression with convolutional autoencoders](/Courses/Deep%20Learning%20With%20Pytorch/Project_22.ipynb)

- [Project 23: Deep autoencoders with cifar-10 dataset](/Courses/Deep%20Learning%20With%20Pytorch/Project_23.ipynb)

### Generative Adversarial Networks 

- [Project 24: Generate new images using MNIST dataset (Fully connected)](/Courses/Deep%20Learning%20With%20Pytorch/Project_24.ipynb)

## Additional Information ℹ️

You can find the datasets and weights in the directory `Datasets` and `Weights` respectively.