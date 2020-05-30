# Face-Mask-detector
Covid-19 face mask detector using Keras and OpenCv.
This code can detect if a person is wearing mask or not.


## Applications

In order take precautionary measures due to covid-19, this can be used by business and authorities to check if a person entering thier premises is wearing mask or not. Also, it is just a one idea but it can be used for different purposes as well.


### Code Requirements
The example code is in Python ([version 3.6](https://www.python.org/download/releases/3.6/) or higher will work). 

### Dependencies

1) import cv2
2) import imutils
3) import keras
4) import scipy
5) import tensorflow
6) import numpy


### Description

A computer vision system that can automatically detect if a person is wearing masks or not in a real-time video stream.

### Dataset

This dataset consists of 1,376 images belonging to two classes:

    *with_mask
    : 690 images
    *without_mask
    : 686 images
 
 ![Dataset](https://github.com/msalmankhaliq/Face-Mask-detector/blob/master/face_mask_detection_dataset.jpg)

 Our goal is to train a custom deep learning model to detect whether a person is or is not wearing a mask.

### Training

Using keras and dnn to train the model with the generated dataset.

After training, we get good accuracies on both test and train set.

<img src="https://github.com/msalmankhaliq/Face-Mask-detector/blob/master/plot.png">

#### Implementing on real-time video streams with OpenCv

Now we implent the model on real-time video streams using OpenCv. The code for that is in the following ipynb file:

```
mask detector on Video .ipynb
```

For more information, [see](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)

### Working Example


[![Watch the video](https://img.youtube.com/vi/t5Ekzprwh-c/hqdefault.jpg)](https://youtu.be/t5Ekzprwh-c)

### License 

 Licensed under the Apache License, Version 2.0. Copyright 2019.

