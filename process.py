#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 20:13:00 2020

@author: ince

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import json
import cv2
import imutils
import xxhash
### toggle annoying warnings
import logging
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)

### RADIOMICS FEATURE EXTRACTOR SETUP
from radiomics import featureextractor
import SimpleITK as sitk
extractor = featureextractor.RadiomicsFeatureExtractor(additionalInfo=True)
extractor.enableAllFeatures()
extractor.enableAllImageTypes()

def image_preprocessing(image: 'np.ndarray') -> 'np.ndarray':
    """
    Performs some transformations on the image in sequence:
        1) Scale the values in [0, 255] (It comes in a uint8 dicom format)
        2) Crop normalization: Crops only the brain section
        3) Resize to the desired target shape
        4) Non-Local Means denoising

    Parameters
    ----------
    image : 'np.ndarray'
        The image you want to perform the processing.

    Returns
    -------
    denoised_image : np.ndarray
        image with all the preprocessing step performed.

    """

    scaled_image   = scale_values(image)
    cropped_image  = crop_normalization(scaled_image)
    resized_image  = resize(cropped_image)
    denoised_image = denoise(resized_image)
    
    return denoised_image
    
    



def scale_values(image, upper_bound=255.0):
    """
    
    Scale the values in [0, upper_bound] (It comes in a uint16 dicom format)
    Parameters
    ----------
    image : np.ndarray
        image in DICOM format, the range is [0, 65536].
    upper_bound : float, optional
        the upper bound of the scaled range. The default is 255.0.

    Returns
    -------
    image : np.ndarray
        the image scaled in [0, upper_bound].

    """
    image = image/image.max()
    image = np.uint8(image * upper_bound)
    return image


def crop_normalization(image, border_length=15):
    """
    
    Crops the brain section removing all the black part around it.
    
    Parameters
    ----------
    image : np.ndarray
        image in uint8 format, the range is [0, 255].
        
    border_length : int
        Specifies the length of the preserved black part,
        it's like a guard to prevent losses due to the crop normalization

    Returns
    -------
    cropped_image : np.ndarray
        the mri cropped aound the brain.

    """
    
    # 2) 
    gray = cv2.GaussianBlur(image, (5, 5), 0)
    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)


    # determine the most extreme points along the contour
    extremes = {}
    extremes['extLeft'] = tuple(c[c[:, :, 0].argmin()][0])
    extremes['extRight'] = tuple(c[c[:, :, 0].argmax()][0])
    extremes['extTop'] = tuple(c[c[:, :, 1].argmin()][0])
    extremes['extBot'] = tuple(c[c[:, :, 1].argmax()][0])

    cropped_image = image[extremes['extTop'][1] - border_length : extremes['extBot'][1] + border_length, 
                          extremes['extLeft'][0] - border_length : extremes['extRight'][0] + border_length]
    
    if (cropped_image.shape[0] < border_length) or (cropped_image.shape[1] < border_length):
        cropped_image = image[extremes['extTop'][1] : extremes['extBot'][1] , 
                              extremes['extLeft'][0] : extremes['extRight'][0]]
    

    return cropped_image



def resize(image, target_shape=(224, 224)):
    """
        
    Resize the image into the desired shape specified as parameter.
    The interpolation method is optimized to the imput shape

    Parameters
    ----------
    image : 'np.ndarray'
        The image you want to perform the resize.
    target_shape : tuple, optional
        The output shape of the resized image. The default is (224, 224).

    Returns
    -------
    resized_image : np.ndarray.
        The image resized in target_shape.

    """
     # 3) Resize to the desired target shape
    interpolation = cv2.INTER_CUBIC if image.shape[0] < target_shape[0] else cv2.INTER_AREA
    resized_image = cv2.resize(image, target_shape, interpolation=interpolation)
    
    return resized_image
    
def denoise(image):
    """
    
    Perform NoN-Local Means denoising on the image passed as parameter.

    Parameters
    ----------
    image : 'np.ndarray'
        The image you want to perform the denoise algorithm.

    Returns
    -------
    denoised_image : np.ndarray
        Denoised image.

    """
    denoised_image = cv2.fastNlMeansDenoising(image, 10, 10, 7, 21)
    return denoised_image



def normalize(X_train, X_test):
    """
    
    Normalize the train and test set by performing in sequence:
        1- Subtract the mean
        2- Divide by the standard deviation
        3- Scale in [0, 1]

    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.

    Returns
    -------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.

    """
    for i in range(len(X_train)):
        if i < len(X_test):
            X_test[i] -= X_test[i].mean()
            X_test[i] /= X_test[i].std()
            X_test[i] /= X_test[i].max()
            
        X_train[i] -= X_train[i].mean()
        X_train[i] /= X_train[i].std()
        X_train[i] /= X_train[i].max()
        
    return X_train, X_test

def compute_radioimcs_and_glue(X, Y):
    """
    
    Given an image, compute the associated radiomics vector and compress everything in a dictionary indexed by
    an hash function in int64 of the image

    Parameters
    ----------
    X : np.ndarray
        Processed image.
    Y : np.ndarray
        Target variable one-hot encoded.

    Returns
    -------
    compressed: dict
        Dataset stored in a dict format {K, (X,R,Y)}:
                1- K is an int64 hash of the image
                2- X is the image after processing 
                3- R is the associated radiomics vector
                4- Y is the target variable

    """
    return {xxhash.xxh64_intdigest(x) : (x, extractor.computeFeatures(sitk.GetImageFromArray(x), sitk.GetImageFromArray(np.ones(x.shape)), "original"), y) for x,y in zip(X, Y)}





def smooth_plot(x, y=None, label='', halflife=10):

    """ 
    Function to plot smoothed x VS y graphs.

    Parameters:	
    ----------

    x - x-axis data.
    y - y-axis data.
    label - Legend for the current graph.
    halflife - Smoothing level.

    ----------

    Yields:	x VS y graphs.

    ----------
    """

    if y is None:
      y_int = x
    else:
      y_int = y
    x_ewm = pd.Series(y_int).ewm(halflife=halflife)
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    if y is None:
      plt.plot(x_ewm.mean(), label=label, color=color)
      plt.fill_between(np.arange(x_ewm.mean().shape[0]), x_ewm.mean() + x_ewm.std(), x_ewm.mean() - x_ewm.std(), color=color, alpha=0.3)
    else:
      plt.plot(x, x_ewm.mean(), label=label, color=color)
      plt.fill_between(x, y_int + x_ewm.std(), y_int - x_ewm.std(), color=color, alpha=0.3)