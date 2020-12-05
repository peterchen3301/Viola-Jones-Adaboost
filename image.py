# -*- coding: utf-8 -*-
"""
Define image properties

@author: peter
"""

import cv2
import numpy as np

class png_gray(object):
    def __init__(self, image_path, isPositive ):
        self.image = cv2.imread( image_path )[:,:,0]
        self.integral = self.integral_image(self.image)
        self.isPositive = isPositive
        
    # Get the integrated image to speed up region summations in later steps
    def integral_image(self, image : np.ndarray ):
    
        if type(image) != np.ndarray:
            raise TypeError("Input must be numpy.ndarray") 
                   
        return image.cumsum(axis=0).cumsum(axis=1)