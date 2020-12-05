# -*- coding: utf-8 -*-
"""
To train cascade classifier by given positive / negative sample path and rounds.

@author: peter
"""

import sys
import numpy as np
import os
from haar_properties import get_haars
from classifier_properties import Weak_Classifier, Strong_Classifier
from image import png_gray

    
def train( _pos_dir : str, _neg_dir : str, rounds : int ):
    
    print( "##### Training Start #####" )
    
    # Get sample images
    print( ">>>>> Loading training samples <<<<<" )
    imgs, img_shape, n_pos, n_neg = load_samples( _pos_dir, _neg_dir )
    print( "     total images : ", n_pos + n_neg, 
          ", positive images : ", n_pos, ", negative images : ", n_neg )
    
    # Get Haar features
    print( ">>>>> Generating features <<<<<" )
    haar_features = get_haars( img_shape, (0,0), 1, 1, 1, 256 )
    print( "     total features : ", len(haar_features) )
    
    # Initializing weights
    print( ">>>>> Initializing weights <<<<<" )
    weights_pos = [1.0 / (2 * n_pos)] * n_pos
    weights_neg = [1.0 / (2 * n_neg)] * n_neg
    weights = weights_pos + weights_neg
    
    # Record classifiers chosen in each round
    cascade = []

    # Start Iterations
    print( ">>>>> Start training iteration <<<<<" )
    for t in range(rounds):
        
        print(" ")
        print( " # Round ", t + 1, " : " )
        # Normalizing sample weights
        weights = [ weight / sum(weights)  for weight in weights ]

        # find the optimal classifier that has the minimum error
        print( "Training weak classifiers ......" )
        opt_classifier, classifier_min_err = None, 1.0
        for i, haar in enumerate(haar_features):
            
            log = ("     training ( " + str(i+1) + " /  " +
                   str(len(haar_features)) + " ) of weak classifiers." )
            sys.stdout.write( '\r' + log )
         
            this_classifier = Weak_Classifier( haar, imgs, weights )
            
            if this_classifier.error < classifier_min_err:
                classifier_min_err = this_classifier.error
                opt_classifier = this_classifier
        
        # If this is not final round, then update new weights for next training
        if t != rounds-1:
            weights = opt_classifier.get_updated_weights()
        
        cascade.append(opt_classifier)
    
    sClassifier = Strong_Classifier( cascade )
    
    return sClassifier

def test( _pos_dir : str, _neg_dir : str, sClassifier : Strong_Classifier ):
    
    print( "##### Test Start #####" )
    
    # Get sample images
    print( ">>>>> Loading test samples <<<<<" )
    imgs, img_shape, n_pos, n_neg = load_samples( _pos_dir, _neg_dir )
    print( "     total images : ", n_pos + n_neg, 
          ", positive images : ", n_pos, ", negative images : ", n_neg )
    
    
    return

def test_one_image( image : png_gray, sClassifier : Strong_Classifier ):
    
        
    
    return 

def load_samples( _pos_dir : str, _neg_dir : str ):
    
    # Load positive and negative training image filepath
    pos_imgs, neg_imgs = load_dataset(_pos_dir,True), load_dataset(_neg_dir,False)
    n_pos, n_neg = len(pos_imgs), len(neg_imgs)
    imgs = pos_imgs + neg_imgs
    
    # Get image shape
    img_shape = np.shape(imgs[0].image)
    
    return imgs, img_shape, n_pos, n_neg
    
def load_dataset( data_dir : str, isPositive : bool ):
    
    images = []
    for file in os.listdir(data_dir):
        if file.endswith(".png"):
            images.append( png_gray( data_dir + "\\" + file, isPositive) )
    return images


