# -*- coding: utf-8 -*-
"""
Define classifier properties and operations

@author: peter
"""

#import time

class Weak_Classifier(object):

    def __init__(self, haar, imaegs, weights):
        self.feature = haar
        self.images = imaegs
        self.weights = weights
        self.polarity, self.threshold, self.feature_vals = self.get_best_threshold()
        self.predictions, self.error = self.predict()
        self.beta = self.error / ( 1 - self.error )
    
    # Train weak classifiers with generated features and define best threshold
    # for each classifiers
    def get_best_threshold(self):
    
        # Record feature values from input images
        haar_vals, feature_vals = [], []
    
        # Get sum of weigths of positive / negative samples
        total_pos, total_neg = 0.0, 0.0
    
        for img, weight in zip( self.images, self.weights ):
            haar_vals.append ( (img.isPositive , weight, self.feature.get_feature_val( img.integral ) ) )
            feature_vals.append( self.feature.get_feature_val( img.integral ) )
            
            if img.isPositive:  total_pos += weight
            else:   total_neg += weight
            
        # sort haar features by value
        haar_vals = sorted( haar_vals, key = lambda x: x[2] )   
        
        # Loop over threshold values and get the one with smallest error
        smallest_error, best_threshold = float('inf') , 0.0
        # Get sum of positive / negative sample weights
        sum_pos, sum_neg, polarity = 0.0, 0.0, 1
    
        for isPositive, weight, threshold in haar_vals:
        
            # minimum_error = min((S+) + (T-) - (S-), (S-) + (T+) - (S+))
            if isPositive:
                 sum_pos += weight
                 this_error = sum_pos + total_neg - sum_neg
                 if this_error < smallest_error:
                     smallest_error = this_error
                     best_threshold = threshold
                     polarity = 1
            else:
                sum_neg += weight
                this_error = sum_neg + total_pos - sum_pos
                if this_error < smallest_error:
                    smallest_error = this_error
                    best_threshold = threshold
                    polarity = -1
    
        return polarity, best_threshold, feature_vals
    
    # Return a boolean whether the sample being classified correctly by this classifier
    def isHit(self, feature_val ):
        
        return self.polarity * feature_val < self.polarity * self.threshold
        
    
    # Predict over samples and get accumulation of error ( 0.0 <= err <= 1.0 ) of this classifier
    def predict(self):
        
        predictions, sum_error = [], 0.0
        for feature_val, weight in zip(self.feature_vals, self.weights):
            
            # While current sample image predicted successfully
            if  self.isHit(feature_val):
                predictions.append(True)
            # While not, increment current sample weight
            else:
                predictions.append(False)
                sum_error += weight   
        # Test
        #print(sum_error)
        #time.sleep(0.01)
        # Test
        return predictions, sum_error
    
    # Return updated weighted for the next training iteration
    def get_updated_weights(self):
        
        new_weights = []       
        for predict, weight in zip( self.predictions, self.weights ):
            
            # weight * beta if classified correctly
            if predict:
                new_weights.append( weight * self.beta )
            # keep weight value if not
            else:
                new_weights.append( weight )
        
        return new_weights

class Strong_Classifier(object):

    def __init__(self, cascade : list(Weak_Classifier) ):
        self.weak_classifiers = cascade
