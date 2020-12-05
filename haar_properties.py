# -*- coding: utf-8 -*-
"""
Define Haar feature properties.

@author: peter
"""

#import cv2
import numpy as np
from haars import *

class Haar(object):
    
    def __init__(self, pattern, upperleft, blockSize):
        self.pattern = pattern
        self.upperleft = upperleft
        self.blockSize = blockSize
        self.pattern_shape = np.shape(pattern)
        self.shape = ( self.pattern_shape[0] * blockSize, 
                      self.pattern_shape[1] * blockSize )
    
    def getBlockProp(self):
        
        props = []        
        for row in range( self.pattern_shape[0] ):
            for col in range( self.pattern_shape[1] ):
                
                this_upperleft = ( self.upperleft[0] + row * self.blockSize, 
                             self.upperleft[1] + col * self.blockSize)
                
                this_lowerright = ( this_upperleft[0] + self.blockSize - 1,
                             this_upperleft[1] + self.blockSize - 1 )
                
                props.append( ( (this_upperleft,this_lowerright), 
                             self.pattern[row][col]) )
        
        return props
    
    def get_feature_val(self, integral_image):

        pos, neg, n_pos_blocks, n_neg_blocks = 0, 0, 0, 0
        
        for (ul, lr), sign in self.getBlockProp():
            _sum11 = integral_image[ lr ]
            _sum00 = integral_image[ ul[0]-1 ][ ul[1]-1 ] if (ul[0] > 0) and (ul[1] > 0) else 0
            _sum10 = integral_image[ lr[0] ][ ul[1]-1 ] if (ul[1] > 0) else 0
            _sum01 = integral_image[ ul[0]-1 ][ lr[1] ] if (ul[0] > 0) else 0
            
            if sign == 1:
                pos += (_sum11 + _sum00 - _sum10 - _sum01)
                n_pos_blocks += 1
            else:
                neg += (_sum11 + _sum00 - _sum10 - _sum01)
                n_neg_blocks += 1
            
        return (pos * n_pos_blocks - neg * n_neg_blocks) / (n_pos_blocks + n_neg_blocks)



def get_haars( img_shape, startpoint = (0,0), windowSize = 1, 
              blockSize0 = 1, blockSize_intvl = 1, blockSize_loop = 256 ):

    haars = []    
    for i in range(haar_cnt):
    
        haar_pattern = np.array( globals()['haar'+str(i)] )
        this_haar = Haar( haar_pattern, startpoint, blockSize0 )
        
        haars += get_augumented_haars( this_haar, img_shape, windowSize, 
                                      blockSize_intvl, blockSize_loop  )
        
    return haars

def get_augumented_haars( haar, img_shape, windowSize, blockSize_intvl, blockSize_loop ):
    
    row_res, col_res = img_shape[0] - haar.upperleft[0], img_shape[1] - haar.upperleft[1]
    haars = []
    
    sizeUps = min( blockSize_loop,
                  (row_res - haar.shape[0]) // (blockSize_intvl * haar.pattern_shape[0]) + 1,
                  (col_res - haar.shape[1]) // (blockSize_intvl * haar.pattern_shape[1]) + 1 )
    
    for i in range(sizeUps):  
        this_haar = Haar( haar.pattern, haar.upperleft, 
                         haar.blockSize + i * blockSize_intvl )
        haars += get_mapped_haars( this_haar, row_res, col_res, windowSize )
        
    return haars

def get_mapped_haars( this_haar, row_res, col_res, windowSize ):
    
    row_iters = (row_res - this_haar.shape[0])//windowSize + 1
    col_iters = (col_res - this_haar.shape[1])//windowSize + 1
    
    haars = []
    for col in range(col_iters):
        for row in range(row_iters):
            this_upperleft = ( this_haar.upperleft[0] + row * windowSize,
                               this_haar.upperleft[1] + col * windowSize )
            haars.append(  Haar(this_haar.pattern, this_upperleft, this_haar.blockSize ) )
    
    return haars