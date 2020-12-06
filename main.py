# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 03:03:35 2020

@author: peter
"""

from train_test import train, test, save_cascade

def main():
    
    # Train classifiers using training samples
    train_pos_dir = "VJ_dataset\\trainset\\faces"
    train_neg_dir = "VJ_dataset\\trainset\\non-faces"
    rounds = 10
    
    cascade = train( train_pos_dir, train_neg_dir, rounds )
    
    # Save training results
    save_cascade(cascade)
    
    # Test with test samples
    test_pos_dir = "VJ_dataset\\testset\\faces"
    test_neg_dir = "VJ_dataset\\testset\\faces"
    
    test( test_pos_dir, test_neg_dir, cascade, 10 )
    
    
if __name__ == '__main__':
    main()