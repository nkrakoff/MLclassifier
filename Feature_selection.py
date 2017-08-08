# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:55:03 2017

@author: baiintern1
"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def selectfeatures(X,y,k,labels):
    """selects the best k features by looking at X and y]
        
        Args:
            X(numpy array): a 2d array with all the columns
            y(list): the classes that each of the samples are in
            k(int): tht total number of features to return
            labels(list): the labels for the columns in X
            
        Returns:
            (list): the k most signficant features form the data"""
    sel = SelectKBest(f_classif,"all")
    sel.fit(X,y)
    z = zip(sel.scores_, labels)
    a = sorted(z,key=lambda x: -x[0])
    nums = []
    count=0
    for i,j in a:
        if count<k:
            nums.append(j)
            count+=1
        print(i,j)
    return nums