# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:48:51 2019

@author: ASUS
"""

#经验熵
from math import log
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],         
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']		
    return dataSet, labels                


def calcShannonEnt(dataSet):
    numEntires = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] =0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    

if __name__ == '__main__':
    dataSet,features = createDataSet()
    print(dataSet)
    print(calcShannonEnt(dataSet))
    
    
    
    
    
        
        
        
        
        
        
        