# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:16:23 2019

@author: ASUS
"""

# 分类器
import numpy as np
import operator

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat,classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet / np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals
        
        
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]
    

def datingClassTest():
    filename = "datingTestSet.txt"
    datingDataMat,datingLabels = file2matrix(filename)
    hoRatio = 0.10
    normMat,ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)
        print("分类结果：%d\t真实类别：%d" % (classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount +=1.0
    print("错误率:%f%%" % (errorCount/float(numTestVecs)*100))

def classifyPerson():
    resultList = ['讨厌','有些喜欢','非常喜欢']
    percentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    filename = "datingTestSet.txt"
    datingDataMat,datingLabels = file2matrix(filename)
    normMat,ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([percentTats,ffMiles,iceCream])
    norminArr = (inArr - minVals) / ranges
    classifierResult = classify0(norminArr,normMat,datingLabels,3)
    print("你可能%s这个人" %(resultList[classifierResult-1]))

    
    
if __name__ == '__main__':
    classifyPerson()