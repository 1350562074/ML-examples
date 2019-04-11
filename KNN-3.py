# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:02:31 2019

@author: ASUS
"""



"""
数字图片是32*32的二进制图像
为了处理方便，将其转换成一行1024列矩阵

sklearn.neighbors模块实现了k-近邻算法
sklearn的KNeighborsClassifier输入可以是矩阵，不用一定转换为向量
"""
from os import listdir
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN


def img2vector(filename):
    returnVect = np.zeros((1,1024))
    f = open(filename)
    for i in range(32):
        lineStr = f.readline()
        
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
            
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector('trainingDigits/%s'%(fileNameStr))
        
    
#    n_neighbors：默认为5，就是k-NN的k的值，选取最近的k个点;
#    algorithm：快速k近邻搜索算法，默认参数为auto，可以理解为算法自己决定合适的搜索算法。
#               除此之外，用户也可以自己指定搜索算法ball_tree、kd_tree、brute方法进行搜索;

    neigh = kNN(n_neighbors = 5, algorithm = 'auto')
    neigh.fit(trainingMat,hwLabels)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
            
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))

    
if __name__ == '__main__':
    handwritingClassTest()
    
    
    
    
 
"""
优点：
·简单好用，精度高，理论成熟，既可以用来做分类也可以用来做回归；
·可用于数值型、离散型数据；
·训练时间复杂度为O(n)；无数据输入假定；
·对异常值不敏感。
"""

"""
缺点：
·计算、空间复杂性高；
·样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；
·一般数值很大的时候不用这个，计算量太大。但是单个样本又不能太少，否则容易发生误分。
·最无法给出数据的内在含义。
"""