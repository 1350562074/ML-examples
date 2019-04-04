# -*- coding: utf-8 -*-

'''
优点：精度高、对异常值不敏感、无数据输入假定
缺点：计算复杂度高、空间复杂度高
适用数据范围：数值型和标称型
'''


import numpy as np
import operator

def createDataSet():
    #四组二维特征
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    labels = ['爱情片','爱情片','动作片','动作片']
    return group,labels


#if __name__=='__main__':
#    #创建数据集
#    group,labels = createDataSet()
#    #打印数据集
#    print(group)
#    print(labels)




def classify0(inX, dataSet, labels, k):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    
    # 重复dataSetSize次(纵向),1次(横向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    
    #sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    classCount = {}
    
    for i in range(k):
        
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        
        #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        
    #python3中用items()替换python2中的iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序；0则根据 键 排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


'''
第二种写法：直接利用欧式距离
def classify0(inx, dataset, labels, k):
	dist = np.sum((inx - dataset)**2, axis=1)**0.5
	k_labels = [labels[index] for index in dist.argsort()[0 : k]]
	# 出现次数最多的标签即为最终类别
	label = collections.Counter(k_labels).most_common(1)[0][0]
	return label
    
 '''


if __name__ == '__main__':
    group, labels = createDataSet()
    test = [13,10]
    test_class = classify0(test, group, labels, 3)
    print(test_class)