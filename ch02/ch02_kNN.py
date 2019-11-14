#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:47:03 2019

@author: zhangsiyuan
"""
#暂存路径： '/Users/zhangsiyuan/Downloads/github/zsy'
from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    #计算距离
    dataSetSize = dataSet.shape[0];
    diffMat = tile(inX , (dataSetSize, 1)) -dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #选择距离最小的k个点
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k) :
        voteIlabel = labels[ sortedDistIndicies[i] ]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),
    key=operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0 ,1.1] , [1.0 ,1.0],[0 ,0],[0 , 0.1]])
    labels= ['A', 'A', 'B', 'B']
    return group, labels

#使用k近邻算法改进约会网站的配对效果
#第一步 准备数据
def file2matrix(filename):
    fr = open(filename)
    #得到文件行数
    arrayOLines = fr.readlines()
    numberOflines = len(arrayOLines)
    #创建返回的Numpy矩阵
    returnMat = zeros((numberOflines , 3))
    classLabelVector =[]
    index=0
    #解析文件数据到列表
    for line in arrayOLines :
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat, classLabelVector
# 数据的载入
# datingDataMat,datingLabels = chapter2_kNN.file2matrix('datingTestSet2.txt')

#第二步分析数据
#散点图的绘制    
    '''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(datingDataMat[: , 1],datingDataMat[:,2],15.0*array(datingLabels),
15.0*array(datingLabels))
#ax.scatter(datingDataMat[: , 1], datingDataMat[:,2])
plt.show()
    '''
 
#第三步 准备数据
def autoNorm(dataSet) :
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m =dataSet.shape[0]
    normDataSet = dataSet -tile(minVals , (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet , ranges,minVals
#normMat,ranges,minVals=chapter2_kNN.autoNorm(datingDataMat)
    
 #第四步测试数据
def datingClassTest():
    hoRatio =0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals= autoNorm(datingDataMat)
    m= normMat.shape[0]
    numTestVecs = int (m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs) :
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],\
                    datingLabels[numTestVecs:m],3)
        print('the classifier came back with:%d,the real answer is:%d'  \
              %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]): errorCount+=1.0
    print (" the total error rate is : %f" %(errorCount/float(numTestVecs)))

    #chapter2_kNN.datingClassTest()
 #第五步，构建完善系统
def classifyPerson():
    resultList = ['not at all', 'in small doses' , 'in large doses']
    percentTats = float(input("per of time spent playing"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat , datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals= autoNorm(datingDataMat)
    intArr = array([ffMiles , percentTats , iceCream])
    classifierResult =classify0((intArr-\
                    minVals)/ranges ,normMat ,datingLabels ,3)
    print("You will probably like this person: ",\
          resultList[classifierResult - 1])
    
    
 #2.3手写识别系统
 #载入数据
def img2vector(filename) :
    returnVect =zeros((1,1024))
    fr=open(filename)
    for i in range(32) :
        lineStr = fr.readline()
        for j in range(32) :
            returnVect[0 , 32*i+j] =int(lineStr[j])
    return returnVect

#对应的简单测试用例 testVector= chapter2_kNN.img2vector(
#\  '/Users/zhangsiyuan/Downloads/github/zsy/testDigits/0_13.txt')       
#测试算法
def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat =zeros((m,1024))
    for i in range(m) :
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s'% (fileNameStr))
    testFileList = listdir('testDigits')
    errorCount=0.0
    mTest = len(testFileList)
    for i in range(mTest) :
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,\
                        trainingMat ,hwLabels ,3)
        print("the classifier came back with: %d, the real answer is :%d"\
              %(classifierResult, classNumStr))
        if(classifierResult != classNumStr) :errorCount +=1.0
        print("\n total number of errrors is :%d"% errorCount)
        print("\n total error rate is : %f"%(errorCount/float(mTest)))
        

    
    
    
