#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:30:30 2019

@author: zhangsiyuan
"""
import operator
#第三章 决策树
#3.1计算给定数据集的香农熵
from math import log
def calcShannonEnt(dataSet) :
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet :
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob ,2)
    return shannonEnt

#3.1(准备测试的数据集)
def createDataSet() :
     dataSet = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no' ]]
     labels = ['no surfacing', 'flippers']
     return dataSet, labels
 
'''测试用命令1
myDat ,labels = createDataSet()
myDat
calcShannonEnt(myDat)
myDat[0][-1] = 'maybe'
myDat
calcShannonEnt(myDat)
'''
#3.2划分数据集：根据给定某一项特征的某个值，提取出所有的对应列表
def splitDataSet(dataSet, axis, value) :
    retDataSet = []
    for featVec in dataSet :
        if featVec[axis]  == value :
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet   
'''测试用命令2
myDat ,labels = createDataSet()  
myDat
splitDataSet(myDat,0,1)
splitDataSet(myDat,0,0)
''' 

#3.3通过计算信息增益，选取最优的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) -1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0 ; bestFeature = -1
    for i in range(numFeatures):
        featList = [ example[i] for example in dataSet ]
        uniqueVals = set(featList)  #合并数据
        newEntropy = 0.0
        for value in uniqueVals :
            subDataSet = splitDataSet(dataSet , i ,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob* calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain) :
            bestInfoGain = infoGain
            bestFeature = i 
    return bestFeature
    
'''测试用例三
  myDat ,labels = createDataSet()
  chooseBestFeatureToSplit(myDat)
  myDat
'''
# 返回次数最多的分类名称
def majorityCnt(classList):   
    classCount = {}
    for vote in classList :
        if vote not in classCount.keys() : classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),\
        key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

#3.4创建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count( classList[0])==len( classList ) :
        return classList[0]
    if len(dataSet[0]) == 1 :
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel : {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals :
        subLabels = labels[:]
        myTree[bestFeatLabel][value] =createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree 
   
 

'''测试用例
 myDat ,labels = createDataSet()
 myTree = createTree(myDat , labels)
 myTree
'''   
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel  
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)