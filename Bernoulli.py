#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import re
import random

def createVL(dSet):
	vSet = set([])     #创建一个空的不重复列表
	for document in dSet:
		vSet = vSet | set(document)     #取并集
	return list(vSet)

def setOfWords2Vec(vList, inSet):
	returnVec = [0] * len(vList)     #创建一个其中元素都为0的向量
	for word in inSet:     #遍历每个词条
		if word in vList:     #如果词条存在于词汇表中，计数+1
		    returnVec[vList.index(word)] = 1
	return returnVec

def trainNB0(trainM, trainC):
	numTrainD = len(trainM)     #计算训练的文档数目
	numWords = len(trainM[0])     #计算每篇文档的词条数
	pTrash = sum(trainC) / float(numTrainD)     #文档属于垃圾邮件的概率
	p0num = np.ones(numWords)
	p1num = np.ones(numWords)     #创建数组ones，词条出现数初始化为1，拉普拉斯平滑
	p0denom = 2.0
	p1denom = 2.0     #分母初始化为2，拉普拉斯平滑
	for i in range(numTrainD):
		if trainC[i] == 1:     #统计属于侮辱类的条件概率所需的数据
		    p1num += trainM[i]
		    p0denom += sum(trainM[i])
		else:     #统计属于非侮辱类的条件概率所需的数据
		    p0num += trainM[i]
		    p0denom += sum(trainM[i])
	p1Vect = np.log(p1num / p1denom)
	p0Vect = np.log(p0num / p0denom)     #取对数防止向下溢出
	return p1Vect, p0Vect, pTrash

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass):
    #p1 = reduce(lambda x, y: x * y, vec2Classify * p1V) * pClass
    #p0 = reduce(lambda x, y: x * y, vec2Classify * p0V) * (1.0 - pClass)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass)
    if p1 > p0:
    	return 1
    else:
    	return 0

def textParse(bigString):     #将字符串转换为字符列表
    listOfTokens = re.split(r'\W*', bigString)     #将特殊符号作为切分标志进行字符串切分
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]    #除了单个字母，其他单词变成小写

def spamtest():
	doclist = []
	classlist = []
	fulltext = []
	for i in range(1, 26):     #遍历25个TXT文件
	    wordlist = textParse(open('第四课训练样本/email/spam/%d.txt' % i).read())     #读取每个垃圾邮件，并将字符串转换成字符串列表
	    doclist.append(wordlist)
	    fulltext.append(wordlist)
	    classlist.append(1)     #标记垃圾邮件
	    wordlist = textParse(open('第四课训练样本/email/ham/%d.txt' % i).read())     #读取每个正常邮件，并将字符串转换成字符串列表
	    doclist.append(wordlist)
	    fulltext.extend(wordlist)
	    classlist.append(0)     #标记正常邮件
	vList = createVL(doclist)     #创建词汇表
	trainset = list(range(50))
	testset = []     #创建存储训练集的索引值的列表和测试集的索引值的列表
	for i in range(25):     #从50个邮件中随机选40个作为训练集，10个做测试集
	    randindex = int(random.uniform(0, len(trainset)))     #随机选取索引值
	    testset.append(trainset[randindex])     #添加测试集的索引值
	    del (trainset[randindex])     #在训练集列表中删除添加到测试集的索引值
	trainmat = []
	trainclasses = []     #创建训练集矩阵和训练集类别标签系向量
	for docindex in trainset:      #遍历训练集
	    trainmat.append(setOfWords2Vec(vList, doclist[docindex]))      #将生成的词袋模型添加到训练矩阵中
	    trainclasses.append(classlist[docindex])     #将类别添加到训练集类别标签系向量中
	p0V, p1V, pspam = trainNB0(np.array(trainmat), np.array(trainclasses))     #训练朴素贝叶斯模型
	errorCount = 0     #错误分类计数
	for docindex in testset:     #遍历测试集
	    wordvector = setOfWords2Vec(vList, doclist[docindex])     #测试集的词袋模型
	    if classifyNB(np.array(wordvector), p0V, p1V, pspam) != classlist[docindex]:     #如果分类错误
	        errorCount += 1     #错误计数+1
	        print('错误率：', float(errorCount) / len(testset))
	        print('错误个数：', errorCount)
	        print('测试集长度：', len(testset))

def testParsetest():
	print(textParse(open('第四课训练样本/email/ham/%d.txt').read()))

if __name__ == '__main__':
	spamtest()