import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readfiles(path):
    for root, dirnames, filenames, in os.walk(path):     #返回path下的所有文件夹和文件
        for filename in filenames:
            path = os.path.join(root, filename)     #创建path
            inBody = False     #标记头部信息
            lines = []
            f = io.open(path, 'r', encoding = 'latin-1')     #打开文件
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':     #先把头部信息跳过,找到第一个空行 表明头部信息已经结束
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readfiles(path):
        rows.append({'message':message, 'class':classification})
        index.append(filename)
    return DataFrame(rows, index = index)

data = DataFrame({'message':[], 'class':[]})

data = data.append(dataFrameFromDirectory('第四课训练样本/email/ham/%d.txt', 'ham'))
data = data.append(dataFrameFromDirectory('第四课训练样本/email/spam/%d.txt', 'spam'))

data.head()

vecotrizer = CountVectorizer()
counts = vecotrizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

examples = ['Free Money now!!!', "Hi, I'm Sitch,how about make a friend?"]
examples_coutns = vecotrizer.transform(examples)
predictions = classifier.predict(examples_coutns)

train_data = data.sample(frac = 0.6)
test_data = data[~data.index.isin(train_data.index)]

vecotrizer = CountVectorizer()
counts = vecotrizer.fit_transform(train_data['message'].values)
classifier = MultinomialNB()
targets = train_data['class'].values
classifier.fit(counts, targets)

i = 0
test_counts = vecotrizer.transform(test_data['message'].values)
predictions = classifier.predict(test_counts)
for i in range(0, predictions.size):
    if test_data['class'].values[i] == predictions[i]:
        i += 1
print("测试集成功率", end = ':')
print(i/predictions.size)