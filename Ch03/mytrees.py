from math import log


def calEntropy(dataset):
    '''
    计算熵
    dataset数据类型
    [[1,1,'yes'],[2,2,'yes'],[1,0,'no']]
    '''
    numEntropy = len(dataset)
    # 存储每个label对应的数量，用于计算
    labelCount = {}
    totalEntropy = 0.0
    for vect in dataset:
        label = vect[-1]
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1
    print(labelCount)
    # 计算每个概率
    for i in labelCount.keys():
        print(i)
        prob = float(labelCount[i])/numEntropy
        totalEntropy -= prob * log(prob,2)
    return totalEntropy


def createDataSet():
    dataSet = [[1, 1, 'maybe'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

dataSet,labels = createDataSet()
print(calEntropy(dataSet))