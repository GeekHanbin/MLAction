import operator
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
    # 计算每个概率
    for i in labelCount.keys():
        prob = float(labelCount[i])/numEntropy
        totalEntropy -= prob * log(prob,2)
    return totalEntropy


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


'''
dataset:数据集
axis:划分数据集特征
value:特征返回值
'''
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        # 取出第i列中数据
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            # 计算这一列中每个独立的value对应的概率
            prob = len(subDataSet)/float(len(dataSet))
            # 信息熵为：这列概率*这个数据集的熵
            newEntropy += prob * calEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain :
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 计算classList中次数出现最多的条数
def majortyCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 倒序排列
    sortClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortClassCount[0][0]



def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同停止划分
    #print(classList)
    # classList.count(classList[0]) 计算classList中第一个元素在classlist中的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #
    if len(dataSet[0]) == 1:
        return majortyCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #print(bestFeat)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel :{} }
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


# 决策树对象存储和读取
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()


def getTree(filename):
    import pickle
    fw = open(filename)
    return pickle.load(fw)


def lensesFunction():
    import treePlotter
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    lensesTree = createTree(lenses,lensesLabels)
    #print(lensesTree)
    treePlotter.createPlot(lensesTree)

dataSet,labels = createDataSet()
lensesFunction()