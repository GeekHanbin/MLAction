from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


def createdata():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lable = ['A', 'A', 'B', 'B']
    return group, lable


def classify0(inX,dataSet,lables,k):
    # 计算已知类别数据集中的点和当前之间的距离 shape 矩阵每维的大小  size:矩阵所有数据的个数  dtype:矩阵每个数据的类型
    datasetsize = dataSet.shape[0]
    # tile:重复向量intX在行和列上datasize和1次
    # 把inX向量在行的维度扩大到和dataset一样大的维度，再计算他们之间的差
    # print(tile(inX, (datasetsize, 1)))
    diffmat = tile(inX, (datasetsize, 1)) - dataSet
    # print(diffmat)
    # 求平方
    sqdiffmat = diffmat**2
    # print(sqdiffmat)
    # 默认的axis=0 就是普通的相加 axis=1以后就是将一个矩阵的每一行向量相加
    # a = np.array([[0, 2, 1]]) a.sum() / a.sum(axis=0) / a.sum(axis=1) 结果：3, [0 1 2], [3]
    sqDistance = sqdiffmat.sum(axis=1)
    # print(sqDistance)
    distance = sqDistance ** 0.5
    sortDistanceIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        # 得到分组
        voteLabel = lables[sortDistanceIndicies[i]]
        # 分组计数
        classCount[voteLabel] = classCount.get(voteLabel , 0) + 1
    sortClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # print(sortClassCount[0][0])
    return sortClassCount[0][0]


# change the file data to matrix
def file2matrix(filename):
    file = open(filename)
    arrayLines = file.readlines()
    numberOfLines = len(arrayLines)
    # 生成以行为维度 3列的 0矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split("\t")
        # 矩阵其实就是多维数组，下面是将returnMat第index位设置为listFromLine的数组
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


# 将数据展示
def displayDataWithmatplotlib():
    datingMat, dataLabels = file2matrix("datingTestSet2.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(datingMat[:, 1])
    # 选择datingMat第一列数据和第二列数据作为展示
    # ax.scatter(datingMat[:, 1], datingMat[:, 2], 15 * array(dataLabels), 15 * array(dataLabels))
    ax.scatter(datingMat[:, 1], 15 * array(dataLabels))
    plt.show()


# 数据归一化 newValue = (oldValue - min)/(max - min)
def autoNorm(dataset):
    '''
    a = np.array([[1,5,3],[4,2,6]])
    print(a.min()) #无参，所有中的最小值
    print(a.min(0)) # axis=0; 每列的最小值
    print(a.min(1)) # axis=1；每行的最小值
    结果：
    1
    [1 2 3]
    [1 2]
    '''
    minValues = dataset.min(0)
    maxValues = dataset.max(0)
    range = maxValues - minValues
    normDataSet = zeros(shape(dataset))
    # 取行的维度
    m = dataset.shape[0]
    normDataSet = dataset - tile(minValues,(m,1))
    normDataSet = dataset / tile(range,(m,1))
    return normDataSet,range,minValues

def datingClass():
    # 测试数据比率
    testRate = 0.1
    datingMat, dataLabels = file2matrix("datingTestSet2.txt")
    normDataSet, ranges, minValues = autoNorm(datingMat)
    m = normDataSet.shape[0]
    numOfTestVect = int(m*testRate)
    errorCount  = 0.0
    for i in range(numOfTestVect):
        classfierResult = classify0(normDataSet[i,:],normDataSet[numOfTestVect:m,:],dataLabels[numOfTestVect:m],3)
        print("classfier result is :" + str(classfierResult))
        if classfierResult != dataLabels[i]:
            errorCount += 1.0
    print("error rate is :" + str(errorCount/numOfTestVect))


def classfiyPerson():
    resultList = ['do not like','just soso','very like']
    percentTag = float(input("percentage of time spent play game"))
    ffmiles = float(input("ffmiles"))
    icecream = float(input("icecream"))
    print(str(percentTag)+str(ffmiles)+str(icecream))
    datingMat, dataLabels = file2matrix("datingTestSet2.txt")
    inArray = array([percentTag,ffmiles,icecream])
    classfyResult = classify0(inArray,datingMat,dataLabels,3)
    print("your result is:"+resultList[classfyResult-1])


# 图像为32*32的格式，转化为1*1024的格式
def img2vector(filename):
    returnVector = zeros((1,1024))
    fi = open(filename)
    for i in range(32):
        line = fi.readline()
        for j in range(32):
            returnVector[0,32*i + j] = str(line[j])
    return returnVector

# 测试图像文本分类
def classfyImag():
    # 读取训练集并保存到矩阵中
    fileList = listdir('trainingDigits')
    m = len(fileList)
    trainMat = zeros((m,1024))
    trainLabel = []
    for i in range(m):
        # 读取标签
        filename = fileList[i]
        filelabel = filename.split("_")[0]
        trainLabel.append(int(filelabel))
        trainMat[i,:] = img2vector('trainingDigits/'+filename)
    # 读取测试数据并输出
    testFileList = listdir('testDigits')
    testm = len(testFileList)
    errorCount = 0.0
    for i in range(testm):
        filename = testFileList[i]
        testLabel = int(filename.split("_")[0])
        testVector = img2vector('testDigits/'+filename)
        classfyResult = classify0(testVector,trainMat,trainLabel,3)
        if classfyResult != testLabel:
            errorCount += 1.0
    print('errorCount:'+str(errorCount))
    print('errorRate:'+str(errorCount/float(testm)))


classfyImag()