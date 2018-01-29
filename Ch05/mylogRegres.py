import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmoid(intX):
    return 1.0/(1+np.exp(-intX))


def gradAscent(dataMatIn,classLabels):
    # 100*3的向量
    dataMatrix = np.mat(dataMatIn)
    #print(dataMatrix.transpose())
    # 将矩阵转置 100*1的向量
    labelMat = np.mat(classLabels).transpose()
    # 数据行和列的维度
    m,n = np.shape(dataMatrix)
    # 向目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    # ones 跟zeros一样 将每行的权重都初始化为1
    weights = np.ones((n,1))
    for k in range(maxCycles):
        # 数据每列乘以权重后计算与目标的差值
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h) # 100*1的向量
        # 权重迭代修改 3*100向量和100*1向量相乘
        weights = weights + alpha * dataMatrix.transpose() * error
        if k == 499 :
            #print(dataMatrix.transpose() * error)
            #print(dataMatrix.transpose())
            #print(error)
            pass
    return weights


# 随机梯度上升
def randomGradAscent(dataMat,classLabels):
    m,n = np.shape(dataMat)
    alpha = 0.01
    weight = np.ones((n,1))
    for i in range(m):
        h = sigmoid(dataMat[i] * weight)
        #print(h)
        error = classLabels[i] - h
        weight = weight + alpha*error*dataMat[i]
    print(weight)
    return weight


def plotBestFit(wei):
    # 矩阵转为array
    weight = wei.getA()
    dataMat,labelMat=loadDataSet()
    # 100*3 数组
    dataArr = np.array(dataMat)
    # 100行
    n = np.shape(dataArr)[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        # 把不同类型数据用不同颜色分开
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    # numpy.arange([start, ]stop, [step, ]dtype=None) 下面代码意思为：-3.0 到 3.0之间 步长为0.1
    x = np.arange(-3.0,3.0,0.1)
    y = (-weight[0]-weight[1]*x)/weight[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


dataMat,labelMat=loadDataSet()
#print(gradAscent(dataMat,labelMat))
plotBestFit(randomGradAscent(dataMat,labelMat))
