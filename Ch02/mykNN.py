from numpy import *
import operator


def createdata():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lable = ['A', 'A', 'B', 'B']
    return group, lable


def classify0(inX,dataSet,lables,k):
    # 计算已知类别数据集中的点和当前之间的距离 shape 矩阵每维的大小  size:矩阵所有数据的个数  dtype:矩阵每个数据的类型
    datasetsize = dataSet.shape[0]
    # tile:重复向量intX在行和列上datasize和1次
    # 把inX向量在行的维度扩大到和dataset一样大的维度，再计算他们之间的差
    print(tile(inX, (datasetsize, 1)))
    diffmat = tile(inX, (datasetsize, 1)) - dataSet
    print(diffmat)
    # 求平方
    sqdiffmat = diffmat**2
    print(sqdiffmat)
    # 默认的axis=0 就是普通的相加 axis=1以后就是将一个矩阵的每一行向量相加
    # a = np.array([[0, 2, 1]]) a.sum() / a.sum(axis=0) / a.sum(axis=1) 结果：3, [0 1 2], [3]
    sqDistance = sqdiffmat.sum(axis=1)
    print(sqDistance)
    distance = sqDistance ** 0.5
    sortDistanceIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        # 得到分组
        voteLabel = lables[sortDistanceIndicies[i]]
        # 分组计数
        classCount[voteLabel] = classCount.get(voteLabel , 0) + 1
    sortClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(sortClassCount)



group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
classify0([1.0,1.2],group, ['A', 'A', 'B', 'B'],4)