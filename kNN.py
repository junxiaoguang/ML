from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def datingClassTest():
    filename = "datingTestSet2.txt"
    horatio = 0.1
    dataingMat, dataingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(dataingMat)
    #datingTestSet2.txt 获取矩阵的行数
    m = normMat.shape[0]
    num_oftest = int(m*horatio)
    errorCount = 0.0
    for i in range(num_oftest):
        classifierResult = classify0(normMat[i, :], normMat[num_oftest:m, :], dataingLabels[num_oftest:m], 3)
        print('the classifierResult is %d, the real result is %d'%(classifierResult, dataingLabels[i]))
        if(classifierResult != dataingLabels[i]):
            errorCount += 1.0
    print('the total error rate is %%' '%f' % (errorCount/float(num_oftest)*100))

def classifyperson():
    resultList = ['not at all', 'is small doses', 'in large doses']
    percentTats = float(input('percentage of time wpent playing video games?'))# 输入在游戏上花费的时间。
    ffmiles = float(input('frequent flier miles earned per year?'))
    iccream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffmiles, percentTats, iccream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print('you will probable like this guy:', resultList[classifierResult-1])

def classify0(inX, dataSet, labels, k):
    # 返回的是dataSet矩阵第一维的长度即矩阵的行数
    dataSetSize = dataSet.shape[0]

    # tile 函数是把inX向量补成和矩阵dataSet一样的矩阵。
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    # 下面几行代码则是求给定待测数据向量与训练集样本的欧式距离
    sqdiffMat = diffMat**2
    # 对矩阵进行安行求和  axis=0则表示安列求和
    sq_dist = sqdiffMat.sum(axis=1)
    distances = sq_dist**0.5

    # 对数组进行从小到大排序，排序结果返回的数组的序号
    dis_sort_indicies = distances.argsort()

    # 新建一个字典
    class_count={}
    for i in range(k):
        votedlabel = labels[dis_sort_indicies[i]]
        class_count[votedlabel] = class_count.get(votedlabel, 0)+1

    sortedClassCount = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 从文件中获取数据并转换成待使用的格式
def file2matrix(filename):
    fr = open(filename)
    # 按行读取数据
    arrayOlines = fr.readlines()
    # 统计数据行数
    numberOfLines = len(arrayOlines)

    returnMat = zeros((numberOfLines, 3))
    classLabelVector=[]
    index = 0
    for line in arrayOlines:
        # 去掉每一行的空格
        line = line.strip()
        listFormLine = line.split('\t')
        returnMat[index, :] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    # 返回每一列中的最小值。 ps dataSet.min(1)是返回每一列的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    normDataSet = zeros(shape(dataSet))

    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


if __name__ == "__main__":
    #classifyperson()
    # datingClassTest()
    # group, labels = createDataSet()
    # result = classify0([0.5, 0.5], group, labels, 3)
    #
    # returnMat, dataingLabels = file2matrix('datingTestSet2.txt')
    #
    # normDataSet, ranges, minVals = autoNorm(returnMat)
    #
    # print(normDataSet)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(returnMat[:, 0], returnMat[:, 1], 15.0*array(dataingLabels), 15.0*array(dataingLabels))
    # plt.show()



