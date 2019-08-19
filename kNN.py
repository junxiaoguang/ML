from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

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

if __name__ == "__main__":
    group, labels = createDataSet()
    result = classify0([0, 0], group, labels, 3)
    print(result)



