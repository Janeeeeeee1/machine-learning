import numpy as np


# 定义树结构，采用的二叉树，左子树：条件为true，右子树：条件为false
# leftBranch:左子树结点
# rightBranch:右子树结点
# col:信息增益最大时对应的列索引
# value:最优列索引下，划分数据类型的值
# results:分类结果
# summary:信息增益最大时样本信息
# data:信息增益最大时数据集
class Tree:
    def __init__(self, leftBranch=None, rightBranch=None, col=-1, value=None, results=None, summary=None, data=None):
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.col = col
        self.value = value
        self.results = results
        self.summary = summary
        self.data = data

    def __str__(self):
        print(u"列号：%d" % self.col)
        print(u"列划分值：%s" % self.value)
        print(u"样本信息：%s" % self.summary)
        return ""

# 划分数据集
def splitDataSet(dataSet, value, column):
    leftList = []
    rightList = []
    # 判断value是否是数值型
    if (isinstance(value, int) or isinstance(value, float)):
        # 遍历每一行数据
        for rowData in dataSet:
            # 如果某一行指定列值>=value，则将该行数据保存在leftList中，否则保存在rightList中
            if (rowData[column] >= value):
                leftList.append(rowData)
            else:
                rightList.append(rowData)
    # value为标称型
    else:
        # 遍历每一行数据
        for rowData in dataSet:
            # 如果某一行指定列值==value，则将该行数据保存在leftList中，否则保存在rightList中
            if (rowData[column] == value):
                leftList.append(rowData)
            else:
                rightList.append(rowData)
    return leftList, rightList


# 统计标签类每个样本个数
'''
该函数是计算gini值的辅助函数，假设输入的dataSet为为['A', 'B', 'C', 'A', 'A', 'D']，
则输出为['A':3,' B':1, 'C':1, 'D':1]，这样分类统计dataSet中每个类别的数量
'''


def calculateDiffCount(dataSet):
    results = {}
    for data in dataSet:
        # data[-1] 是数据集最后一列，也就是标签类
        if data[-1] not in results:
            results.setdefault(data[-1], 1)
        else:
            results[data[-1]] += 1
    return results


# 基尼指数公式实现
def gini(dataSet):
    # 计算gini的值(Calculate GINI)
    # 数据所有行
    length = len(dataSet)
    # 标签列合并后的数据集
    results = calculateDiffCount(dataSet)
    imp = 0.0
    for i in results:
        imp += results[i] / length * results[i] / length
    return 1 - imp


# 生成决策树
'''算法步骤'''
'''根据训练数据集，从根结点开始，递归地对每个结点进行以下操作，构建二叉决策树：
1 设结点的训练数据集为D，计算现有特征对该数据集的信息增益。此时，对每一个特征A，对其可能取的
  每个值a，根据样本点对A >=a 的测试为“是”或“否”将D分割成D1和D2两部分，利用基尼指数计算信息增益。
2 在所有可能的特征A以及它们所有可能的切分点a中，选择信息增益最大的特征及其对应的切分点作为最优特征
  与最优切分点，依据最优特征与最优切分点，从现结点生成两个子结点，将训练数据集依特征分配到两个子结点中去。
3 对两个子结点递归地调用1,2，直至满足停止条件。
4 生成CART决策树。
'''''''''''''''''''''


# evaluationFunc= gini :采用的是基尼指数来衡量信息关注度
def buildDecisionTree(dataSet, evaluationFunc=gini):
    # 计算基础数据集的基尼指数
    baseGain = evaluationFunc(dataSet)
    # 计算每一行的长度（也就是列总数）
    columnLength = len(dataSet[0])
    # 计算数据项总数
    rowLength = len(dataSet)
    # 初始化
    bestGain = 0.0  # 信息增益最大值
    bestValue = None  # 信息增益最大时的列索引，以及划分数据集的样本值
    bestSet = None  # 信息增益最大，听过样本值划分数据集后的数据子集
    # 标签列除外（最后一列），遍历每一列数据
    for col in range(columnLength - 1):
        # 获取指定列数据
        colSet = [example[col] for example in dataSet]
        # 获取指定列样本唯一值
        uniqueColSet = set(colSet)
        # 遍历指定列样本集
        for value in uniqueColSet:
            # 分割数据集
            leftDataSet, rightDataSet = splitDataSet(dataSet, value, col)
            # 计算子数据集概率，python3 "/"除号结果为小数
            prop = len(leftDataSet) / rowLength
            # 计算信息增益
            infoGain = baseGain - prop * evaluationFunc(leftDataSet) - (1 - prop) * evaluationFunc(rightDataSet)
            # 找出信息增益最大时的列索引，value,数据子集
            if (infoGain > bestGain):
                bestGain = infoGain
                bestValue = (col, value)
                bestSet = (leftDataSet, rightDataSet)
    # 结点信息
    #    nodeDescription = {'impurity:%.3f'%baseGain,'sample:%d'%rowLength}
    nodeDescription = {'impurity': '%.3f' % baseGain, 'sample': '%d' % rowLength}
    # 数据行标签类别不一致，可以继续分类
    # 递归必须有终止条件
    if bestGain > 0:
        # 递归，生成左子树结点，右子树结点
        leftBranch = buildDecisionTree(bestSet[0], evaluationFunc)
        rightBranch = buildDecisionTree(bestSet[1], evaluationFunc)
        return Tree(leftBranch=leftBranch, rightBranch=rightBranch, col=bestValue[0]
                    , value=bestValue[1], summary=nodeDescription, data=bestSet)
    else:
        # 数据行标签类别都相同，分类终止
        return Tree(results=calculateDiffCount(dataSet), summary=nodeDescription, data=dataSet)


def createTree(dataSet, evaluationFunc=gini):
    # 递归建立决策树， 当gain=0，时停止回归
    # 计算基础数据集的基尼指数
    baseGain = evaluationFunc(dataSet)
    # 计算每一行的长度（也就是列总数）
    columnLength = len(dataSet[0])
    # 计算数据项总数
    rowLength = len(dataSet)
    # 初始化
    bestGain = 0.0  # 信息增益最大值
    bestValue = None  # 信息增益最大时的列索引，以及划分数据集的样本值
    bestSet = None  # 信息增益最大，听过样本值划分数据集后的数据子集
    # 标签列除外（最后一列），遍历每一列数据
    for col in range(columnLength - 1):
        # 获取指定列数据
        colSet = [example[col] for example in dataSet]
        # 获取指定列样本唯一值
        uniqueColSet = set(colSet)
        # 遍历指定列样本集
        for value in uniqueColSet:
            # 分割数据集
            leftDataSet, rightDataSet = splitDataSet(dataSet, value, col)
            # 计算子数据集概率，python3 "/"除号结果为小数
            prop = len(leftDataSet) / rowLength
            # 计算信息增益
            infoGain = baseGain - prop * evaluationFunc(leftDataSet) - (1 - prop) * evaluationFunc(rightDataSet)
            # 找出信息增益最大时的列索引，value,数据子集
            if (infoGain > bestGain):
                bestGain = infoGain
                bestValue = (col, value)
                bestSet = (leftDataSet, rightDataSet)

    impurity = u'%.3f' % baseGain
    sample = '%d' % rowLength

    if bestGain > 0:
        bestFeatLabel = u'serial:%s\nimpurity:%s\nsample:%s' % (bestValue[0], impurity, sample)
        myTree = {bestFeatLabel: {}}
        myTree[bestFeatLabel][bestValue[1]] = createTree(bestSet[0], evaluationFunc)
        myTree[bestFeatLabel]['no'] = createTree(bestSet[1], evaluationFunc)
        return myTree
    else:  # 递归需要返回值
        bestFeatValue = u'%s\nimpurity:%s\nsample:%s' % (str(calculateDiffCount(dataSet)), impurity, sample)
        return bestFeatValue


# 分类测试：
'''根据给定测试数据遍历二叉树，找到符合条件的叶子结点'''
'''例如测试数据为[5.9,3,4.2,1.75]，按照训练数据生成的决策树分类的顺序为
   第2列对应测试数据4.2 =>与决策树根结点（2）的value（3）比较，>=3则遍历左子树，否则遍历右子树，
   叶子结点就是结果'''


def classify(data, tree):
    # 判断是否是叶子结点，是就返回叶子结点相关信息，否就继续遍历
    if tree.results != None:
        return u"%s\n%s" % (tree.results, tree.summary)
    else:
        branch = None
        v = data[tree.col]
        # 数值型数据
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.leftBranch
            else:
                branch = tree.rightBranch
        else:  # 标称型数据
            if v == tree.value:
                branch = tree.leftBranch
            else:
                branch = tree.rightBranch
        return classify(data, branch)


def loadCSV(fileName):
    def convertTypes(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s

    data = np.loadtxt(fileName, dtype='str', delimiter=',')
    data = data[1:, :]
    dataSet = ([[convertTypes(item) for item in row] for row in data])
    return dataSet


# 多数表决器
# 列中相同值数量最多为结果
def majorityCnt(classList):
    import operator
    classCounts = {}
    for value in classList:
        if (value not in classCounts.keys()):
            classCounts[value] = 0
        classCounts[value] += 1
    sortedClassCount = sorted(classCounts.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 剪枝算法(前序遍历方式：根=>左子树=>右子树)
'''算法步骤
1. 从二叉树的根结点出发，递归调用剪枝算法，直至左、右结点都是叶子结点
2. 计算父节点（子结点为叶子结点）的信息增益infoGain
3. 如果infoGain < miniGain，则选取样本多的叶子结点来取代父节点
4. 循环1,2,3，直至遍历完整棵树
'''''''''


def prune(tree, miniGain, evaluationFunc=gini):
    print(u"当前结点信息:")
    print(str(tree))
    # 如果当前结点的左子树不是叶子结点，遍历左子树
    if (tree.leftBranch.results == None):
        print(u"左子树结点信息:")
        print(str(tree.leftBranch))
        prune(tree.leftBranch, miniGain, evaluationFunc)
    # 如果当前结点的右子树不是叶子结点，遍历右子树
    if (tree.rightBranch.results == None):
        print(u"右子树结点信息:")
        print(str(tree.rightBranch))
        prune(tree.rightBranch, miniGain, evaluationFunc)
    # 左子树和右子树都是叶子结点
    if (tree.leftBranch.results != None and tree.rightBranch.results != None):
        # 计算左叶子结点数据长度
        leftLen = len(tree.leftBranch.data)
        # 计算右叶子结点数据长度
        rightLen = len(tree.rightBranch.data)
        # 计算左叶子结点概率
        leftProp = leftLen / (leftLen + rightLen)
        # 计算该结点的信息增益（子类是叶子结点）
        infoGain = (evaluationFunc(tree.leftBranch.data + tree.rightBranch.data) -
                    leftProp * evaluationFunc(tree.leftBranch.data) - (1 - leftProp) * evaluationFunc(
                    tree.rightBranch.data))
        # 信息增益 < 给定阈值，则说明叶子结点与其父结点特征差别不大，可以剪枝
        if (infoGain < miniGain):
            # 合并左右叶子结点数据
            dataSet = tree.leftBranch.data + tree.rightBranch.data
            # 获取标签列
            classLabels = [example[-1] for example in dataSet]
            # 找到样本最多的标签值
            keyLabel = majorityCnt(classLabels)
            # 判断标签值是左右叶子结点哪一个
            if keyLabel in tree.leftBranch.results:
                # 左叶子结点取代父结点
                tree.data = tree.leftBranch.data
                tree.results = tree.leftBranch.results
                tree.summary = tree.leftBranch.summary
            else:
                # 右叶子结点取代父结点
                tree.data = tree.rightBranch.data
                tree.results = tree.rightBranch.results
                tree.summary = tree.rightBranch.summary
            tree.leftBranch = None
            tree.rightBranch = None
