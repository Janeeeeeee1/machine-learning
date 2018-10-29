from math import log
import operator

def createDataSet():
    '''
    function: 创建数据集
    returns: dataSet: 数据集
             labels: 标签
    '''

    #创建数据集
    dataSet = [['sunny', 'hot', 'high', 'false', 'no'],
                ['sunny', 'hot', 'high', 'true', 'no'],
                ['overcast', 'hot', 'high', 'false', 'yes'],
                ['rainy', 'mild', 'high', 'false', 'yes'],
                ['rainy', 'cool', 'normal', 'false', 'yes'],
                ['rainy', 'cool', 'normal', 'true', 'no'],
                ['overcast', 'cool', 'normal', 'true', 'yes'],
                ['sunny', 'mild', 'high', 'false', 'no'],
                ['sunny', 'cool', 'normal', 'false', 'yes'],
                ['rainy', 'mild', 'normal', 'false', 'yes'],
                ['sunny', 'mild', 'normal', 'true', 'yes'],
                ['overcast', 'mild', 'high', 'true', 'yes'],
                ['overcast', 'hot', 'normal', 'false', 'yes'],
                ['rainy', 'mild', 'high', 'true', 'no']]
    #创建标签
    labels=['Outlook','Temperature','Humidity','Windy']
    #返回创建的数据集和标签
    return dataSet,labels

def calcShannonEnt(dataSet):
    '''
    function: 计算香农熵
    param dataSet: 数据集
    return: 香农熵
    '''

    #计算数据集中实例的总书
    numEntries = len(dataSet)
    #创建一个数据字典
    labelCounts = {}
    #为所有可能的分类创造字典
    for featVec in dataSet:
        #字典的健等于最后一列的值
        currentLabel = featVec[-1]
        #如果当前健不存在，则扩展字典把当前健加入字典中
        if currentLabel not in labelCounts.keys():
            #为当前健赋值
            labelCounts[currentLabel]=0
        #每个健对应的值记录的是当前类别出现的次数
        labelCounts[currentLabel]+=1
    #初始化香农熵
    shannonEnt=0.0
    #计算香农熵
    for key in labelCounts:
        #利用每个类别发生次数和总实例数，来计算每种类别发生的可能性
        prob = float(labelCounts[key])/numEntries
        #计算香农熵，log(prob,2)，以2为底数求prob的对数
        shannonEnt -= prob * log(prob,2)
    #返回香农熵
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    '''
    function: 按照给定特征划分数据集
    param dataSet: 待划分的数据集
    param axis: 划分数据集的特征
    param value: 需要返回的特征的值
    return: retDataSet：符合特征的数据集
    '''

    #创建新的list对象
    retDataSet=[]
    # 抽取数据集
    for featVec in dataSet:
        # 将符合特征的数据抽取出来
        if featVec[axis] == value:
            # 截取列表中第axis+1个之前的数据
            reducedFeatVec = featVec[:axis]
            # 将第axis+2之后的数据接入到上述数据集
            reducedFeatVec.extend(featVec[axis + 1:])
            # 将处理结果作为列表接入到返回数据集
            retDataSet.append(reducedFeatVec)
    # 返回符合特征的数据集
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    function: 选择最好的数据集划分方式
    param dataSet: 待划分的数据集
    return: bestFeature：划分数据集最好的特征
    '''

    # 初始化特征数量
    numFeatures = len(dataSet[0]) - 1
    # 计算原始香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 初始化信息增益和最佳特征
    bestInfoGain = 0.0
    bestFeature = -1
    #选出最好的划分数据集的特征
    for i in range(numFeatures):
        #创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        #创建集合去重，以得到列表中唯一的元素值
        uniqueVals = set(featList)
        #初始化香农熵
        newEntropy = 0.0
        #初始化因素自身信息增益
        splitInfo = 0.0
        #计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            #计算这个属性的分类信息熵
            splitInfo -= prob * log(prob,2)
        #得到信息增益率
        infoGain = (baseEntropy - newEntropy)/splitInfo
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    #返回划分最好的特征
    return bestFeature

def majorityCnt(classList):
    '''
    function: 决定叶子节点的分类
    :param classList: 分类列表
    :return: sortedClassCount[0][0]：叶子结点分类结果
    '''

    # 创建字典
    classCount = {}
    # 给字典赋值
    for vote in classList:
        # 如果字典中没有该键值，则创建
        if vote not in classCount.keys():
            classCount[vote] = 0
        # 为每个键值计数
        classCount[vote] += 1
    # 对classCount进行排序,sorted by 值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回叶子结点分类结果
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    '''
    function: 创建树
    :param dataSet: 数据集
    :param labels: 标签列表
    :return: myTree：创建的树的信息
    '''

    # 创建分类列表
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #如果只有一个因素了，则直接返回叶子节点分类结果
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选取最好的分类特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 创建字典存储树的信息
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    # 从列表中创建集合
    uniqueVals = set(featValues)
    # 遍历当前选择特征包含的所有属性值
    for value in uniqueVals:
        # 复制类标签
        subLabels = labels[:]
        # 递归调用函数createTree()，返回值将被插入到字典变量myTree中
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    # 返回字典变量myTree
    return myTree

def storeTree(inputTree,filename):
    import pickle
    #新建文件
    fw = open(filename,'wb')
    #写入数据
    pickle.dump(inputTree,fw,0)
    fw.close()

def grabTree(filename):
    import pickle
    #打开文件
    fr=open(filename,'rb')
    #导出数据
    return pickle.load(fr)