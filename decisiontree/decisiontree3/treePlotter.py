import matplotlib.pyplot as plt

def getNumLeaves(myTree):
    '''
    function:得到叶子节点的个数
    :param myTree:  决策树的数据
    :return:  叶子节点的个数
    '''

    # 初始化树的叶子节点个数
    numLeaves = 0
    #得到健
    firstStr=list(myTree.keys())[0]
    #得到键对应的值
    secondDict = myTree[firstStr]
    #如果是字典则继续递归调用getNumLeaves(),如果不是字典则numLeaves加一
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeaves += getNumLeaves(secondDict[key])
        else:
            numLeaves +=1
    return numLeaves

def getTreeDepth(myTree):
    '''
    function: 得到树的深度
    :param myTree: 决策树数据
    :return:  树的深度
    '''

    #初始化最大深度
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    #判断子节点是否为字典，是则递归调用getTreeDepth(),否则thisDepth=1
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        #把每一个分支的深度进行比较，最大的赋给maxDepth
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

#设置画节点的盒子样式
decisionNode = dict(boxstyle = "sawtooth",fc="0.8")
leafNode = dict(boxstyle = "round4",fc="0.8")
#设置画箭头的样式
arrow_args = dict(arrowstyle="<-")
#绘图相关参数的设置

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
	Function：	绘制带箭头的注解
	Args：		nodeTxt：文本注解
				centerPt：箭头终点坐标
				parentPt：箭头起始坐标
				nodeType：文本框类型
	Returns：	无
	"""
    #在全局变量createPlot.ax1中绘图
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
			 xytext=centerPt, textcoords='axes fraction',
			 va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    #在全局变量createPlot0.ax1中绘图
#	createPlot0.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
#			 xytext=centerPt, textcoords='axes fraction',
#			 va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )


def plotMidText(cntrPt, parentPt, txtString):
    """
    Function：   在父子节点间填充文本信息

    Args：       cntrPt：树信息
                parentPt：父节点坐标
                txtString：文本注解

    Returns：    无
    """
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    """
    Function：   创建数据集和标签

    Args：       myTree：树信息
                parentPt：箭头起始坐标
                nodeTxt：文本注解

    Returns：    无
    """
    #计算树的宽
    numLeaves = getNumLeaves(myTree)  #this determines the x width of this tree
    #计算树的高
    depth = getTreeDepth(myTree)
    #第一个关键字为第一次划分数据集的类别标签，附带的取值表示子节点的取值
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    #下一个节点的位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeaves))/2.0/plotTree.totalW, plotTree.yOff)
    #计算父节点和子节点的中间位置，并在此处添加简单的文本信息
    plotMidText(cntrPt, parentPt, nodeTxt)
    #绘制此节点带箭头的注解
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    #新的树，相当于脱了一层皮
    secondDict = myTree[firstStr]
    #按比例减少全局变量plotTree.yOff
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    #
    for key in secondDict.keys():
        #判断子节点是否为字典类型
        if type(secondDict[key]).__name__=='dict':
            #是的话表明该节点也是一个判断节点，递归调用plotTree()函数
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            #不是的话更新x坐标值
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            #绘制此节点带箭头的注解
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            #绘制此节点带箭头的注解
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    #按比例增加全局变量plotTree.yOff
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    """
    Function：   使用文本注解绘制树节点

    Args：       inTree：

    Returns：    无
    """
    #创建一个新图形
    fig = plt.figure(1, facecolor='white')
    #清空绘图区
    fig.clf()
    #创建一个字典
    axprops = dict(xticks=[], yticks=[])
    #给全局变量createPlot.ax1赋值
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    #取得叶节点数目
    plotTree.totalW = float(getNumLeaves(inTree))
    #取得树最大层
    plotTree.totalD = float(getTreeDepth(inTree))
    #设置起点值
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    #绘制数
    plotTree(inTree, (0.5,1.0), '')
    #显示最终绘制结果
    plt.show()