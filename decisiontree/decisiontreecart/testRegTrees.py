import regTrees as rt
import plotRegTrees as pt

if __name__ == '__main__':
    dataSet = rt.loadCSV("dataSet.csv")
    myTree = rt.createTree(dataSet, evaluationFunc=rt.gini)
    print(u"myTree:%s"%myTree)
    #绘制决策树
    print(u"绘制决策树：")
    pt.createPlot1(myTree)
    decisionTree = rt.buildDecisionTree(dataSet, evaluationFunc=rt.gini)
    testData = [5.9,3,4.2,1.75]
    r = rt.classify(testData, decisionTree)
    print(u"分类后测试结果:")
    print(r)
    print()
    rt.prune(decisionTree, 0.4)
    r1 = rt.classify(testData, decisionTree)
    print(u"剪枝后测试结果:")
    print(r1)