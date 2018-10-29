import treePlotter
import trees

myDat,labels= trees.createDataSet()
myTree = trees.createTree(myDat, labels)
print(myTree)

treePlotter.createPlot(myTree)

trees.storeTree(myTree, 'myStoreTree')


