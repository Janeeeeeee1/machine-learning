import trees45

myDat,labels= trees45.createDataSet()
myTree = trees45.createTree(myDat, labels)
print(myTree)