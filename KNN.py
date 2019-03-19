import numpy as np
def knn(k,X,Y,input):
    '''
    :param k: k neighbours
    :param X: dataset
    :param Y: labels
    :param input: the sample we want to predict
    :return: label of input
    '''
    Distance =  np.sqrt(np.sum((input-X)**2,axis=1))
    sortedDistance = np.argsort(Distance)  #从小到大排，返回索引
    dict={}
    for i in range(k):
        y = Y[sortedDistance[i]]
        dict[y] += 1
    sorteddict = sorted(dict.items(),key=lambda x:-x[1])
    return sorteddict[0][0]


