import numpy as np
import matplotlib.pyplot as plt
"""
1.随机取K个中心点
2.计算所有点到中心点的距离，将所有点分别放入中心点所在的簇
3.更新中心点
4.迭代，直到中心点不变，结束迭代
"""

#获取数据集
def loadDataset(filename,delimiter):
    return np.loadtxt(filename,delimiter=delimiter)

#取出K个中心点
def init_centers(dataset,k):
    centersIndex = np.random.choice(len(dataset),k,replace=False)  #无放回的随机选择
    return dataset[centersIndex]

#计算距离
def distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

#kmeans
def K_means(dataset,k):
    """
    :param dataset: dataset
    :param k:  number of cluster
    :return:  one-dimensional array containing the clustered index
    """
    centers = init_centers(dataset,k)
    m, n = dataset.shape
    clusters = np.full(n,np.nan)
    #迭代标志
    flag = True
    while flag:
        flag = False
        #计算所有点到簇中心的距离
        for i in range(m):
            minDist = np.inf
            clustersIndex = 0
            for j in range(k):
                dist = distance(dataset[i],centers[j])
                if dist < minDist:
                    minDist = dist
                    clustersIndex = j
            # 只有当值和clustersIndex不相等时才更新clusters中的值
            if clusters[i] != clustersIndex:
                clusters[i]=clustersIndex
                flag = True
        #更新簇中心
        for i in range(k):
            subDataset = dataset[np.where(clusters == i)]
            centers[i] = np.mean(subDataset,axis=1)
    return clusters

#画图
def plot(dataset,clusters,centers):
    m, n = dataset.shape
    if m>2:
        print('维度大于2')
    colors = ['r','y','g','b','m']
    for i in range(m):
        clustersIndex = clusters[i].astype(np.int)
        plt.scatter(dataset[i][0],dataset[i][1],color=colors[clustersIndex],marker='o')
    for i in range(len(centers)):
        plt.scatter(centers[i][0],centers[i][1],color=colors[i],marker='v')
    plt.show()