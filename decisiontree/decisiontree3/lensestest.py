import treePlotter
import trees


def main():
    #打开文件
    fr = open('lenses.txt')
    #读取文件信息,得到一个dataSet,是一个二维列表
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    #定义标签
    lensesLabels= ['age', 'prescript', 'astigmatic', 'tearRate']
    #创建树
    lensesTree= trees.createTree(lenses, lensesLabels)
    print(lensesTree)

    #画图
    treePlotter.createPlot(lensesTree)

#如果是主函数而不是被其他.py文件调用的话，则循行if内的代码
if __name__ =='__main__':
    main()