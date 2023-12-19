# from KDtree_index import build_tree
import time
from getindex import getindex
from config import opt
import h5py as h5
from numpy import float64
import  datas_load as load

def read_bin(bin_path):#我们之前已经写好了文件 直接读入返回bin[]即可


    f = open(bin_path, 'r')
    num_leaf = int(f.readline())
    # print(num_leaf)
    bin = []

    for i in range(num_leaf):
        bin.append([])

    for i in range(num_leaf):
        # num = int(f.readline())
        # print(num)
        x = f.readline()
        num = len(x.split())
        # print(num)
        # 把字符串一个一个分隔开，但是此时数组里还是存的字符串，我们要把它转换成数字（数据点下标）
        bin[i] = x.split(' ',num - 1)
        bin[i] = [int(x) for x in bin[i]]
        # print(bin[i])

    return bin

def getDataBin(file_name,index_name):#我们之前已经写好了文件 直接读入返回bin[]即可

    dataMatrix, trainDataset, trainKNN = load.LoadDataset(file_name)

    save_name = ""
    dir_list = file_name.split("/")
    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"
    save_name +=index_name+'/'
    f = open(save_name+index_name+'bin.txt', 'r')
    num_leaf = int(f.readline())
    # print(num_leaf)


    getDataBin = [-1 for i in range(0, dataMatrix.shape[0])] # 每个数据点id所在叶子节点下标


    for i in range(num_leaf):
        # num = int(f.readline())
        # print(num)
        x = f.readline()
        num = len(x.split())

        linedata = []
        for xx in range (num):
            linedata.append(0)

        # 把字符串一个一个分隔开，但是此时数组里还是存的字符串，我们要把它转换成数字（数据点下标）
        linedata = x.split(' ',num - 1)
        for x in linedata:
            getDataBin[int(x)] = i

        # print(bin[i])

    return getDataBin


if __name__ == "__main__":

   file_name = "D:/study/every code/PM-hanao/Final_PM_model/dataset/audio/datasetKnn.hdf5"
   bin = read_bin(file_name)
   print(type(bin))
   print(len(bin))
