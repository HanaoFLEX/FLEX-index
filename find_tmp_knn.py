from get_bin import read_bin
import datas_load as load
import numpy as np
from config import opt
import h5py as h5
from getindex import getindex
import time

# def dis(a, b):
#     a = np.array(a)
#     b = np.array(b)
#     return sum(((a - b)**2))**(1/2)
#     # return sum(((a - b)**2))
def Compute_Euclidean(v1, v2):
    """计算两点之间的欧式距离

    :param v1: 欧式空间中点1
    :param v2: 欧式空间中点2
    :return: 点1和点2之间的距离
    """
    sum_distance = 0
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    sum_distance += np.power(v1 - v2, 2).sum()
    distance = np.power(sum_distance, 0.5)

    return distance


def find_candid_point(bin,output):
    candid = []
    candid_index = np.argwhere(output.detach().cpu().numpy()>=opt.sigma)
    # print("candid_index",candid_index)
    for x in candid_index:
        candid.extend(bin[int(x)])
        # print(candid)
    return candid ,len(candid_index)


def Knn(k,pointset_index, matrix,point):
    """ k近邻搜索方法

    :param k: 近邻数量
    :param point: 中心点
    :param pointset_index: 点集合的索引号
    :param matrix: 各个点的坐标
    :return: （返回中心点的k近邻点在matrix中的索引号，中心点到各个点的距离）
    """
    time1 = time.time()
    pointset_index = np.asarray(list(set(pointset_index)))
    v1 = np.asarray(point)
    # max_distance = -float("inf")
    K_list = []
    K_distance = []

    for i in pointset_index:
        v2 = np.asarray(matrix[i])
        distance = Compute_Euclidean(v1, v2)

        if len(K_list) < k:
            K_list.append(i)
            K_distance.append(distance)
            continue

        max_distance = max(K_distance)
        if distance < max_distance:
            index = K_distance.index(max_distance)
            K_distance[index] = distance
            K_list[index] = i

    if len(K_distance) < k:
        k = len(K_distance)
    # 对列表排序，主要是为了其他方法实现时的方便
    K_distance_original = K_distance.copy()
    K_distance.sort()
    K_list_original = K_list.copy()
    for i in range(k):
        index = K_distance_original.index(K_distance[i])
        K_list[i] = K_list_original[index]
        K_distance_original[index] = -1
    K_distance.sort()

    # print("knn_time",time.time()-time1)
    return K_list, K_distance


#
# def find_knn(candid,dataMatrix,goal_point,id):
#
#     time1 = time.time()
#     # print("在find_tmp文件 现在查的点是",goal_point
#     M = MaxHeap(goal_point)
#     # i = 0
#     for x in candid:
#         # x存的是数据下标 将下标转化为具体数据
#         # print(candid[i])
#         # i +=1
#         # print(x)
#         x= tuple(dataMatrix[x])
#         # print(x, "对应数据嘛")
#         if M.get_size() < opt.knn_num :
#             M.push(x)
#         elif dis(goal_point, x) < dis(goal_point, M.find_max()):
#             M.pop()
#             M.push(x)
#
#     knn = []
#     distance = []
#     time_findknn = time.time()
#     print("knn",time_findknn-time1)
#     while M.get_size() > 0:
#         x = M.pop()
#         knn.append(x)
#         distance.append(dis(goal_point, x))
#
#     time_while = time.time()
#     print("while_",time_while-time_findknn)
#
#     # time = 0.0
#     knn = knn[::-1]
#     distance = distance[::-1]
#     knn2 = [id[x] for x in knn]
#
#     return knn2, distance


if __name__ == '__main__':

    file_name = "Datas/audio/audioKNN.hdf5"
    id2 = getindex(file_name)

   #得到各个bin里面存的点
    bin = read_bin(file_name)

    # 自己写一个output[]测试
    output = []
    #hh 给叶子节点标号？？？
    for i in range(4):
        output.append(1)
    for i in range(1020):
        output.append(0)
    # output[]=[1, 1, 1, 1, 0, 0, 0, .... 0, 0, 0] len(output) = 1024
    # print(output)

    # 根据output 查第一个叶子节点的第一个数据的 knn
    dataMartix = h5.File(file_name)['dataMatrix']
    f = dataMartix[bin[0][0]]
    knn, dis = find_knn(output, opt.sigma, f, bin, id2)
    print(knn)
    print(dis)
