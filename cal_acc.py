# from KDtree_index.build_tree  import build_tree
from config import opt
from get_bin import read_bin
# from find_tmp_knn import find_knn
import h5py as h5
import numpy as np
import time
import datas_load as load
from getindex import getindex

def overall_radio(knn_dis, knn_dis_): #真正用于计算的 knn_dis_ 就是knn_dis*(*无法作为命名符号)
    # print(knn_dis_)
    # print(knn_dis)
    # result_len  表示找到的knn的个数，可能找到不足 我们设定每次比较找到的knn 与  和找到的knn一样长度的real_knn作比较
    result_num = len(knn_dis_)
    # print("we_find_knnnum", result_num)

    # 如果找不到k个值，就往后面补零
    if result_num!=opt.knn_num :
        knn_dis_ = np.pad(knn_dis_, (0,opt.knn_num-len(knn_dis_)), 'constant', constant_values=(0,0))
        # print(knn_dis_)

    if opt.knn_num>1:
        knn_dis_ = np.array(knn_dis_)[1:]#预防除零异常

        knn_dis = np.array(knn_dis)[1:opt.knn_num]
    else :
        knn_dis_ = np.array(knn_dis_)[0]
        knn_dis = np.array(knn_dis)[0]
    # tmp = 0
    # print(knn_dis_)
    # print(knn_dis)
    if opt.knn_num == 1 :
        if knn_dis_ -0.0 > 1e-5: # knn = 1 的时候 找的结果不正确
            return 1.0001# 不然的话 knn_dis 第一个值为0  如果正常计算就会有分母为0  的时候
        else :
            return  1.0
    else :
        if knn_dis_[0] -0.0 > 1e-5:
            return (sum(knn_dis_ / knn_dis) + 1.00001) / opt.knn_num
        else:
            return (sum(knn_dis_ / knn_dis) + 1) / opt.knn_num


def cal_overall_radio(file_name,knn,dis,knn1,dis1):
    #  index 是测试集下标
    # x是测试集下标对应的数据
    save_name = ""
    dir_list = file_name.split("/")
    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"


    time1 = time.time()
    # knn1, dis1 = find_knn(file_name,dataMatrix,output, opt.sigma, x, bin,id)
    # time_fin_KNN = time.time()
    # print(knn1,dis1)
    # print("\nall_find-knn",time_fin_KNN-time1)


    overall = overall_radio(dis, dis1)
    # time_overall = time.time()
    # print("time_overall",time_overall-time_fin_KNN)


    recall,flag,result_num = recall_(knn, knn1)
    # time_recall = time.time()
    # print("time_recall",time_recall-time_overall)

    # print("Acc_time",time.time()-time1)
    return overall,recall,flag,result_num

def recall_(knn, knn_):
    # print(knn[:opt.knn_num])
    # print(knn_)
    result_num = len(knn_)
    flag = 0
    # flag 记录是否找到了k个点，如果不是则返回1
    if result_num!=opt.knn_num :
        knn_ = np.pad(knn_,(0, opt.knn_num-len(knn_)), 'constant', constant_values=(0,0))
        print(knn_)
        flag = 1
    return len(set(knn[:opt.knn_num]) & set(knn_)) / opt.knn_num ,flag,result_num


# def cal_recall(output,file_name):
#     tree, id, num_leaf = build_tree(opt.data_file)
#     bin = read_bin(file_name)
#     f = h5.File(opt.data_file)['testset'][:].tolist()
#
#     for x in f:
#         tim1 = time.time()
#         knn, dis = cal_knn_dis(tree, x)
#         knn1, dis1 = find_knn(output, opt.sigma, x, bin)
#         wucha = recall(knn, knn1)
#         print('recall:', wucha)
#         print('用时:', time.time() - tim1)


if __name__ == '__main__':

    # file_name = "Datas/audio/audioKNN.hdf5"
    # f = h5.File(file_name)['testset']

    output = []
    for i in range(4):
        output.append(1)
    adc = []
    for i in range(50):
        adc.append(1)

    # print(cal_overall_radio(output,file_name,0,f[0]))
    overall_radio(adc,output)
    recall_(adc,output)