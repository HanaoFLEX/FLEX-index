import time
import train
import result
import datas_load as load
from config import opt

import DSCNN_2
import RandAssign as rand_gamma

ModleList = [DSCNN_2.DSCNN_model]

Model_name = ["DSCNN_2model"]

dataset_name_list = [ "audio","sun",  "enron", "nuswide", "notre","sift"]
# datanum = [53387, 79106, 94987, 268643, 332668, 994461]

dir_path = "D:/hanaao-code/model_dataset/"

index_name = "M"
file_last_name = "_50NNlabel.hdf5"
dataset_name = "datasetKnn.hdf5"

knn = [50]# 找的近邻的个数

for i in range(2):
    for aa in range(len(knn)):
        if knn[aa]<30 :
           opt.topk = 1
           opt.knn_num = knn[aa]
           opt.reassign_bin_num = 1
        else:
           opt.topk = 2
           opt.reassign_bin_num = 2
           opt.knn_num = knn[aa]

        num =0
        model = ModleList[num]
        label_filename = dir_path + dataset_name_list[i] + '/' + index_name+'/'+dataset_name_list[i] +index_name+file_last_name
        if dataset_name_list[i] == "sift":
            maxclusterSize=210
        else:
            maxclusterSize=70

        file_name = dir_path + dataset_name_list[i] +"/"+dataset_name
        print("\nfile_name = ",file_name,"\nlabel_name = ",label_filename)

        dataMatrix, trainDataset, trainKnn = load.LoadDataset(file_name)
        #
        trainlabel = load.Label_LoadDataset(label_filename)
        print("train", file_name, model)
        train.trainModel(model, file_name,trainDataset, trainlabel, Model_name[num],label_filename,index_name,0.8,dataset_name_list[i])

        print("rand_gamma  gamma = ", opt.topk)
        rand_gamma.Reassign(model, file_name, index_name, Model_name[num], 20, MaxclusterSize=maxclusterSize)

        print("============================knn_num =", opt.knn_num, model,"==topk =",opt.topk,model)
        # print("============================reassign_bin_num =", opt.reassign_bin_num, model)
        print("test", file_name)
        result.testModel(model, file_name,index_name, Model_name[num],"Rand_gamma",0.8)

        # opt.topk = 50
        # result.testModel(model, file_name, index_name, Model_name[num], "bin", 0.8)
