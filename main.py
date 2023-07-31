import time
import train
import result
import datas_load as load
from config import opt
import DSCNN
import Reassign_Rand
import Reassign_Gre
import compare_model.CaiModel as CModel
import DSCNN_2
# import different_layer_model.DSCNN_3 as DSCNN3
# import noFcWeightedloss as nofc
# import attenxDSCNN  as atten
# import different_layer_model.DSCNN_4 as DSCNN4
# import different_layer_model.DSCNN_5 as DSCNN5
# import different_layer_model.DSCNN_6 as DSCNN_6
import alpha_same_layer_diff_wider.DSCNN_group2 as group2
import alpha_same_layer_diff_wider.DSCNN_group3 as group3
import alpha_same_layer_diff_wider.DSCNN_group4 as group4
import alpha_same_layer_diff_wider.DSCNN_group5 as group5
import different_lalphasame_layer_diff_deeper.DSCNN_4 as asameDSCNN3
import different_lalphasame_layer_diff_deeper.DSCNN_6 as asameDSCNN4
import different_lalphasame_layer_diff_deeper.DSCNN_8 as asameDSCNN5
import different_lalphasame_layer_diff_deeper.DSCNN_10 as asameDSCNN6
import Reassign_Rand_top_gamma_assignReal_KNN as rand_gamma
import computa_rhoInProve as proRho
# import different_a_mult_b.mult1_3 as mult1_3
# import different_a_mult_b.mult3_1 as mult3_1
# import DSCNN_DSCNN.dscnn_dscnn as dscnn_dscnn
# import DSCNN_DSCNN.cnn_cnn as cnn_cnn

import Reassign_Rand_WeightnoSort as Rand_WeightNoSort
ModleList = [DSCNN_2.DSCNN_model,group2.DSCNN_model,group3.DSCNN_model,group4.DSCNN_model,group5.DSCNN_model]
             # asameDSCNN3.DSCNN_model,asameDSCNN4.DSCNN_model, asameDSCNN5.DSCNN_model,asameDSCNN6.DSCNN_model]
# DSCNN.DSCNN_model,CModel.CaiModel,DSCNN3.DSCNN_model,DSCNN4.DSCNN_model,DSCNN5.DSCNN_model,
#              mult1_3.DSCNN_model,mult3_1.DSCNN_model,dscnn_dscnn.DSCNN_model,cnn_cnn.DSCNN_model,DSCNN_6.DSCNN_model]

Model_name = ["DSCNN_2model","group2","group3","group4","group5"]
              # "asameDSCNN3","asameDSCNN4","asameDSCNN5","asameDSCNN6","DSCNN_model","Cai_model","DSCNN3","DSCNN4","DSCNN5","1mult3","3mult1","dscnn_dscnn","cnn_cnn","DSCNN_6"]
              # "fc_model","fc_model_1","DSCNN_change_mlti_","DSCNN_double","m1m1m3","m2m2m3","m2m1"]
dataset_name_list = [ "audio","sun",  "enron", "nuswide", "notre","sift"]
datanum = [53387, 79106, 94987, 268643, 332668, 994461]
# dir_path is  dataset_path
# dir_path = "/root/autodl-tmp/hanao/model/tmp/pycharm_project_705/PM-tree_model+bin/dataset/"
dir_path = "./dataset/"
# index_name = "LSH-PM"
index_name = "M"
file_last_name = "_50NNlabel.hdf5"
dataset_name = "datasetKnn.hdf5"
a = [10]
# a=[8,9]# 再分配 和 最终遍历的叶子节点个数
knn = [50]# 找的近邻的个数

for aa in range(1):
    for i in range(3):
        # if knn[aa]<30 :
        #    opt.topk = 1
        #    opt.knn_num = knn[aa]
        #    opt.reassign_bin_num = 1
        # else:
        #    opt.topk = 2
        #    opt.reassign_bin_num = 2
        #    opt.knn_num = knn[aa]

        opt.knn_num = 50
        opt.topk = 2
        opt.reassign_bin_num = 2
        opt.ratio = ratio[aa]
        num =0
        model = ModleList[num]
        print(model)
        label_filename = dir_path + dataset_name_list[i] + '/' + index_name+'/'+dataset_name_list[i] + index_name+file_last_name
        if dataset_name_list[i] == "sift":
            maxclusterSize=210
            # opt.topk = 2
            # opt.reassign_bin_num = 2
        else:
            maxclusterSize=70

        file_name = dir_path + dataset_name_list[i] + '/' + dataset_name
        print("\nfile_name = ",file_name,"\nlabel_name = ",label_filename)
        dataMatrix, trainDataset, trainKnn = load.LoadDataset(file_name)
        trainlabel = load.Label_LoadDataset(label_filename)
        # print("train", file_name,model)
        # train.trainModel(model, file_name,trainDataset, trainlabel, Model_name[num],label_filename,index_name,0.8)
        time1 = time.time()
        rand_gamma.Reassign(model, file_name, index_name, Model_name[num], 20, MaxclusterSize=maxclusterSize)
        # print("======================reassign need ",(time.time()-time1)/60,"Min ================")
        print("============================knn_num =", opt.knn_num, model,"==topk =",opt.topk,model)
        print("============================reassign_bin_num =", opt.reassign_bin_num, model)
        print("test", file_name)
        # filepathpro = "C:/hanaocode/PM-model-part111111/PM-tree_model+bin/dataset/"+dataset_name_list[i]+"/M/"
        # rhoinProve = proRho.computeRho(filepathpro,datanum[i])
        # print("rhoinProve=",rhoinProve)
        result.testModel(model, file_name,index_name, Model_name[num],"Rand_gamma",0.8)
        # opt.topk = 50
        # result.testModel(model, file_name, index_name, Model_name[num], "bin", 0.8)
