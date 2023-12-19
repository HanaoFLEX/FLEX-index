
import torch
import random
import time
from torch.utils.data import Dataset, DataLoader
from config import opt
import datas_load as load
from getindex import getindex,get_variable_index
from get_bin import read_bin,getDataBin
import numpy as np
from find_tmp_knn import find_candid_point,Knn,Compute_Euclidean
import csv


torch.set_printoptions(profile="full")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')
print(device)

def sort_result(result):

    sort_result = sorted(enumerate(result), key=lambda xx: xx[1], reverse=True)  # x[1]是因为在enumerate(a)中，a数值在第1位
    sort_result_id = [yy[0] for yy in sort_result]

    return sort_result_id

def find_knn(result,bin_num,dataMatrix,queryPoint,bin,DataBinId):


    candid = []
    _, predicted = torch.topk(result.data, k=opt.knn_num, dim=0)
    for pr in predicted:
        if pr < bin_num:
            candid.extend(bin[int(pr)])

    model_knn, model_dis = Knn(opt.knn_num, candid, dataMatrix, queryPoint)

    knn_bin = []
    for dataId in model_knn:
        knn_bin.append(DataBinId[dataId])

    return model_knn,model_dis,knn_bin

# 获取
def get_goalNode(file_name,Model_name,index_name,model,dataset):# 只返回前gamma个最大的块
    save_name = ""
    dir_list = file_name.split("/")

    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"
    save_name +=index_name +'/'
    save_file_name = save_name + Model_name + '/'

    # 记录每个数据点 涉及到的边
    goal_leafnode = dict()

    # get label_size
    f = open(save_name+index_name + 'bin.txt', 'r')
    bin_num = int(f.readline())
    # LeafNodeitem 记录每个叶子节点涉及到的边 LeafNodeitem[leafid] = [[dataid_1,weight],[dataid_2,weight].....]
    outlabel_len = bin_num
    f.close()

    input_dim = len(dataset[0])
    output_dim = outlabel_len

    # .load加载预训练参数 .load_state_dict将预训练参数加载到模型中
    net = model(input_dim, output_dim).to(device)
    net.load_state_dict(torch.load(save_file_name+index_name +"_net_latest.pth"))
    net.eval()
    print("加载参数完毕")
    # 记录后选块
    dataset = torch.Tensor(dataset).to(device)
    trainloader = DataLoader(dataset=dataset, batch_size=50, shuffle=False)
    j = 0
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):

            inputs = data.to(device)
            outputs = net(inputs)
            for index in range(len(inputs)):
                x = index + j * trainloader.batch_size
                result = outputs[index]
                _, predicted = torch.topk(result.data, k=50, dim=0)  # predicted  <class 'torch.Tensor'>

                predicted = predicted.numpy().tolist()  # predicted <class 'list'>
                predicted = list(np.ravel(predicted))  # 将[[1，2，3]]  变为[1,2,3]
                goal_leafnode[x] =  predicted[:opt.reassign_bin_num] # 记录数据集每个点被模型预测的topk 个位置
            j +=1
    return goal_leafnode


def ItemsOptimization(model,file_name,Model_name,index_name,dataMatrix,trainKnn, trainset,bin,DataBinId,getindex):

    graph = dict()

    goal_leafnode =get_goalNode(file_name,Model_name,index_name,model,trainset)# 返回测试集每个点前gamma个块


    dataset_len = dataMatrix.shape[0]

    with open(str(opt.reassign_bin_num) + "graph_init.txt", "w") as graph_init:
        graph_init.write("pointID bin_ID value\n")
        for i in range(dataset_len):
            bin_id = DataBinId[i]
            graph[i,bin_id] = 1
            graph_init.write(str(i)+"\t"+str(bin_id)+"\t"+"1\n")
        graph_init.close()

    train_len = len(trainset)

    for i in range(train_len):
        leafNodes = goal_leafnode[i]
        real = trainKnn[i][0]
        real = real[:opt.knn_num]
        for lNode in leafNodes:
            for pointIndex in real:
                pointIndex = int(pointIndex)
                if (pointIndex,lNode) not in graph.keys():
                    graph[pointIndex,lNode] = 2
                else:
                    graph[pointIndex, lNode] =  graph[pointIndex,lNode] +2

    return goal_leafnode,graph,dataset_len

def get_point_weigh_matrix(graph,dataset_len):
    index_dictionary = dict()
    point_sumgamma_weight = dict()
    point_weight_id = dict()
    sort_id = []
    max_weight = 0.0
    for i in range(dataset_len):
        index_dictionary[i] = []
        point_sumgamma_weight[i] = 0

    for keys in graph.keys():
        key = list(keys)
        index_dictionary[key[0]].append([key[1],graph[keys]])

    # 对每个点 到各个叶子节点的权值进行排序  形成最终的权重矩阵
    sum_degree = 0.0
    max_degree = 0
    for key in index_dictionary.keys():
        index_dictionary[key] = sorted(index_dictionary[key], key=lambda k: k[1], reverse=True)  # 与该数据点相连的 叶子节点 的边进行排序
        degree =  len(index_dictionary[key])
        if degree > max_degree:
            max_degree = degree
        sum_degree += degree
    avg_degree = sum_degree/(len(index_dictionary)*1.0)

    for key in index_dictionary.keys():
        i = 0
        for item in index_dictionary[key]:
            if i == opt.reassign_bin_num:
                break
            point_sumgamma_weight[key] += item[1]  # 将point 的总权值累加
            i = i + 1
        if point_sumgamma_weight[key] > max_weight:
            max_weight = point_sumgamma_weight[key]

    print("max_weight", max_weight)
    for num in range(max_weight + 1):
        point_weight_id[num] = []

    for key in index_dictionary.keys():
        point_weight_id[point_sumgamma_weight[key]].append(key)  # point_weight_id[quanzhong] = [pointid1, id2...]

    for num in range(max_weight):
        a = max_weight - num
        if a in point_weight_id:
            for pointid in point_weight_id[a]:
                sort_id.append(pointid)

    return index_dictionary,sort_id,point_sumgamma_weight,avg_degree,max_degree
# 得到 整个训练集 过模型后 模型预测概率从高到低的节点顺序 以及 预测knn-> model_KNN
def Reassign(model, file_name,index_name,Model_name,find_num,MaxclusterSize=70,):
    save_name = ""
    dir_list = file_name.split("/")
    sumweight = 0.0
    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"
    dataname = dir_list[i]
    # save_file_name = save_name +Model_name+ '/'
    save_name +=index_name+"/"
    bin_path = save_name+index_name+"bin.txt"
    bin = read_bin(bin_path)
    bin_num = len(bin)
    get_index = getindex(file_name)
    DataBinId = getDataBin(file_name,index_name)
    dataMatrix, trainDataset, trainKnn = load.LoadDataset(file_name)

    # 需要优化的查询点
    print(trainDataset.shape)
    # 计算各边的权值
    goal_leafnode,graph,dataset_len,= ItemsOptimization(model, file_name,Model_name,index_name,dataMatrix,trainKnn, trainDataset, bin, DataBinId, get_index )

    # 构建最终的矩阵 每一行是涉及Knn的一个点 与所有叶子节点的关联信息， 第一行的总权值最大，然后权值依次下降    point_weigh_matrix = [] # 最终按权重大小构建的 矩阵
    pointWithNoBin = []
    full_bin = []
    for i in range(bin_num):
        bin[i] = []
    point_weigh_matrix,sort_id,point_all_weight,avg_degree,max_degree = get_point_weigh_matrix(graph,dataset_len)



    # 重新分配涉及到knn的点
    for key in sort_id:
        flag = False
        for xx in range(find_num):
            p = random.random()
            p = p * point_all_weight[key]
            sum1 = 0.0
            sum2 = 0.0
            for item in point_weigh_matrix[key]:
                sum2 +=item[1]
                if sum1<p<=sum2 :
                    if (len(bin[item[0]]) < MaxclusterSize):
                        bin[item[0]].append(key)
                        sumweight = sumweight + item[1]
                        flag = True
                    elif item[0] not in full_bin:
                        full_bin.append(item[0])
                    break
                sum1 += item[1]
            if flag == True:
                break
        if flag == False:
            pointWithNoBin.append(key)

    first_full_bin = len(full_bin)
    first_pointWithNoBin = len(pointWithNoBin)
    print("涉及Knn中已经满的叶子结点个数为： ", len(full_bin))
    print("涉及knn的点 冲突点的个数为: ", len(pointWithNoBin))

    point_num = len(pointWithNoBin)
    if point_num != 0:
        not_full_bin = []
        for bin_id in range(bin_num): # 涉及重分配的叶子结点
            if bin_id not in full_bin:
                not_full_bin.append(bin_id)
        for dataid in pointWithNoBin:
            for bin_id in not_full_bin:
                if len(bin[bin_id]) < MaxclusterSize:
                    bin[bin_id].append(dataid)
                    break
                elif bin_id not in full_bin:
                    full_bin.append(bin_id)
                    del not_full_bin[not_full_bin.index(bin_id)]

    print("save txt")
    # with open(save_name + index_name + "k="+str(opt.knn_num)+"topk="+str(opt.reassign_bin_num)+ "reassignL3p+ "+"point_weigh_matrix.txt", "w") as matrix:
    #     for key in sort_id:
    #         matrix.write(str(key))
    #         for item in point_weigh_matrix[key]:
    #             matrix.write(str(item))
    #         matrix.write("\n")
    #     matrix.close()

    with open(save_name +index_name+"k="+str(opt.knn_num)+"topk="+str(opt.reassign_bin_num)+ "rand_gamma_bin.txt", "w") as ha:
        data_num =0
        ha.write(str(bin_num)+'\n')
        full = 0
        for bin_id in range (bin_num):
            x = len(bin[bin_id])
            if x==MaxclusterSize:
                full+=1
            for dataid in bin[bin_id]:
                ha.write(str(dataid)+" ")
            data_num += x
            ha.write('\n')
        ha.close()

    print("max_degree:",max_degree)
    print("avg_degree:", avg_degree)
    file_Rand = open("LFNN_reassign_Rand_result" + index_name + ".csv", 'a+')
    file_result = csv.writer(file_Rand)
    file_result.writerow(
        [dataname, "LFNN", index_name, "RandGamma",
         "topk: " + str(opt.topk) + " reassign_bin_num: " + str(opt.reassign_bin_num) ,
         str(opt.knn_num),
         str(trainDataset.shape[0]),
         str(len(point_weigh_matrix)),#knn_point_num
         str(first_pointWithNoBin),#number of conflict points
         str(first_pointWithNoBin * 1.0 / len(point_weigh_matrix)),#conflict ratio
         str(avg_degree)+"avg_degree",#ave_edge is
         str(max_degree)+"max_degree",
         str(sumweight)+"sum_weight",
         ])
    file_Rand.close()


if __name__ == '__main__':
    ModleList = [DSCNN.DSCNN_model]
    Model_name = ["DSCNN_model"]
    find_num = 20# find num 产生随机数的次数，如果find_num 次后叶子节点还是满的那么就认为 这个点没有进行分配，后续在进行分配
    file_name = "D:/hanaao-code/PM-model-part111111/PM-tree_model+bin/dataset/audio/datasetKnn.hdf5"
    Reassign(ModleList[0], file_name, Model_name[0], find_num,MaxclusterSize=70)
    print("is ok!!!!!")
    # bin_name = "D:/hanaao-code/PM-model-part111111/PM-tree_model+bin/dataset/audio/repartion_bin.txt"

    # bin = read_bin(bin_name)
    # data_num = 0
    # for i in range(len(bin)):
    #     x = len(bin[i])
    #     print('i = ',i,'x = ',x)
    #     data_num +=x
    # print("data_num = ",data_num)


