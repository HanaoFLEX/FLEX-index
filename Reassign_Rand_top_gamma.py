
import torch
import random
import time
from torch.utils.data import Dataset, DataLoader
from config import opt
import datas_load as load
from getindex import getindex,get_variable_index
from get_bin import read_bin,getDataBin
import numpy as np
import DSCNN
from find_tmp_knn import find_candid_point,Knn,Compute_Euclidean
import csv


torch.set_printoptions(profile="full")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')
print(device)

def sort_result(result):
    '''
    :param result: model_result
    :return:  sort_result_bin : get Original id of sorted result
    '''
    sort_result = sorted(enumerate(result), key=lambda xx: xx[1], reverse=True)  # x[1]是因为在enumerate(a)中，a数值在第1位
    sort_result_id = [yy[0] for yy in sort_result]

    # print("len(sort_result_id)\n", sort_result_id)
    return sort_result_id

def find_knn(result,bin_num,dataMatrix,queryPoint,bin,DataBinId):
    '''

    :param result: model(queryPoint) result
    :param bin_num: leaf_node_number
    :param dataMatrix: dataset
    :param queryPoint:
    :return: model_knn :model predict knn[]
            knn_bin: get model_knn  binId

    '''

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
def get_goalNode(file_name,Model_name,index_name,model,dataset):
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
    # 先存储 数据集中所有点knn被模型预测 的叶子节点Id

    # 计算数据集中每个点经过模型会落在 各个叶子结点的概率 并返回topk 叶子节点id
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
                goal_leafnode[x] =  predicted # 记录数据集每个点被模型预测的topk 个位置
            j +=1
    return goal_leafnode


def ItemsOptimization(model, file_name,Model_name,index_name,dataMatrix,trainKnn, optimizeMatrix,bin,DataBinId,getindex):
    '''

    :param trainKnn: 训练集的knn
    :param LeafNodeitem:
    :param optimizeMatrix: 训练集
    :param goal_leafnode: model 预测的 概率从大到小的叶子节点id
    :param bin:
    :param DataBinId:
    :param getindex:
    :param k:
    :param MaxclusterSize:
    :param topk: 将knn 在分到的叶子节点个数
    :return:
    '''

    LeafNodeitem = dict()
    for i in range(len(bin)):
        LeafNodeitem[i] = []  # 每个叶子节点id 作为他的key值
    pointInKnn = []# 判断当前点是不是之前出现过，如果没有出现过则 建立一条该点 和本身所在叶子结点的边
    pointNotInKnn = []
    sample_dis_avg_bound=get_sample_predKnn(file_name,Model_name,index_name,model,dataMatrix)
    opt.Avg_Dis = sample_dis_avg_bound
        # np.mean(sample_dis_avg)-np.sqrt(np.var(sample_dis_avg)) * opt.ratio
    # print("opt.avg_dis = ",opt.Avg_Dis)

    goal_leafnode =get_goalNode(file_name,Model_name,index_name,model,dataMatrix)
    knn_pred, dis_avg = get_predictedKnn(file_name,Model_name,index_name,model,dataMatrix,optimizeMatrix)
    print("len(optimizeMatrix)= ",len(optimizeMatrix))
    compute_real = 0
    for i in range(len(optimizeMatrix)):
        point = optimizeMatrix[i]
        point_DatasetId = getindex[tuple(point)]
        leafNodes = goal_leafnode[point_DatasetId][:opt.reassign_bin_num] # 找到我想要 将50NN 放到哪两个叶子节点   opt.reassign_bin_num =2
        if dis_avg[i]<=opt.Avg_Dis:
            real = knn_pred[i]
        else:
            compute_real += 1
            real = trainKnn[i][0]  # 获取当前点的 real-knn
        # print("real",real)
        real = real[:opt.knn_num]
            # 把所有参加 Knn 的元素放入到涉及的叶子的Item中
        for lNode in leafNodes: # hanao  leafnodes 离point最近的叶子节点  我的话得是概率最大的叶子节点
                # 猜测 如果     real-knn中的点如果有与目标叶子节点的边就对应权重加一，不然就创建一条边（real-point 到 目标叶子节点）
            for pointIndex in real: # 将 real knn 和 目标叶子节点建立边
                pointIndex = int(pointIndex)
                if pointIndex not in pointInKnn:# 如果这个点是新点 则先创建一个该点和其原来所在叶子节点得边
                    bin_id = DataBinId[pointIndex]
                    pointInKnn.append(pointIndex)
                    LeafNodeitem[bin_id].append([pointIndex,1])
                flag = True
                for item in LeafNodeitem[lNode]:  #items ：# beam search得到的备份的元素[元素ID，在Knn中出现的次数]

                    if pointIndex == item[0]:  #  #如果 real-knn 的这个点与这个叶子节点中已经有了边
                        item[1] += 2
                        flag = False
                        break
                    # 如果点和叶子节点以前没有边，则加入一条边
                if flag:
                    LeafNodeitem[lNode].append([pointIndex, 2]) # 这里不是1 是为了假如 一个数据点只是一个数据的knn, 那么如果这里权重为一，他就可能被分到原来的叶子节点里，而不是分到knn想进入的那个叶子节点
    print("涉及到的KNN点个数： ",len(pointInKnn))
    leafWithItems = dict() # 把不涉及到的叶子节点去掉 然后leafWithItems只存涉及到Knn的叶子节点
    for i in range(len(bin)):
        if len(LeafNodeitem[i]) != 0:
            leafWithItems[i] = LeafNodeitem[i]

    print("*****与knn相关的叶子节点个数len(leafWithItems", len(leafWithItems))
    print("compute real KNN number is ", compute_real)
    for leaf in leafWithItems.keys():  # 右顶点（和Knn点相关的叶子节点集合）
        for pointIndex in bin[leaf]:
            pointIndex = int(pointIndex)
            if pointIndex not in pointInKnn:
                pointNotInKnn.append(pointIndex)
        bin[leaf] = []  # 清空涉及Knn点的叶子节点
    return  leafWithItems,pointInKnn,pointNotInKnn,goal_leafnode


def get_predictedKnn(file_name, Model_name, index_name, model, dataset,trainDataset):
    save_name = ""
    dir_list = file_name.split("/")

    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"
    save_name += index_name + '/'
    save_file_name = save_name + Model_name + '/'

    # 记录每个数据点 涉及到的边
    goal_leafnode = dict()

    # get label_size
    bin_path = save_name + index_name + "bin.txt"
    f = open(bin_path, 'r')
    bin_num = int(f.readline())
    outlabel_len = bin_num
    f.close()
    bin = read_bin(bin_path)

    input_dim = len(dataset[0])
    output_dim = outlabel_len

    # .load加载预训练参数 .load_state_dict将预训练参数加载到模型中
    net = model(input_dim, output_dim).to(device)
    net.load_state_dict(torch.load(save_file_name + index_name + "_net_latest.pth"))
    net.eval()
    print("加载参数完毕")
    # 先存储 数据集中所有点knn被模型预测 的叶子节点Id

    # 计算数据集中每个点经过模型会落在 各个叶子结点的概率 并返回topk 叶子节点id
    trainDataset = torch.Tensor(trainDataset)
        # .to(device)
    trainloader = DataLoader(dataset=trainDataset, batch_size=50, shuffle=False)
    j = 0
    dis_avg = []
    knn_pred = []
    for i in range(len(trainDataset)):
        knn_pred.append([])

    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):

            inputs = data.to(device)
            outputs = net(inputs)
            for index in range(len(inputs)):
                x = index + j * trainloader.batch_size
                result = outputs[index]
                _, predicted = torch.topk(result.data, k=50, dim=0)  # predicted  <class 'torch.Tensor'>
                candid = []
                for pr in predicted:
                    if (int(pr) < bin_num):
                        candid.extend(bin[int(pr)])
                knn1, dis1 = Knn(opt.knn_num, candid, dataset, trainDataset[x])
                knn_pred.append(knn1)
                dissum=0
                for dis in dis1:
                    dissum = dissum + dis
                disavg = dissum/(len(dis1))
                dis_avg.append(disavg)
            j += 1

    with open(save_name + index_name + "k=" + str(opt.knn_num) + "topk=" + str(
            opt.reassign_bin_num) + "avg_dis"+str(opt.Avg_Dis)+"rand_gamma", "w") as ha:
        for i in range(len(dis_avg)):
            ha.write(str(dis_avg[i])+" ")
            if i%50==0:
                ha.write('\n')
        ha.write("\nmean="+str(np.mean(dis_avg))+" var= "+str(np.var(dis_avg)))
    print("\n训练集距离均值为：",np.mean(dis_avg),"\n训练集距离方差为：",np.sqrt(np.var(dis_avg)))
    ha.close()

    return knn_pred,dis_avg


def get_sample_predKnn(file_name, Model_name, index_name, model, dataset):
    save_name = ""
    dir_list = file_name.split("/")

    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"
    save_name += index_name + '/'
    save_file_name = save_name + Model_name + '/'
    c = 1000

    num = range(0,dataset.shape[0]-1)
    sample_set= random.sample(num, c)#从数据集采样数据

    sampleDataset = []
    for i in sample_set:
        sampleDataset.append(dataset[i])
    # 记录每个数据点 涉及到的边
    goal_leafnode = dict()

    # get label_size
    bin_path = save_name + index_name + "bin.txt"
    f = open(bin_path, 'r')
    bin_num = int(f.readline())
    outlabel_len = bin_num
    f.close()
    bin = read_bin(bin_path)

    input_dim = len(dataset[0])
    output_dim = outlabel_len

    # .load加载预训练参数 .load_state_dict将预训练参数加载到模型中
    net = model(input_dim, output_dim).to(device)
    net.load_state_dict(torch.load(save_file_name + index_name + "_net_latest.pth"))
    net.eval()
    print("加载参数完毕")
    # 先存储 数据集中所有点knn被模型预测 的叶子节点Id

    # 计算数据集中每个点经过模型会落在 各个叶子结点的概率 并返回topk 叶子节点id
    sampleDataset = torch.Tensor(sampleDataset)
        # .to(device)
    trainloader = DataLoader(dataset=sampleDataset, batch_size=50, shuffle=False)
    j = 0
    dis_avg = []
    knn_pred = []
    for i in range(len(sampleDataset)):
        knn_pred.append([])
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):

            inputs = data.to(device)
            outputs = net(inputs)
            for index in range(len(inputs)):
                x = index + j * trainloader.batch_size
                result = outputs[index]
                _, predicted = torch.topk(result.data, k=50, dim=0)  # predicted  <class 'torch.Tensor'>
                candid = []
                for pr in predicted:
                    if (int(pr) < bin_num):
                        candid.extend(bin[int(pr)])
                knn1, dis1 = Knn(opt.knn_num, candid, dataset, sampleDataset[x])
                knn_pred.append(knn1)
                dissum = 0
                for dis in dis1:
                    dissum = dissum + dis
                disavg = dissum / (len(dis1))
                dis_avg.append(disavg)
            j += 1

    with open(save_name + index_name + "sample"+str(c)+" k=" + str(opt.knn_num) + "topk=" + str(
            opt.reassign_bin_num) + "avg_dis" + str(opt.Avg_Dis) + "rand_gamma", "w") as ha:
        for i in range(len(dis_avg)):
            ha.write(str(dis_avg[i]) + " ")
            if i % 50 == 0:
                ha.write('\n')
        ha.write("\nmean=" + str(np.mean(dis_avg)) + " var= " + str(np.var(dis_avg)))
        print("\n",c,"样本距离均值为：", np.mean(dis_avg), "\n样本距离方差为：", np.sqrt(np.var(dis_avg)))
        ha.close()
    dis_avg_bound = np.mean(dis_avg) - np.sqrt(np.var(dis_avg)) * opt.ratio
    print("\ndis_avg_bound = ",dis_avg_bound)
    return dis_avg_bound


def get_point_weigh_matrix(leafWithItems,pointInKnn):
    index_dictionary = dict()
    point_all_weight = dict()  # point_all_weight[元素ID] = all_weight
    sort_id = []
    for pointID in pointInKnn:
        point_all_weight[pointID] = 0
        index_dictionary[pointID] = []
    for leaf in leafWithItems.keys():  # 遍历涉及到Knn的叶子节点  遍历二部图右顶点 leaf 是叶子节点ID
        for item in leafWithItems[leaf]:  # 遍历叶子节点中的每一条[元素ID，在Knn中出现的次数]
            index_dictionary[item[0]].append([leaf, item[1]])# index_dictionary[元素ID] = [[叶子ID，weight],[.]]


    # # 写成概率值
    # for key in index_dictionary.keys():
    #     for item in index_dictionary[key]:
    #         item[1] = (item[1] * 1.0) / (point_all_weight[key] * 1.0)

    # 对每个点 到各个叶子节点的权值进行排序  形成最终的权重矩阵
    sum_edge = 0.0
    for key in index_dictionary.keys():
        index_dictionary[key] = sorted(index_dictionary[key], key=lambda k: k[1], reverse=True)  # 与该数据点相连的 叶子节点 的边进行排序
        sum_edge +=len(index_dictionary[key])
    ave_edge = sum_edge/(len(index_dictionary)*1.0)

    for key in index_dictionary.keys():
        i = 0
        for item in index_dictionary[key]:
            if i == opt.reassign_bin_num:
                break
            point_all_weight[key] += item[1]  # 将point 的总权值累加
            i = i + 1

    # 按照权值对 设计KNN的点进行排序
    sort_knn_pointID = sorted(point_all_weight.items(), key=lambda k: k[1],reverse=True) # 字典排完序是[(key1,value1),(key2,value2)]
    for key in sort_knn_pointID:
        sort_id.append(key[0])
    return index_dictionary,sort_id,point_all_weight,ave_edge
# 得到 整个训练集 过模型后 模型预测概率从高到低的节点顺序 以及 预测knn-> model_KNN
def Reassign(model, file_name,index_name,Model_name,find_num,MaxclusterSize=70,):
    '''

    :param model:
    :param file_name: dataset_file
    :param Model_name: 将模型参数存储在 Model_name的文件夹下
    :param portion_size:  trainset block number
    :param topk: 将knn在分配到的叶子节点个数
    ：param find_num : find num 产生随机数的次数，如果find_num 次后叶子节点还是满的那么就认为 这个点没有进行分配，后续在进行分配
    :return: 1. model_binId(size = [traindataset.shape[0],bin_num])
             2. model_knn pred_knn_id
    '''
    save_name = ""
    dir_list = file_name.split("/")

    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"
    dataname = dir_list[i]
    # save_file_name = save_name +Model_name+ '/'
    save_name +=index_name+"/"
    bin_path = save_name+index_name+"bin.txt"
    bin = read_bin(bin_path)
    bin_num = len(bin)
    get_index = getindex(file_name)  # 给整个数据集中的数据点进行标号
    DataBinId = getDataBin(file_name,index_name)  # DataBiniD[data_id] = [leaf_id]
    dataMatrix, trainDataset, trainKnn = load.LoadDataset(file_name)
    dataMatrix, testDataset, testKnn = load.LoadDataset(file_name,test = True)
    # 需要优化的查询点
    optimizeMatrix = trainDataset
    # 计算各边的权值
    time_start = time.time()
    leafWithItems,pointInKnn,pointNotInKnn,goal_leafnode= ItemsOptimization(model, file_name,Model_name,index_name,dataMatrix,trainKnn, optimizeMatrix, bin, DataBinId, get_index )

    # 构建最终的矩阵 每一行是涉及Knn的一个点 与所有叶子节点的关联信息， 第一行的总权值最大，然后权值依次下降    point_weigh_matrix = [] # 最终按权重大小构建的 矩阵
    pointWithNoBin = []
    full_bin = []

    point_weigh_matrix,sort_id,point_all_weight,ave_edge = get_point_weigh_matrix(leafWithItems,pointInKnn)
    print("ave_edge = ",ave_edge)
    # with open(save_name + "+1-" + str(opt.reassign_bin_num) + "point_weigh_matrix.txt", "w") as matrix:
    #     for key in sort_id:
    #         matrix.write("["+str(key)+": ")
    #         for item in point_weigh_matrix[key]:
    #             matrix.write("("+str(item)+"), ")
    #         matrix.write("]\n")
    #     matrix.close()

    # 重新分配涉及到knn的点
    for key in sort_id:
        flag = False
        for xx in range(find_num):
            p = random.random()# 产生一个0-1的随机数
            p = p * point_all_weight[key]
            sum1 = 0.0
            sum2 = 0.0
            for item in point_weigh_matrix[key]: # 计算这个概率落在哪个叶子节点
                sum2 +=item[1]
                if sum1<p<=sum2 :
                    if (len(bin[item[0]]) < MaxclusterSize):  # 找到不满的点就放进去
                        bin[item[0]].append(key)
                        flag = True
                    elif item[0] not in full_bin:
                        full_bin.append(item[0])
                    break # 找到预测的叶子节点了 不管满没满都重新预测
                sum1 += item[1]
            if flag == True:# 已经把这个点放进叶子节点了
                break
        if flag == False: # 根据概率预测的的叶子节点都满了！！！ 我们就把他先存起来  避免现在随机分配这个点 会占用其它点最优叶子节点
            pointWithNoBin.append(key)  # 将当前数据Id存进来

    first_full_bin = len(full_bin)
    first_pointWithNoBin = len(pointWithNoBin)
    print("涉及Knn中已经满的叶子结点个数为： ", len(full_bin))
    print("涉及knn的点 冲突点的个数为: ", len(pointWithNoBin))
    print("每个点产生随机数的次数：",find_num)
    # 重分配 未分配到与自身Knn相关的叶子节点的点  根据模型预测的叶子节点来依次进入

    if len(pointWithNoBin) != 0:
        for dataid in pointWithNoBin:
            for bin_id in goal_leafnode[dataid]:
                if bin_id in leafWithItems.keys(): # 模型预测的叶子节点 同时在我们设计重分配的叶子结点（保证所有我们重分配的点  都放在我们重分配的叶子节点里）
                    if len(bin[bin_id]) < MaxclusterSize:
                        bin[bin_id].append(dataid)
                        index = pointWithNoBin.index(dataid)
                        del pointWithNoBin[index]  # 找到叶子节点就pointwithnobin 中删掉此数据点
                        break
                    elif bin_id not in full_bin:
                        full_bin.append(bin_id)

    print("t通过 goal_leafnode  还没有找到bin的数据点个数： ", len(pointWithNoBin))

    # 如果还没有找到 则挨个放入leafWithItems(涉及Knn的所有叶子节点)
    if len(pointWithNoBin) != 0:
        not_full_bin = []
        for bin_id in leafWithItems.keys(): # 涉及重分配的叶子结点
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


        print("最终所有涉及knn的叶子节点中 满的叶子节点个数为", len(full_bin))
        print("len(pointNotInKnn) = ",len(pointNotInKnn))
        print("len(pointInKnn) = ", len(pointInKnn))
        print("conflict ratio is ",str(first_pointWithNoBin*1.0/len(point_weigh_matrix)),"\n")

    # 重分配 未涉及到knn的点  如果原来其所在叶子节点没满 则放在原来位置  否则按照模型对其进行放置  在放不下 就放在leafWithItems 没满的叶子节点里
    for dataid in pointNotInKnn:
        flag = False
        if len(bin[DataBinId[dataid]]) < MaxclusterSize:
            bin[DataBinId[dataid]].append(dataid)
            flag = True
        if flag == False:
            for bin_id in goal_leafnode[dataid]:
                if bin_id in leafWithItems.keys():
                    if len(bin[bin_id]) < MaxclusterSize:
                        bin[bin_id].append(dataid)
                        flag = True
                        break
            if flag == False:
                for bin_id in leafWithItems.keys():
                    # print("pointNotInKnn 进来 leafWithItems ")
                    if len(bin[bin_id]) < MaxclusterSize:
                        bin[bin_id].append(dataid)
                        flag = True
                        break
    time_end = time.time()
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
            # print("bin_id ", bin_id, "  data_num= ", x)
            for dataid in bin[bin_id]:
                ha.write(str(dataid)+" ")
            data_num += x
            ha.write('\n')
        print("最终满的叶子节点个数为：", full)
        ha.close()
    # with open(save_name + index_name+"+1-" + "k="+str(opt.knn_num)+"topk="+str(opt.reassign_bin_num) + "reassignL3P_result.txt", "w") as result:
    #     result.write("the number of every point make random times: "+str(find_num)+"\n")
    #     result.write("knn_point_num: " + str(len(point_weigh_matrix)) + '\n')
    #     result.write("knn_point_bin_num: " + str(len(leafWithItems)) + "\n")
    #     result.write("the full bin number of the first assgin point is: " + str(first_full_bin) + '\n')
    #     result.write("number of conflict points  is " + str(first_pointWithNoBin) + "\n")
    #     result.write("conflict ratio is "+ str(first_pointWithNoBin*1.0/len(point_weigh_matrix))+"\n")
    #     result.write("FINALL the full bin num is  " + str(full) + "\n")
    #     result.write("ave_edge is "+str(ave_edge)+"\n")
    # result.close()
    # print("bin_num= ",bin_num)
    # print("all_bin_data_num= ",data_num)
    file_Rand = open("LFNN_reassign_Rand_result" + index_name + ".csv", 'a+')
    file_result = csv.writer(file_Rand)
    file_result.writerow(
        [dataname, "LFNN", index_name, "RandGamma",
         "topk: " + str(opt.topk) + " reassign_bin_num: " + str(opt.reassign_bin_num) ,
         str(opt.knn_num),
         str(trainDataset.shape[0]),
         str((time_end-time_start)/60) + "min",
         str(len(point_weigh_matrix)),#knn_point_num
         str(find_num),#the number of every point make random times
         str(len(leafWithItems)),#knn_point_bin_num
         str(first_full_bin),#the full bin number of the first assgin point
         str(first_pointWithNoBin),#number of conflict points
         str(first_pointWithNoBin * 1.0 / len(point_weigh_matrix)),#conflict ratio
         str(full),#FINALL the full bin num is
         str(ave_edge),#ave_edge is
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


