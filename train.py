import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import datas_load as load
from torch.utils.data import TensorDataset
import time
from config import opt
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import DSCNN
import numpy as np
import math
import csv
# from sklearn.metrics import recall_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu')

print(device)



def lossFunction1(outputs,labels):
    weights = torch.zeros_like(labels)
    weights.copy_(labels)
    weights[weights==1]=1000.0
    weights[weights == 0]=1.0
    loss = 0.0
    # print(labels)
    for i in range(labels.shape[0]):
        loss = -torch.sum(weights[i]*labels[i] * torch.log(outputs[i])) + loss
        # print(outputs[i])

    return (loss/(labels.shape[0]*1.0)).requires_grad_(True)

def lossFunction2(outputs,labels):
    weights = torch.zeros_like(labels)
    weights.copy_(labels)
    weights[weights == 1] = 1.0
    weights[weights == 0] = 1.0
    loss = 0.0


    for i in range(outputs.shape[0]):
        criterion = nn.CrossEntropyLoss(
            weight=torch.Tensor(weights[i]),
            size_average=True)
        ### 要想使用GPU，此处必须使用cuda()
        criterion.cuda()
        ### y为预测结果， tags为原标签
        loss = criterion(outputs[i], labels[i]) + loss

    # knn_loss = 0.0
    # noknn_loss=0.0
    # file_LFNN = open("LFNN_result" + "weightLoss" + ".csv", 'a+')
    # file_result = csv.writer(file_LFNN)
    #     index = torch.nonzero(labels[i] == 1).squeeze()
    #     output_knn = torch.zeros(len(index))
    #     label_knn = torch.ones(len(index))
    #     criterion1 = nn.CrossEntropyLoss(
    #         weight=torch.ones(len(index)),
    #         size_average=True)
    #     for j in range(len(index)):
    #         output_knn[j]=outputs[i][index[j]]
    #     knn_loss= criterion1(output_knn, label_knn) +knn_loss
    #
    #     index2 = torch.nonzero(labels[i] == 0).squeeze()
    #     output_noknn = torch.zeros(len(index2))
    #     label_noknn = torch.zeros(len(index2))
    #     criterion2 = nn.CrossEntropyLoss(
    #         weight=torch.ones(len(index2)),
    #         size_average=True)
    #     for j in range(len(index)):
    #         output_noknn[j] = outputs[i][index[j]]
    #     noknn_loss = criterion2(output_noknn, label_noknn) + noknn_loss
    #
    # file_result.writerow(["knnloss"+str(knn_loss/(outputs.shape[0]*1.0)),"noknnloss"+str(noknn_loss/(outputs.shape[0]*1.0)),"loss"+str(loss/(outputs.shape[0]*1.0))])
    # print(knn_loss/(outputs.shape[0]*1.0),noknn_loss/(outputs.shape[0]*1.0),loss/(outputs.shape[0]*1.0))
    # file_LFNN.close()
    return loss/(outputs.shape[0]*1.0)

def lossFunction3(outputs,labels):
    c = 0.001
    # print(outputs)
    weights = torch.zeros_like(outputs)
    weights.copy_(outputs)
    weights[weights<c]=0.0
    weights[weights >= c]=1.0
    loss = 0.0

    for i in range(outputs.shape[0]):
        criterion = nn.CrossEntropyLoss(
            weight=torch.Tensor(weights[i]),
            size_average=True)
        ### 要想使用GPU，此处必须使用cuda()
        criterion.cuda()
        ### y为预测结果， tags为原标签
        loss = criterion(outputs[i], labels[i]) + loss
    return loss / (outputs.shape[0] * 1.0)

def lossFunctiontopk(outputs,labels):
    k=100
    copy_out = torch.zeros_like(outputs)
    copy_out.copy_(outputs)
    copy_out, indices = torch.topk(copy_out,k,dim=1)
    weights = torch.zeros_like(outputs)
    loss = 0.0

    for i in range(indices.shape[0]):
            weights[i][indices[i]] = 1

    for i in range(outputs.shape[0]):
        criterion = nn.CrossEntropyLoss(
            weight=torch.Tensor(weights[i]),
            size_average=True)
        ### 要想使用GPU，此处必须使用cuda()
        criterion.cuda()
        ### y为预测结果， tags为原标签
        loss = criterion(outputs[i], labels[i]) + loss
    return loss / (outputs.shape[0] * 1.0)





def trainModel(model, file_name,trainDataset,trainlabel,model_name,label_file_name,index_name,rate):

    save_name = ""
    dir_list = label_file_name.split("/")
    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"

    save_file_name = save_name +model_name+'/'

    print(trainDataset.shape)
    print(trainlabel.shape)

    # print(trainlabel[0])
    input_dim = len(trainDataset[0])
    output_dim = len(trainlabel[0])
    net = model(input_dim, output_dim).to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0005)  # 0.0005 optim.SGD

    loss_function = nn.CrossEntropyLoss()

    target = torch.Tensor(trainlabel)
    query = torch.Tensor(trainDataset)

    train_ids = TensorDataset(query, target)

    trainloader = DataLoader(dataset=train_ids, batch_size=50, shuffle=True, num_workers=0)

    file = open(save_file_name + index_name + str(opt.train_topk)+'train_result.txt', 'w')
    file.write("batchsize: "+str(trainloader.batch_size)+'\n')
    all_time =time.time()

    file_LFNN = open("Loss_epoch" + index_name +model_name+str(rate)+ ".csv", 'a+')
    file_result = csv.writer(file_LFNN)


    print(
        "00       10       20       30epoch\n|---------|---------|---------|\n",
        end="")

    for epoch in range(10):  # 20 loop over the dataset multiple times

        running_loss = 0.0

        # 一个for i 循环是一批，所以如果训练集200条数据 然后i为（0-24） （batchsize=8  8*25=200）
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data

            # # labels 补充到长度可以开根号 when no m-fc
            # num = math.ceil(math.sqrt(labels.shape[1])) * math.ceil(math.sqrt(labels.shape[1])) - labels.shape[1]
            # array = np.zeros((labels.shape[0],num))
            # array = torch.Tensor(array)
            # labels=torch.cat((labels, array), 1)

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            # print(outputs.shape, labels.shape)
            # loss = lossFunctiontopk(outputs, labels)# 重写

            loss = loss_function(outputs, labels) #交叉熵 <class 'torch.Tensor'> tensor(214.1839, device='cuda:0', grad_fn=<DivBackward1>)

            # print(type(loss),loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss
        print("*", end="")
        file_result.writerow(
            [str(epoch),
             str(running_loss.item() / len(trainDataset))])
    print("*", end="")
    file.write("train_use_time: " + str(time.time() - all_time) + "s\n")
    print("\n",file_name, "train_use_time:", time.time() - all_time)
    net.eval()
    torch.save(net.state_dict(), save_file_name + index_name+"_net_latest.pth")
    file.close()
    file_LFNN.close()


