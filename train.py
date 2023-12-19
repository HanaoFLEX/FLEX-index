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

import csv
# from sklearn.metrics import recall_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu')

print(device)

def trainModel(model, file_name,trainDataset,trainlabel,model_name,label_file_name,index_name,rate,datasetname):

    save_name = ""
    dir_list = label_file_name.split("/")
    for i in range(len(dir_list) - 1):
        datasetname = dir_list[i]
        save_name += dir_list[i] + "/"

    save_file_name = save_name +model_name+'/'
    iter = 10
    learnrate = 0.0005

    input_dim = len(trainDataset[0])
    output_dim = len(trainlabel[0])
    net = model(input_dim, output_dim).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learnrate)  # 0.0005 optim.Adam

    loss_function = nn.CrossEntropyLoss()

    target = torch.Tensor(trainlabel)
    query = torch.Tensor(trainDataset)

    train_ids = TensorDataset(query, target)

    trainloader = DataLoader(dataset=train_ids, batch_size=50, shuffle=True, num_workers=0)


    all_time =time.time()

    print(
        "00       10       20       30epoch\n|---------|---------|---------|\n",
        end="")

    for epoch in range(iter):  # 20 loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)

            loss = loss_function(outputs, labels) #交叉熵 <class 'torch.Tensor'> tensor(214.1839, device='cuda:0', grad_fn=<DivBackward1>)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss
        print("*", end="")

    print("*", end="")
    # file.write("train_use_time: " + str(time.time() - all_time) + "s\n")
    print("\n",file_name, "train_use_time:", time.time() - all_time)

    file_LFNN = open("LFNNtrain_time" + index_name + ".csv", 'a+')
    file_result = csv.writer(file_LFNN)
    file_result.writerow(
        [datasetname, "LFNN", index_name, model_name,
         "trainrate:" + str(rate) + " topk: " + str(opt.topk) + " reassign_bin_num: " + str(
             opt.reassign_bin_num),
         str(opt.knn_num),
         str(time.time() - all_time) + "s\n",
         str(rate), "epoch:"+str(iter),"lr:"+str(learnrate),
         ])
    file_LFNN.close()

    net.eval()
    torch.save(net.state_dict(), save_file_name + index_name+"_net_latest.pth")
