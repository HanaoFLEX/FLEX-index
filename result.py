import torch
import time
from torch.utils.data import DataLoader
from config import opt
from find_tmp_knn import Knn
import datas_load as load
from cal_acc import  cal_overall_radio
from getindex import getindex
from get_bin import read_bin
import csv

torch.set_printoptions(profile="full")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu')
print(device)

def testModel(model, file_name,index_name,Modle_name,bin_name,trainset_rate):
    save_name = ""
    dir_list = file_name.split("/")
    for i in range(len(dir_list) - 1):
        save_name += dir_list[i] + "/"
    dataname = dir_list[i]
    save_name += index_name + "/"
    save_file_name = save_name +Modle_name+ '/'

    dataMatrix, testDataset, testKnn = load.LoadDataset(file_name,test = True)
    print(testDataset.shape)

    if bin_name =="bin":
        bin_path = save_name +index_name+ "bin.txt"
    elif bin_name == "Rand_gamma":
        bin_path = save_name +index_name+ "k="+str(opt.knn_num)+"topk="+str(opt.reassign_bin_num) +"rand_gamma_bin.txt"


    f = open(bin_path, 'r')
    bin_num = int(f.readline())
    outlabel_len = bin_num

    f.close()

    input_dim = len(testDataset[0])
    output_dim = outlabel_len
    test_ids = torch.Tensor(testDataset).to(device)

    testloader = DataLoader(dataset=test_ids, batch_size=50, shuffle=False)
    id = getindex(file_name)
    bin = read_bin(bin_path)

    net = model(input_dim,output_dim).to(device)
    net.load_state_dict(torch.load(save_file_name +index_name+"_net_latest.pth"))
    net.eval()
    print("加载参数完毕")
    SumTime = 0.
    sum_overcall = 0.
    sum_recall = 0.
    j = 0
    comPDists = 0
    print("0%   10   20   30   40   50   60   70   80   90   100%\n|----|----|----|----|----|----|----|----|----|----|\n",end="")

    c = int(testDataset.shape[0]/50)
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):

            inputs = data.to(device)

            break1 = time.perf_counter()
            outputs = net(inputs)
            break2 = time.perf_counter()
            SumTime += break2 - break1
            for index in range(len(inputs)):

                result = outputs[index]

                # 获取当前测试的是第几个，用于求召回率的函数
                x = index + j * testloader.batch_size

                break3 = time.perf_counter()

                candid = []
                _, predicted = torch.topk(result.data, k=opt.topk, dim=0)

                for pr in predicted:
                    if(int(pr)<bin_num):
                        candid.extend(bin[int(pr)])
                comPDists += len(candid)
                knn1, dis1 = Knn(opt.knn_num, candid, dataMatrix, testDataset[x])
                break4 = time.perf_counter()

                SumTime += break4 - break3
                recall, flag, findknn_num = cal_overall_radio(file_name, testKnn[x][0], testKnn[x][1], knn1,
                                                                       dis1)
                # print("\nrecall",recall)

                sum_recall = sum_recall + recall
                if x%c == 0:
                    print("*",end="")

            j = j + 1


    print("\nave_recall: ", (sum_recall/ len(testDataset))*100, "%")
    print("ave_comPDists: ", comPDists / len(testDataset))
    print("ave query time：{}\n".format(SumTime / testDataset.shape[0]))
    print("trainset_rate",trainset_rate)

    file_LFNN = open("LFNN_result"+index_name+".csv", 'a+')
    file_result = csv.writer(file_LFNN)
    file_result.writerow(
        [dataname, "LFNN",index_name, Modle_name,bin_name,"trainrate:"+str(trainset_rate)+" topk: " + str(opt.topk) + " reassign_bin_num: " + str(opt.reassign_bin_num) ,
         str(opt.knn_num),
         str(testDataset.shape[0]),
         str(comPDists / len(testDataset)),
         str(SumTime / testDataset.shape[0]*1000) + "ms",
         str((sum_recall/ len(testDataset))*100)+"%",
         str(trainset_rate)])
    file_LFNN.close()

