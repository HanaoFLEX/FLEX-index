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
import DSCNN

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


    # dataMatrix, trainDataset, trainKnn = load.LoadDataset(file_name)
    # testDataset = trainDataset
    # testKnn = trainKnn
    if bin_name =="bin":
        bin_path = save_name +index_name+ "bin.txt"
        # bin_path = save_name + "PMbin.txt"
    elif bin_name == "Rand_WeightNoSort":
        bin_path = save_name +index_name+"k="+str(opt.knn_num)+"topk="+str(opt.reassign_bin_num)+"Rand_WeightNoSort_bin.txt"
    elif bin_name == "reassignGre":
        bin_path = save_name +index_name+"k="+str(opt.knn_num)+"topk="+str(opt.reassign_bin_num)+"reassign_weight_1_bin.txt"
    elif bin_name == "Rand_gamma":
        bin_path = save_name +index_name+ "k="+str(opt.knn_num)+"topk="+str(opt.reassign_bin_num) +"rand_gamma_bin.txt"
    elif bin_name == "reassignRand":
        bin_path = save_name + index_name + "k="+str(opt.knn_num)+"topk="+str(opt.reassign_bin_num) + "reassignL3p_bin.txt"

    f = open(bin_path, 'r')
    bin_num = int(f.readline())
    # outlabel_len = math.ceil(math.sqrt(bin_num)) * math.ceil(math.sqrt(bin_num))
    outlabel_len = bin_num

    f.close()

    input_dim = len(testDataset[0])
    output_dim = outlabel_len
    # print(input_dim,output_dim,"+++++++++++")
    # testDataset = testDataset[:40000]
    test_ids = torch.Tensor(testDataset).to(device)

    testloader = DataLoader(dataset=test_ids, batch_size=50, shuffle=False)
    id = getindex(file_name)
    bin = read_bin(bin_path)

    net = model(input_dim,output_dim).to(device)
    # h） .load加载预训练参数 .load_state_dict将预训练参数加载到模型中
    net.load_state_dict(torch.load(save_file_name +index_name+"_net_latest.pth"))
    # net.load_state_dict(torch.load(save_file_name + index_name  + "1_net_latest.pth"))
    net.eval()
    print("加载参数完毕")
    SumTime = 0.
    sum_overcall = 0.
    sum_recall = 0.
    AVG = 0
    j = 0
    # file = open(save_file_name + index_name+bin_name+"knn="+str(opt.knn_num)+" topk="+str(opt.topk)+'test_result.txt', 'w')
    # file.write(str(file_name)+"\n")
    comPDists = 0
    print("0%   10   20   30   40   50   60   70   80   90   100%\n|----|----|----|----|----|----|----|----|----|----|\n",end="")

    c = int(testDataset.shape[0]/50)
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            time11 = time.time()
            # inputs, labels = data
            inputs = data.to(device)

            break1 = time.perf_counter()
            outputs = net(inputs)
            break2 = time.perf_counter()
            SumTime += break2 - break1
            # print("\n\n一批过完Model",time.time()-time11)
            for index in range(len(inputs)):

                result = outputs[index]

                # 获取当前测试的是第几个，用于求召回率的函数
                x = index + j * testloader.batch_size
                # print(x,"\nresult_label", result)
                # print('\n',x)

                # csv_writer.writerow([x, result])
                break3 = time.perf_counter()

                candid = []
                _, predicted = torch.topk(result.data, k=opt.topk, dim=0)
                # print("predicted",predicted)
                # print(predicted,"\n")
                for pr in predicted:
                    if(int(pr)<bin_num):
                        candid.extend(bin[int(pr)])
                comPDists += len(candid)
                # candid, max_sigma_number = find_candid_point(bin,result)
                # print("max_sigma_number",max_sigma_number)

                # knn1, dis1 = find_knn(candid,dataMatrix, testDataset[x], id)
                knn1, dis1 = Knn(opt.knn_num, candid, dataMatrix, testDataset[x])
                break4 = time.perf_counter()
                # print("knn+for_time",break4-break3)
                # last_time = break4 - break3

                SumTime += break4 - break3
                overall, recall, flag, findknn_num = cal_overall_radio(file_name, testKnn[x][0], testKnn[x][1], knn1,
                                                                       dis1)
                # print("\nrecall",recall)

                sum_recall = sum_recall + recall
                sum_overcall = sum_overcall +overall
                if x%c == 0:
                    print("*",end="")

            j = j + 1


    print("\nave_recall: ", (sum_recall/ len(testDataset))*100, "%")
    print("ave_overcall: ",sum_overcall/len(testDataset))
    print("ave_comPDists: ", comPDists / len(testDataset))
    print("ave query time：{}\n".format(SumTime / testDataset.shape[0]))
    print("trainset_rate",trainset_rate)
    # file.write("ave_recall: "+str((sum_recall/ len(testDataset))*100)+"%\n")
    # file.write("ave_overcall: "+str(sum_overcall/len(testDataset))+'\n')
    # file.write("ave query time: "+str(SumTime / testDataset.shape[0]*1000)+"ms\n")
    # file.write("ave_comPDists: " + str(comPDists / len(testDataset)) + '\n')
    # file.close()
    file_LFNN = open("LFNN_result"+index_name+".csv", 'a+')
    file_result = csv.writer(file_LFNN)
    file_result.writerow(
        [dataname, "LFNN",index_name, Modle_name,bin_name,"trainrate:"+str(trainset_rate)+" topk: " + str(opt.topk) + " reassign_bin_num: " + str(opt.reassign_bin_num) ,
         str(opt.knn_num),
         str(testDataset.shape[0]),
         str(comPDists / len(testDataset)),
         str(SumTime / testDataset.shape[0]*1000) + "ms",
         str((sum_recall/ len(testDataset))*100)+"%",
         str(sum_overcall/len(testDataset)),
         str(trainset_rate)])
    file_LFNN.close()


if __name__ == '__main__':
    ModleList = [DSCNN.DSCNN_model, Atten_Model.ML_atten_Model, view_attention.view_atten_Model, Model.ML_Model,
                 Model.ML_2_Model, Model.ML_3_Model, Model.ML_4_Model, Model.ML_5_Model]
    Model_name = ["DSCNN_model", "ML_atten_Model", "view_attention", "ML_Model", "ML_2_Model", "ML_3_Model",
                  "ML_4_Model", "ML_5_Model"]
    dataset_name_list = ["audio", "sun", "enron", "nuswide", "notre"]
    # dir_path is  dataset_path
    dir_path = "dataset/"
    file_last_name = "_dataset50NN_PM.hdf5"
    # dataMatrix, trainDataset, trainlabel = load.LoadDataset(fileList[0])
    a = [45]
    for aa in range(1):
        for i in range(1):
            opt.topk = a[aa]
            num = 0
            model = ModleList[num]
            file_name = dir_path + dataset_name_list[i] + '/' + dataset_name_list[i] + file_last_name
            dataMatrix, trainDataset, trainlabel = load.LoadDataset(file_name)
            print(file_name, trainlabel.shape)
            print("topk = ", opt.topk, "\n")
            print("test", file_name)
            testModel(model, file_name, Model_name[num])


