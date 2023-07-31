import h5py as h5
# from config import opt
import numpy as np
# #
# def read_datas(path = opt.data_file, key = "dataset"):
#     f = h5.File(path, "r")
#     x = f[key][:]
#     f.close()
#     return x
# file_name = opt.data_file
def Label_LoadDataset(file_name) :
    with h5.File(file_name, "r") as f:

        # for key in f.keys():
        #     print(f[key].name)

        label = f["trainlabel"]

        label = np.asarray(label)


        return label



def LoadDataset(file_name, test=False):

    with h5.File(file_name, "r") as f:
        # np.savetxt("generate_data/iiii.txt", f)
        # print("ok")
        # for key in f.keys():
        #     print(f[key].name)
        #     print(f[key].shape)
            # print(f[key].value)
        dataMatrix = f["dataMatrix"]

        trainDataset = f["trainDataset"]
        trainKnn = f["trainKnn"]
        #
        testDataset = f["testDataset"]
        testKnn = f["testKnn"]

        dataMatrix = np.asarray(dataMatrix)

        trainDataset = np.asarray(trainDataset)
        trainKnn = np.asarray(trainKnn)

        testDataset = np.asarray(testDataset)
        testKnn = np.asarray(testKnn)

        if test:
            result = (dataMatrix, testDataset, testKnn)

        else:
            result = (dataMatrix, trainDataset, trainKnn)
        # result = (dataMatrix, trainDataset, trainKnn)

        return result

def PM_LoadDataset(file_name, test=False):

    with h5.File(file_name, "r") as f:
        # np.savetxt("generate_data/iiii.txt", f)
        print("ok")
        for key in f.keys():
            print(f[key].name)
            # print(f[key].shape)
            # print(f[key].value)
        dataMatrix = f["dataMatrix"]

        trainDataset = f["trainDataset"][:10]
        # trainlabel = f["trainlabel"]

        testDataset = f["testDataset"][:10]
        # testrealKnn = f["testrealKnn"]

        dataMatrix = np.asarray(dataMatrix)
        #
        trainDataset = np.asarray(trainDataset)
        # trainlabel = np.asarray(trainlabel)

        testDataset = np.asarray(testDataset)
        # testrealKnn = np.asarray(testrealKnn)

        # if test:
        #     result = (dataMatrix, testDataset, testrealKnn)
        #
        # else:
        #     result = (dataMatrix, trainDataset, trainlabel)
        # # result = (dataMatrix, trainDataset, trainKnn)

        return dataMatrix,trainDataset,testDataset

def LoadDataset_sift(file_name, test=False):

    with h5.File(file_name, "r") as f:
        # np.savetxt("generate_data/iiii.txt", f)
        print("ok")
        for key in f.keys():
            print(f[key].name)
            # print(f[key].shape)
            # print(f[key].value)
        dataMatrix = f["dataset"]

        trainDataset = f["train_dataset"]
        trainKnn = f["train_knn"]
        #
        testDataset = f["test_dataset"]
        testKnn = f["test_knn"]

        dataMatrix = np.asarray(dataMatrix)

        trainDataset = np.asarray(trainDataset)
        trainKnn = np.asarray(trainKnn)

        testDataset = np.asarray(testDataset)
        testKnn = np.asarray(testKnn)

        if test:
            result = (dataMatrix, testDataset, testKnn)

        else:
            result = (dataMatrix, trainDataset, trainKnn)
        # result = (dataMatrix, trainDataset, trainKnn)

        return result

if __name__ == "__main__":
    dir_name = "dataset+Knn/"
    FileList = ["audio", "sun",
                "enron", "nuswide",
                "notre","sift"]

    i = 5
    file = dir_name + FileList[i] + '/'

    file = file+"sift.hdf5"
    print(file)

    dataMatrix, testDataset, testKnn = LoadDataset_sift(file,test = True)

    dataMatrix, trainDataset, trainKnn =LoadDataset_sift(file,test = False)

    print(dataMatrix.shape)
    print(testDataset.shape)
    print(trainDataset.shape)
    print(trainKnn.shape)
    print(testKnn.shape)