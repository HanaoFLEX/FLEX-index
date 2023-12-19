import h5py as h5
# from config import opt
import numpy as np

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

