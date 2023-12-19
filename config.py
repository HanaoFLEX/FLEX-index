class DefaultConfig(object):
    knn_num = 50  #knn中的k
    # leaf_max_num = 100 #
    # data_file = "audio.hdf5" #数据文件的路径
    topk =50 # 最终遍历叶子节点数量
    reassign_bin_num = 2 # 再分配到叶子节点数量
    train_topk = 50
    ratio = 1.2
    Avg_Dis = 40000
    train_rate = 0.8
opt = DefaultConfig()
