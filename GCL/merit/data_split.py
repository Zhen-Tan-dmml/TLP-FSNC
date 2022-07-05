from dgl.data import CoraGraphDataset, CoraFullDataset, RedditDataset, CoauthorCSDataset, AmazonCoBuyComputerDataset, CiteseerGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
import numpy as np
import os
import random
import torch
from config import *
from torch.nn.functional import one_hot


def download(dataset):
    if dataset == 'Cora':
        return CoraGraphDataset()
    elif dataset == 'CoraFull':
        return CoraFullDataset(raw_dir="../dataset")
    elif dataset == 'Reddit':
        return RedditDataset()
    elif dataset == 'ogbn-arxiv':
        return DglNodePropPredDataset(name='ogbn-arxiv', root="../dataset")
    elif dataset == 'Coauthor-CS':
        return CoauthorCSDataset()
    elif dataset == 'Amazon-Computer':
        return AmazonCoBuyComputerDataset()
    elif dataset == 'CiteSeer':
        return CiteseerGraphDataset()
    else:
        print("dataset not support!")
        return None


def load(dataset):
    datadir = os.path.join('data', dataset)
    # class_split = {"train": 0.6, "test": 0.4}

    ds = download(dataset)
    if dataset == 'ogbn-arxiv':
        ds = ds[0]
    adj = ds[0].adj().to_dense().numpy().astype(int)
    adj = sp.csr_matrix(adj)
    feat = ds[0].ndata['feat'][:]
    labels = ds[0].ndata['label'][:]

    class_list = [i for i in range(ds.num_classes)]
    train_num = class_split[dataset]["train"]
    dev_num = class_split[dataset]["dev"]
    test_num = class_split[dataset]["test"]
    random.shuffle(class_list)
    train_class = class_list[: train_num]
    dev_class = class_list[train_num : train_num + dev_num]
    test_class = class_list[train_num + dev_num:]
    print("train_num: {}; dev_num: {}; test_num: {}".format(train_num, dev_num, test_num))
    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(torch.squeeze(labels).tolist()):
        id_by_class[cla].append(id)

    idx_train = []
    for cla in train_class:
        idx_train.extend(id_by_class[cla])

    labels = one_hot(labels).numpy()

    return adj, feat, labels, train_class, dev_class, test_class, id_by_class


def test_task_generator(id_by_class, class_list, n_way, k_shot, m_query):

    # sample class indices
    class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []

    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected


if __name__ == '__main__':
    load('CiteSeer')
#     return
