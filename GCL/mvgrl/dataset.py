from dgl.data import CoraGraphDataset, CoraFullDataset, RedditDataset, CoauthorCSDataset, AmazonCoBuyComputerDataset, CiteseerGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from utils import preprocess_features, normalize_adj
from sklearn.preprocessing import MinMaxScaler
from utils import compute_ppr_all
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os
import random
import torch
from config import *


def download(dataset):
    if dataset == 'Cora':
        return CoraGraphDataset()
    elif dataset == 'CoraFull':
        return CoraFullDataset()
    elif dataset == 'Reddit':
        return RedditDataset()
    elif dataset == 'ogbn-arxiv':
        return DglNodePropPredDataset(name = "ogbn-arxiv", root='../dataset')
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

    if not os.path.exists(datadir):
        os.makedirs(datadir)
        ds = download(dataset)
        if dataset == 'ogbn-arxiv':
            ds = ds[0]
        adj = ds[0].adj().to_dense().numpy()
        diff = compute_ppr_all(adj, 0.2)
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

        # idx_train = []
        # for cla in train_class:
        #     idx_train.extend(id_by_class[cla])
        
        np.save(f'{datadir}/adj.npy', adj)
        np.save(f'{datadir}/diff.npy', diff)
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/labels.npy', labels)
        # np.save(f'{datadir}/idx_train.npy', np.array(idx_train))
        np.save(f'{datadir}/train_class.npy', np.array(train_class))
        np.save(f'{datadir}/dev_class.npy', np.array(dev_class))
        np.save(f'{datadir}/test_class.npy', np.array(test_class))
        np.save(f'{datadir}/id_by_class.npy', id_by_class)
    else:
        adj = np.load(f'{datadir}/adj.npy')
        diff = np.load(f'{datadir}/diff.npy')
        feat = np.load(f'{datadir}/feat.npy')
        labels = np.load(f'{datadir}/labels.npy')
        # idx_train = np.load(f'{datadir}/idx_train.npy')
        train_class = np.load(f'{datadir}/train_class.npy')
        dev_class = np.load(f'{datadir}/dev_class.npy')
        test_class = np.load(f'{datadir}/test_class.npy')
        id_by_class = np.load(f'{datadir}/id_by_class.npy', allow_pickle=True).item()

    if dataset == 'citeseer':
        feat = preprocess_features(feat)

        epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        avg_degree = np.sum(adj) / adj.shape[0]
        epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0])
                                      for e in epsilons])]

        diff[diff < epsilon] = 0.0
        scaler = MinMaxScaler()
        scaler.fit(diff)
        diff = scaler.transform(diff)

    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    return adj, diff, feat, labels, train_class, dev_class, test_class, id_by_class


def test_task_generator(id_by_class, class_list, n_way, k_shot, m_query):

    # sample class indices
    class_selected = random.sample(class_list.tolist(), n_way)
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
