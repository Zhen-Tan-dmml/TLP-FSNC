import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import CoraFull, Reddit2, Coauthor, Planetoid, Amazon
import random
import numpy as np

# class_split = {"train": 0.6,"test": 0.4}

class_split = {
    "CoraFull": {"train": 40, 'dev': 15, 'test': 15},  # Sufficient number of base classes
    "ogbn-arxiv": {"train": 20, 'dev': 10, 'test': 10},
    "Coauthor-CS": {"train": 5, 'dev': 5, 'test': 5},
    "Amazon-Computer": {"train": 4, 'dev': 3, 'test': 3},
    "Cora": {"train": 3, 'dev': 2, 'test': 2},
    "CiteSeer": {"train": 2, 'dev': 2, 'test': 2},
    "Reddit": {"train": 21, 'dev': 10, 'test': 10},
}


def split(dataset_name):
    
    if dataset_name == 'Cora':
        dataset = Planetoid(root='~/dataset/' + dataset_name, name="Cora")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'CiteSeer':
        dataset = Planetoid(root='~/dataset/' + dataset_name, name="CiteSeer")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'Amazon-Computer':
        dataset = Amazon(root='~/dataset/' + dataset_name, name="Computers")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'Coauthor-CS':
        dataset = Coauthor(root='~/dataset/' + dataset_name, name="CS")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'CoraFull':
        dataset = CoraFull(root='../dataset/' + dataset_name)
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'Reddit':
        dataset = Reddit2(root='../dataset/' + dataset_name)
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name = dataset_name, root='../dataset/' + dataset_name)
        num_nodes = dataset.data.num_nodes
    else:
        print("Dataset not support!")
        exit(0)
    data = dataset.data
    class_list = [i for i in range(dataset.num_classes)]
    print("********" * 10)

    train_num = class_split[dataset_name]["train"]
    dev_num = class_split[dataset_name]["dev"]
    test_num = class_split[dataset_name]["test"]

    random.shuffle(class_list)
    train_class = class_list[: train_num]
    dev_class = class_list[train_num : train_num + dev_num]
    test_class = class_list[train_num + dev_num :]
    print("train_num: {}; dev_num: {}; test_num: {}".format(train_num, dev_num, test_num))

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(torch.squeeze(data.y).tolist()):
        id_by_class[cla].append(id)

    train_idx = []
    for cla in train_class:
        train_idx.extend(id_by_class[cla])

    degree_inv = num_nodes / (dataset.data.num_edges * 2)

    return dataset, np.array(train_idx), id_by_class, train_class, dev_class, test_class, degree_inv


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



