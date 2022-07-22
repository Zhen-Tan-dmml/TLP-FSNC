import pickle
import random
import numpy as np
from copy import deepcopy
import torch

class_split = {
    "CoraFull": {"train": 40, 'dev': 15, 'test': 15},  # Sufficient number of base classes
    "Reddit": {"train": 21, 'dev': 10, 'test': 10},
    "ogbn-arxiv": {"train": 20, 'dev': 10, 'test': 10},
    "Coauthor-CS": {"train": 5, 'dev': 5, 'test': 5},
    "Amazon-Computer": {"train": 4, 'dev': 3, 'test': 3},
    "Cora": {"train": 3, 'dev': 2, 'test': 2},
    "CiteSeer": {"train": 2, 'dev': 2, 'test': 2},
}


config = {
    "seed": 1234,
    "dataset": "CiteSeer", # CoraFull(70)/Coauthor-CS(15)/ogbn-arxiv(40)/Cora(7)/Amazon-Computer(10)/CiteSeer(6)
    "batch_size": 128,
    "n_way": 2,
    "k_shot": 5,
    "m_qry": 10,
    "test_num": 100,
    "patience": 10,
    "sup": "sup",
    "epoch_num": 10000,
}


def relabeling(labels, train_class, dev_class, test_class, id_by_class):
    print("Start relabeling...")
    labels = torch.argmax(labels[0], dim=1)
    labels = labels.tolist()
    contrast_labels = deepcopy(labels)
    masked_class = dev_class + test_class
    masked_idx = []
    for cla in masked_class:
        masked_idx.extend(id_by_class[cla])

    train_class.sort()
    train_class_map = {i: train_class.index(i) for i in train_class}

    tmp_class = len(train_class)
    for cla, idx_list in id_by_class.items():
        if cla in train_class:
            for idx in idx_list:
                contrast_labels[idx] = train_class_map[cla]
        else:
            for idx in idx_list:
                contrast_labels[idx] = tmp_class
                tmp_class += 1
    print("Relabeling finished!")
    return contrast_labels


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


def save_object(obj, filename):
    with open(filename, 'wb') as fout:  # Overwrites any existing file.
        pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)



