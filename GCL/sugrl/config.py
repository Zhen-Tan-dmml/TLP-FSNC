import pickle
import random
import numpy as np

class_split = {
    "CoraFull": {"train": 40, 'dev': 15, 'test': 15},  # Sufficient number of base classes
    "Reddit": {"train": 21, 'dev': 10, 'test': 10},
    "ogbn-arxiv": {"train": 20, 'dev': 10, 'test': 10},
    "Coauthor": {"train": 5, 'dev': 5, 'test': 5},
    "Computers": {"train": 4, 'dev': 3, 'test': 3},
    "Cora": {"train": 3, 'dev': 2, 'test': 2},
    "CiteSeer": {"train": 2, 'dev': 2, 'test': 2},
}


config = {
    "seed": 1234,
    "dataset": "ogbn-arxiv", # CoraFull(70)/Coauthor(15)/ogbn-arxiv(40)/Cora(7)/Computers(10)/CiteSeer(6)
    "batch_size": 128,
    "n_way": 5,
    "k_shot": 5,
    "m_qry": 10,
    "test_num": 20,
    "patience": 10,
    "sup": False,
    "epoch_num": 10000,
}

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



