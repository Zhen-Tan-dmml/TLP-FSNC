from unicodedata import name
from dgl.data import CoraFullDataset, CoauthorCSDataset, AmazonCoBuyComputerDataset, CoraGraphDataset, CiteseerGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from scipy import sparse
from torch.nn.functional import one_hot
from config import *
import random
from os.path import exists, join
from os import makedirs
import torch
import numpy as np


data_name = config["dataset"]
if data_name == "CoraFull":
    dataset = CoraFullDataset()
elif data_name == "Amazon-Computer":
    dataset = AmazonCoBuyComputerDataset()
elif data_name == "Cora":
    dataset = CoraGraphDataset()
elif data_name == "CiteSeer":
    dataset = CiteseerGraphDataset()
elif data_name == "Coauthor-CS":
    dataset = CoauthorCSDataset()
elif data_name == "ogbn-arxiv":
    dataset = DglNodePropPredDataset(name = 'ogbn-arxiv', root = '../dataset')
    dataset = dataset[0]
adj = dataset[0].adj().to_dense().numpy().astype(int)
adj = sparse.csr_matrix(adj) 
features = dataset[0].ndata['feat'].numpy()
labels = dataset[0].ndata['label']

class_list = [i for i in range(dataset.num_classes)]
train_num = class_split[data_name]["train"]
dev_num = class_split[data_name]["dev"]
test_num = class_split[data_name]["test"]
random.shuffle(class_list)
train_class = class_list[: train_num]
dev_class = class_list[train_num : train_num + dev_num]
test_class = class_list[train_num + dev_num:]

id_by_class = {}
for i in class_list:
    id_by_class[i] = []
for id, cla in enumerate(torch.squeeze(labels).tolist()):
    id_by_class[cla].append(id)

idx_train = []
for cla in train_class:
    idx_train.extend(id_by_class[cla])
labels = one_hot(labels).numpy()

print("train_num: {}; dev_num: {}; test_num: {}".format(train_num, dev_num, test_num))

path = join("./fs_data", data_name)
if not exists(path):
    makedirs(path)
save_object(adj, join(path, "adj.pk"))
save_object(features, join(path, "features.pk"))
save_object(labels, join(path, "labels.pk"))
save_object(train_class, join(path, "train_class.pk"))
save_object(dev_class, join(path, "dev_class.pk"))
save_object(test_class, join(path, "test_class.pk"))
save_object(id_by_class, join(path, "id_by_class.pk"))

 


 