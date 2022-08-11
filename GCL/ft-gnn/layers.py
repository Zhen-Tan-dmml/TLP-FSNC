import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, Reddit2
from ogb.nodeproppred import PygNodePropPredDataset, DglNodePropPredDataset
from sklearn.preprocessing import StandardScaler

class_split = {
    "CoraFull": {"train": 40, 'dev': 15, 'test': 15},  # Sufficient number of base classes
    "ogbn-arxiv": {"train": 20, 'dev': 10, 'test': 10},
    "Coauthor-CS": {"train": 5, 'dev': 5, 'test': 5},
    "Amazon-Computer": {"train": 4, 'dev': 3, 'test': 3},
    "Cora": {"train": 3, 'dev': 2, 'test': 2},
    "CiteSeer": {"train": 2, 'dev': 2, 'test': 2},
    "Reddit": {"train": 21, 'dev': 10, 'test': 10},
}


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GNN_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNN_Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv1 = GraphConvolution(in_channels, self.hidden_channels)
        self.prelu1 = nn.PReLU(self.hidden_channels)

    def forward(self, x, adj):
        x1 = self.conv1(x, adj)
        x1 = self.prelu1(x1)
        return x1


class Classifier(nn.Module):
    def __init__(self, nhid, nclass):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, x):
        x = self.fc(x)
        return x


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def split(dataset_name):
    if dataset_name == 'CoraFull':
        dataset = CoraFull(root='../subg-Con/dataset/' + dataset_name)

        data = dataset.data
        ns = data.edge_index
        n1s = ns[0].tolist()
        n2s = ns[1].tolist()

        num_nodes = max(max(n1s), max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

    elif dataset_name == 'Coauthor-CS':
        dataset = Coauthor(root='../subg-Con/dataset/' + dataset_name, name='CS')

        data = dataset.data
        ns = data.edge_index
        n1s = ns[0].tolist()
        n2s = ns[1].tolist()

        num_nodes = max(max(n1s), max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    elif dataset_name == 'Amazon-Computer':
        dataset = Amazon(root='../subg-Con/dataset/' + dataset_name, name='Computers')

        data = dataset.data
        ns = data.edge_index
        n1s = ns[0].tolist()
        n2s = ns[1].tolist()

        num_nodes = max(max(n1s), max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    elif dataset_name == 'Cora':
        dataset = Planetoid(root='../subg-Con/dataset/' + dataset_name, name='Cora')

        data = dataset.data
        ns = data.edge_index
        n1s = ns[0].tolist()
        n2s = ns[1].tolist()

        num_nodes = max(max(n1s), max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

    elif dataset_name == 'CiteSeer':
        dataset = Planetoid(root='../subg-Con/dataset/' + dataset_name, name='CiteSeer')

        data = dataset.data
        ns = data.edge_index
        n1s = ns[0].tolist()
        n2s = ns[1].tolist()

        num_nodes = max(max(n1s), max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

    elif dataset_name == 'Reddit':
        dataset = Reddit2(root='../subg-Con/dataset/' + dataset_name)

        data = dataset.data
        ns = data.edge_index
        n1s = ns[0].tolist()
        n2s = ns[1].tolist()

        num_nodes = max(max(n1s), max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    elif dataset_name in ['ogbn-arxiv', 'ogbn-products']:
        # dataset = PygNodePropPredDataset(name=dataset_name, root='../subg-Con/dataset/' + dataset_name)
        dataset = DglNodePropPredDataset(name=dataset_name, root='../subg-Con/dataset/' + dataset_name)
        data, labels = dataset[0]
        N = data.num_nodes()
        adj = sp.coo_matrix((data.adjacency_matrix()._values().numpy(), (data.adjacency_matrix()._indices()[0].numpy(),
                                                        data.adjacency_matrix()._indices()[1].numpy())), shape=(N,N))
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        data = data.ndata

    
    if not dataset_name in ['ogbn-arxiv', 'ogbn-products']:
        labels  =data.y
        features = data.x
    else:
        features = data['feat']
    labels = torch.squeeze(labels)
    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = torch.LongTensor(np.where(labels)[1])
    class_list = list(set(labels.tolist()))

    train_num = class_split[dataset_name]['train']
    dev_num = class_split[dataset_name]['dev']
    test_num = class_split[dataset_name]['test']
    random.shuffle(class_list)

    train_class = class_list[: train_num]
    dev_class = class_list[train_num : train_num + dev_num]
    test_class = class_list[train_num + dev_num :]
    print("train_num: {}; dev_num: {}; test_num: {}".format(train_num, dev_num, test_num))

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels.tolist()):
        id_by_class[cla].append(id)

    return adj, features, labels, degree, train_class, dev_class, test_class, id_by_class



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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1


