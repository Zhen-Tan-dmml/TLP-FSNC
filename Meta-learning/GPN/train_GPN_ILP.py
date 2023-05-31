import networkx as nx
import numpy as np
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
import sys
import scipy
import sklearn
import json
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import pickle as pkl
import scipy.sparse as sp
from base_model import GCN
from base_model import GraphConvolution
import time
import datetime
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
#from torch_geometric.data import Data
time_start=time.time()

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    if len(output.shape)==2:
        preds = output.max(1)[1].type_as(labels)
    else:
        preds=output
    f1 = f1_score(labels, preds, average='weighted')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def task_generator(id_by_class, class_list, n_way, k_shot, m_query, maximum_value_train_each_class=None):

    # sample class indices
    class_selected = np.random.choice(class_list, n_way,replace=False).tolist()
    id_support = []
    id_query = []
    for cla in class_selected:
        if maximum_value_train_each_class:
            temp = np.random.choice(id_by_class[cla][:maximum_value_train_each_class], k_shot + m_query,replace=False)
        else:
            temp = np.random.choice(id_by_class[cla], k_shot + m_query,replace=False)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected



def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M





class_split = {
    "CoraFull": {"train": 40, 'dev': 15, 'test': 15},  # Sufficient number of base classes
    "ogbn-arxiv": {"train": 20, 'dev': 10, 'test': 10},
    "Coauthor-CS": {"train": 5, 'dev': 5, 'test': 5},
    "Amazon-Computer": {"train": 4, 'dev': 3, 'test': 3},
    "Cora": {"train": 3, 'dev': 2, 'test': 2},
    "CiteSeer": {"train": 2, 'dev': 2, 'test': 2},
}


config = {
    "seed": 1234,
    "dataset": "Amazon-Computer", # CoraFull(70)/Coauthor-CS(15)/ogbn-arxiv(40)/Cora(7)/Amazon-Computer(10)/CiteSeer(6)
    "batch_size": 4,
    "n_way": 2,
    "k_shot": 5,
    "m_qry": 10,
    "test_num": 20,
    "patience": 10,
    "sup": False,
    "epoch_num": 10000,
}



class_split = {
    "CoraFull": {"train": 40, 'dev': 15, 'test': 15},  # Sufficient number of base classes
    "ogbn-arxiv": {"train": 20, 'dev': 10, 'test': 10},
    "Coauthor-CS": {"train": 5, 'dev': 5, 'test': 5},
    "Amazon-Computer": {"train": 4, 'dev': 3, 'test': 3},
    "Cora": {"train": 3, 'dev': 2, 'test': 2},
    "CiteSeer": {"train": 2, 'dev': 2, 'test': 2},
    'dblp': {"train": 80, 'dev': 27, 'test': 30},
}
class dblp_data():
    def __init__(self):
        self.x=None
        self.edge_index=None
        self.num_nodes=None
        self.y=None
        self.num_edges=None
        self.num_features=7202
class dblp_dataset():
    def __init__(self,data,num_classes):
        self.data=data
        self.num_classes=num_classes

def load_DBLP(root=None, dataset_source='dblp'):
    dataset=dblp_data()
    n1s = []
    n2s = []
    for line in open("../../few_shot_data/{}_network".format(dataset_source)):
        n1, n2 = line.strip().split('\t')
        if True:
            n1s.append(int(n1))
            n2s.append(int(n2))



    num_nodes = max(max(n1s), max(n2s)) + 1
    print('nodes num', num_nodes)

    data_train = sio.loadmat("../../few_shot_data/{}_train.mat".format(dataset_source))
    data_test = sio.loadmat("../../few_shot_data/{}_test.mat".format(dataset_source))

    labels = np.zeros((num_nodes, 1))
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]

    features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()


    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    dataset.edge_index=torch.tensor([n1s,n2s])
    dataset.y=labels
    dataset.x=features
    dataset.num_nodes=num_nodes
    dataset.num_edges=dataset.edge_index.shape[1]

    return dblp_dataset(dataset, num_classes=80+27+30)


def load_data(dataset_source, inductive=False):
    from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.datasets import CoraFull, Coauthor, Planetoid, Amazon

    if dataset_source=='ogbn-arxiv':
        dataset = PygNodePropPredDataset(root='../dataset/ogbn-arxiv', name='ogbn-arxiv')
    elif dataset_source=='CoraFull':
        dataset = CoraFull(root='../dataset/corafull')
    elif dataset_source=='Coauthor-CS':
        dataset= Coauthor(root='../dataset/coauthor-cs',name='CS')
    elif dataset_source=='Cora':
        dataset=Planetoid(root='../dataset/cora',name='Cora')
    elif dataset_source=='CiteSeer':
        dataset=Planetoid(root='../dataset/citeseer',name='CiteSeer')
    elif dataset_source=='Amazon-Computer':
        dataset=Amazon(root='../dataset/amazon',name='Computers')
    elif dataset_source == 'dblp':
        dataset = load_DBLP(root='./few_shot_data/')

    if dataset_source=='dblp':
        graph=dataset.data
    else:
        graph=dataset[0]

    #remove extra dimension
    labels = graph.y.squeeze()
    class_num=(labels.max()+1).item()

    n1s=graph.edge_index[0]
    n2s=graph.edge_index[1]

    num_nodes = graph.x.shape[0]
    print('nodes num',num_nodes)


    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))
    degree = np.sum(adj, axis=1)
    degree = torch.FloatTensor(degree)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    features=graph.x

    train_class_num=class_split[dataset_source]['train']
    valid_class_num=class_split[dataset_source]['dev']
    test_class_num=class_split[dataset_source]['test']
        
    class_list_test = np.random.choice(list(range(class_num)),test_class_num,replace=False).tolist()
    train_class=list(set(list(range(class_num))).difference(set(class_list_test)))
    class_list_valid = np.random.choice(train_class,valid_class_num, replace=False).tolist()
    class_list_train = list(set(train_class).difference(set(class_list_valid)))
    
    #json.dump([class_list_train,class_list_valid,class_list_test],open('./few_shot_data/{}_class_split.json'.format(dataset_source),'w'))
    #class_list_train,class_list_valid,class_list_test=json.load(open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

    idx_train,idx_valid,idx_test=[],[],[]

    for i in range(labels.shape[0]):
        if labels[i] in class_list_train:
            idx_train.append(i)
        elif labels[i] in class_list_valid:
            idx_valid.append(i)
        else:
            idx_test.append(i)
    print(labels.shape)

    if inductive:
        # subgraph
        gen_subgraph_edge_index, _ = subgraph(idx_train, graph.edge_index)
        # start and end nodes
        n1s=gen_subgraph_edge_index[0]
        n2s=gen_subgraph_edge_index[1]

        #Creating sparse matrix in COO format
        # dimension of the adj matrix changes between inductive and transductive
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))
        # Degree of each node
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        #Adj matrix
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

    class_list =  class_list_train+class_list_valid+class_list_test

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels.numpy().tolist()):
        id_by_class[cla].append(id)

        

    #class_list_train,class_list_valid,class_list_test=json.load(open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class


parser = argparse.ArgumentParser()

parser.add_argument('--use_cuda', action='store_true',default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=123456, help='Random seed.')

parser.add_argument('--train_episodes', type=int, default=1000,
                    help='Number of episodes to train.')
parser.add_argument('--episodes', type=int, default=100,
                    help='Number of episodes to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--test_mode', type=str, default='GPN')



parser.add_argument('--way', type=int, default=10, help='way.')
parser.add_argument('--shot', type=int, default=3, help='shot.')
parser.add_argument('--qry', type=int, help='k shot for query set', default=10)
parser.add_argument('--dataset', default='dblp', help='Dataset:Amazon_clothing/Amazon_eletronics/dblp')
args = parser.parse_args(args=[])
args.cuda = torch.cuda.is_available() and args.use_cuda


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# -------------------------Meta-training------------------------------


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class GPN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x


class GPN_Valuator(nn.Module):
    """
    For the sake of model efficiency, the current implementation is a little bit different from the original paper.
    Note that you can still try different architectures for building the valuator network.

    """

    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Valuator, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.fc3 = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.fc3(x)

        return x



def train(class_selected, id_support, id_query, n_way, k_shot):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot])+1e-9)
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)

    loss_train.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train


def test(class_selected, id_support, id_query, n_way, k_shot):
    encoder.eval()
    scorer.eval()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot])+1e-9)
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = F.nll_loss(output, labels_new)

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler






n_query = args.qry
meta_test_num = 20
meta_valid_num = 20

# Sampling a pool of tasks for validation/testing

from collections import defaultdict

results=defaultdict(dict)


use_contrast=True
use_contrast_distinguish=False
use_label_supervise=True
use_contrast_normal=False
use_whether_label_contrast=True



use_predict_as_emb=False

save_time=datetime.datetime.now()

#names = ['Amazon_clothing', 'Amazon_eletronics', 'dblp']

#names=['ogbn-arxiv']
names = ["Cora"]
#for dataset in ['dblp','Amazon_clothing','Amazon_eletronics']:
for dataset in names:


    #dataset = args.dataset
    adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(dataset, inductive=True)

    if dataset in ["Cora","CiteSeer","Amazon-Computer"]:
        nway = [2]
    else:
        nway = [2,5]
    for n_way in nway:
        for k_shot in [1,3,5]:
            accs=[]
            stds=[]
            for repeat in range(1):
           
                encoder = GPN_Encoder(nfeat=features.shape[1], nhid=args.hidden, dropout=args.dropout)

                scorer = GPN_Valuator(nfeat=features.shape[1],
                                      nhid=args.hidden,
                                      dropout=args.dropout)

                optimizer_encoder = optim.Adam(encoder.parameters()
                                               , lr=args.lr, weight_decay=args.weight_decay)

                optimizer_scorer = optim.Adam(scorer.parameters(),
                                              lr=args.lr, weight_decay=args.weight_decay)

                if args.cuda:
                    encoder.cuda()
                    scorer.cuda()
                    features = features.cuda()
                    adj = adj.cuda()
                    labels = labels.cuda()
                    degrees = degrees.cuda()

                # Train model
                count=0
                best_valid_acc=0
                t_total = time.time()
                meta_train_acc = []
                for episode in range(args.train_episodes):

                    
                    id_support, id_query, class_selected = task_generator(id_by_class, class_list_train, n_way, k_shot, m_query=1,maximum_value_train_each_class=0)
                    acc_train, f1_train = train(class_selected, id_support, id_query, n_way, k_shot)
                    meta_train_acc.append(acc_train)

                    if episode > 0 and episode % 10 == 0:
                        print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))


                        valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query) for i in range(meta_valid_num)]


                        # validation
                        meta_test_acc = []
                        meta_test_f1 = []
                        for idx in range(meta_valid_num):
                            id_support, id_query, class_selected = valid_pool[idx]

                            if args.test_mode!='LR':
                                acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                            else:
                                acc_test, f1_test = LR_test(class_selected, id_support, id_query, n_way, k_shot)
                            meta_test_acc.append(acc_test)
                            meta_test_f1.append(f1_test)

                        valid_acc=np.array(meta_test_acc).mean(axis=0)
                        print("Meta-valid_Accuracy: {}, Meta-valid_F1: {}".format(valid_acc,
                                                                                  np.array(meta_test_f1).mean(axis=0)))

                        if valid_acc>best_valid_acc:
                            best_valid_acc=valid_acc
                            count=0
                        else:
                            count+=1
                            if count>=10:
                                break


                # testing
                print(time.time() - time_start, 'Epoch {}'.format(episode))
                test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, n_query) for i in range(meta_test_num)]
                meta_test_acc = []
                meta_test_f1 = []
                embs=encoder(features, adj)
                nmi=0
                ari=0
                for idx in range(meta_test_num):
                    id_support, id_query, class_selected = test_pool[idx]

                    if args.test_mode!='LR':
                        acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                    else:
                        acc_test, f1_test = LR_test(class_selected, id_support, id_query, n_way, k_shot)
                    meta_test_acc.append(acc_test)
                    meta_test_f1.append(f1_test)

                    if idx==0:
                        print("Task Num: {} Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(idx,np.array(meta_test_acc).mean(axis=0),
                                                                                np.array(meta_test_f1).mean(axis=0)))


                    train_z=embs[id_support].detach().cpu().numpy()
                    test_z=embs[id_query].detach().cpu().numpy()
                    train_y=labels[id_support].detach().cpu().numpy()
                    test_y=labels[id_query].detach().cpu().numpy()
                    kmeans = KMeans(init="k-means++", n_clusters=n_way, random_state=0).fit(
                        np.concatenate([train_z, test_z], 0))
                    y_pred = kmeans.labels_
                    nmi += normalized_mutual_info_score(y_pred, np.concatenate([train_y, test_y], 0)) / meta_test_num
                    ari += adjusted_rand_score(np.concatenate([train_y, test_y], 0), y_pred) / meta_test_num

                print('nmi: {:.4f} ari: {:.4f}'.format(nmi, ari))
                acc=np.mean(meta_test_acc)
                std=np.std(meta_test_acc)

                accs.append(acc)
                stds.append(std)



            results[dataset]['{}-way {}-shot'.format(n_way, k_shot)] = [np.mean(accs), np.mean(stds)]
            results[dataset]['{}-way {}-shot_print'.format(n_way,
                                                           k_shot)] = 'acc: {:.4f}\n std: {:.4f}\n interval: {:.4f}'.format(
                np.mean(accs), np.mean(stds), np.mean(stds) * 0.196)

            json.dump(results,open('./GPN_result.json','w'))


