import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification
from data_split import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from setting import setting


def train(model: Model, x, edge_index, optimizer):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)


def fs_test(model, x, edge_index, y, test_num, id_by_class, test_class, n_way, k_shot, m_qry):
    model.eval()
    z = model(x, edge_index)
    z = z.detach().cpu().numpy()
    scaler = MinMaxScaler()
    scaler.fit(z)
    z = scaler.transform(z)

    test_acc_all = []
    for i in range(test_num):
        test_id_support, test_id_query, test_class_selected = \
            test_task_generator(id_by_class, test_class, n_way, k_shot, m_qry)

        train_z = z[test_id_support]
        test_z = z[test_id_query]

        train_y = np.array([test_class_selected.index(i) for i in torch.squeeze(y)[test_id_support]])
        test_y = np.array([test_class_selected.index(i) for i in torch.squeeze(y)[test_id_query]])

        clf = LogisticRegression(solver='lbfgs', max_iter=1000,
                                     multi_class='auto').fit(train_z, train_y)

        test_acc = clf.score(test_z, test_y)
        test_acc_all.append(test_acc)

    final_mean = np.mean(test_acc_all)
    final_std = np.std(test_acc_all)

    return final_mean, final_std


def train_eval():
    dataset, train_idx, id_by_class, train_class, dev_class, test_class, degree_inv = split(args.dataset)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                    base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    cnt_wait = 0
    best_acc = 0
    for epoch in range(1, num_epochs + 1):
        _ = train(model, data.x, data.edge_index, optimizer)
        if (epoch - 1) % 10:
            final_mean, final_std = fs_test(model, data.x, data.edge_index, data.y, setting['test_num'], id_by_class, dev_class, setting['n_way'], setting['k_shot'], setting['m_qry'])
            print("===="*20)
            print("novel_dev_acc: " + str(final_mean))
            print("novel_dev_std: " + str(final_std))
            if best_acc < final_mean:
                best_acc = final_mean
                cnt_wait = 0
                torch.save(model.state_dict(), 'model.pkl')
            else:
                cnt_wait += 1

        if cnt_wait == setting['patience']:
            print('Early stopping!')
            break


    print("=== Final Test ===")
    model.load_state_dict(torch.load('model.pkl'))
    final_mean, final_std = fs_test(model, data.x, data.edge_index, data.y, setting['test_num'], id_by_class, test_class, setting['n_way'], setting['k_shot'], setting['m_qry'])
    print("novel_test_acc: " + str(final_mean))
    print("novel_test_std: " + str(final_std))

    return final_mean, final_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='fs_config.yaml')
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(1234)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    acc_mean = []
    acc_std = []
    for __ in range(5):
        m, s = train_eval()
        acc_mean.append(m)
        acc_std.append(s)
    print("======"*10)
    print("Final acc mean: " + str(np.mean(acc_mean)))
    print("Final acc std: " + str(np.mean(acc_std)))
    
