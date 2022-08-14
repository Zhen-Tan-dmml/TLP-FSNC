import argparse
import os.path as osp
import random
from time import perf_counter as t

import numpy as np
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from model import Encoder, Model, drop_feature
from eval import label_classification
from data_split import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from setting import setting
from copy import deepcopy
from math import sqrt
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+1e-9)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = torch.mean(loss.view(anchor_count, batch_size))

        return loss

def relabeling(labels, train_class, dev_class, test_class, id_by_class):
    #print("Start relabeling...")
    #labels = torch.argmax(labels[0], dim=1)
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
    #("Relabeling finished!")
    return contrast_labels

def train(model: Model, x, contrast_labels,edge_index, optimizer):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)


    #value_lambda=0.2
    loss = model.loss(z1, z2, batch_size=0)
    loss_s=loss_sup(torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1), contrast_labels )
    loss=loss*value_lambda+loss_s*(1-value_lambda)
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

    nmi=0
    ari=0
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

        kmeans = KMeans(init="k-means++", n_clusters=n_way, random_state=0).fit(np.concatenate([train_z,test_z],0))
        y_pred=kmeans.labels_
        nmi+=normalized_mutual_info_score(y_pred, np.concatenate([train_y,test_y],0))/test_num
        ari+=adjusted_rand_score( np.concatenate([train_y,test_y],0),y_pred)/test_num



    final_mean = np.mean(test_acc_all)
    final_std = np.std(test_acc_all)

    return final_mean, final_std, nmi, ari


def train_eval():
    dataset, train_idx, id_by_class, train_class, dev_class, test_class, degree_inv = split(args.dataset)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    contrast_labels = relabeling(data.y, train_class, dev_class, test_class, id_by_class)
    contrast_labels = torch.LongTensor(contrast_labels)
    contrast_labels = contrast_labels.cuda()


    encoder = Encoder(dataset.num_features, num_hidden, activation,
                    base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    cnt_wait = 0
    best_acc = 0
    from tqdm import tqdm
    for epoch in tqdm(range(1, num_epochs + 1)):
        _ = train(model, data.x, contrast_labels, data.edge_index, optimizer)
        if (epoch - 1) % 10:
            final_mean, final_std, _, _ = fs_test(model, data.x, data.edge_index, data.y, setting['test_num'], id_by_class, dev_class, setting['n_way'], setting['k_shot'], setting['m_qry'])
            #print("===="*20)
            #print("novel_dev_acc: " + str(final_mean))
            #print("novel_dev_std: " + str(final_std))
            if best_acc < final_mean:
                best_acc = final_mean
                cnt_wait = 0
                torch.save(model.state_dict(), 'model.pkl')
            else:
                cnt_wait += 1

        if cnt_wait == setting['patience']:
            #print('Early stopping!')
            break


    #print("=== Final Test ===")
    model.load_state_dict(torch.load('model.pkl'))
    final_mean, final_std, nmi, ari = fs_test(model, data.x, data.edge_index, data.y, setting['test_num'], id_by_class, test_class, setting['n_way'], setting['k_shot'], setting['m_qry'])
    #print("novel_test_acc: " + str(final_mean))
    #print("novel_test_std: " + str(final_std))

    return final_mean, final_std, nmi, ari


if __name__ == '__main__':
    for value_lambda in [0.,1.0]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='CoraFull')
        parser.add_argument('--gpu_id', type=int, default=0)
        parser.add_argument('--config', type=str, default='fs_config.yaml')
        args = parser.parse_args()


        loss_sup=SupConLoss()



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
        mnis=[]
        aris=[]
        for __ in range(1):
            m, s, mni, ari = train_eval()
            acc_mean.append(m)
            acc_std.append(s)
            mnis.append(mni)
            aris.append(ari)
        print("======"*10)
        print('value: lambda',value_lambda)
        print("acc: {:.6f}" .format(np.mean(acc_mean)))
        print("std: {:.6f}".format(np.mean(acc_std)))
        print("interval: {:.6f}".format(1.96 * (np.mean(acc_std) / np.sqrt(100))))
        print("mni: {:.6f}" .format(np.mean(mni)))
        print("ari: {:.6f}".format(np.mean(ari)))
