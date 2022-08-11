from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os

import torch
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from layers import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', default='CoraFull', help='Dataset:Amazon_clothing/reddit/dblp')
parser.add_argument('--pretrain_model', required=False, help='Existing model path.')
parser.add_argument('--overwrite_pretrain', action='store_true', help='Delete existing pre-train model')
parser.add_argument('--output_path', default='./pretrain_model', help='Path for output pre-trained model.')
parser.add_argument('--n_way', type=int, help='n way', default=5)
parser.add_argument('--k_shot', type=int, help='k shot', default=5)
parser.add_argument('--m_qry', type=int, help='m query', default=10)
parser.add_argument('--tune_epoch', type=int, help='test number', default=1000)
parser.add_argument('--test_num', type=int, help='test number', default=100)
parser.add_argument('--patience', type=int, help='epoch patience number', default=10)


args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

test_num = args.test_num
n_way = args.n_way
k_shot = args.k_shot
m_qry = args.m_qry
patience = args.patience

path_tmp = os.path.join(args.output_path, str(args.dataset))
if args.overwrite_pretrain and os.path.exists(path_tmp):
    cmd = "rm -rf " + path_tmp
    os.system(cmd)

if not os.path.exists(path_tmp):
    os.makedirs(path_tmp)

random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = args.dataset
adj, features, labels, degrees, class_list_train, class_list_dev, class_list_test, id_by_class = split(dataset)
pretrain_idx = []
for c, v in id_by_class.items():
    if c in set(class_list_train):
        pretrain_idx.extend(v)

# Model and optimizer
encoder = GNN_Encoder(features.shape[1], args.hidden)

train_classifier = Classifier(nhid=args.hidden, nclass=len(class_list_train))


optimizer_encoder = optim.Adam(encoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

train_optimizer_classifier = optim.Adam(train_classifier.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)



if args.cuda:
    encoder.cuda()
    train_classifier.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    degrees = degrees.cuda()
    train_classifier.cuda()


def pretrain_epoch(pretrain_idx):
    encoder.train()
    train_classifier.train()
    optimizer_encoder.zero_grad()
    train_optimizer_classifier.zero_grad()
    embeddings = encoder(features, adj)
    output = train_classifier(embeddings)[pretrain_idx]
    output = F.log_softmax(-output, dim=1)

    labels_new = torch.LongTensor([class_list_train.index(i) for i in labels[pretrain_idx]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)
    loss_train.backward()
    optimizer_encoder.step()
    train_optimizer_classifier.step()
    return 


def pretest_epoch(pretest_idx):
    encoder.eval()
    train_classifier.eval()
    embeddings = encoder(features, adj)
    output = train_classifier(embeddings)[pretest_idx]
    output = F.log_softmax(-output, dim=1)

    labels_new = torch.LongTensor([class_list_train.index(i) for i in labels[pretest_idx]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = F.nll_loss(output, labels_new)

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test



def fs_test(test_num, test_class):
    encoder.eval()
    z = encoder(features, adj)
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

        train_y = np.array([test_class_selected.index(i) for i in torch.squeeze(labels)[test_id_support]])
        test_y = np.array([test_class_selected.index(i) for i in torch.squeeze(labels)[test_id_query]])

        clf = LogisticRegression(solver='lbfgs', max_iter=1000,
                                     multi_class='auto').fit(train_z, train_y)

        test_acc = clf.score(test_z, test_y)
        test_acc_all.append(test_acc)

    final_mean = np.mean(test_acc_all)
    final_std = np.std(test_acc_all)
    final_interval = 1.96 * (final_std / np.sqrt(len(test_acc_all)))

    return final_mean, final_std, final_interval


def train_eval():

    # Train model
    t_total = time.time()

    best_dev_acc = 0.
    tolerate = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        pretrain_epoch(pretrain_idx)
        if (epoch - 1) % 10 == 0:
            print("-------Epochs {}-------".format(epoch))
            # validation
            dev_acc, dev_std, dev_interval = fs_test(test_num, class_list_dev)

            print("===="*20)
            print("dev_acc: " + str(dev_acc))
            print("dev_std: " + str(dev_std))
            print("dev_interval: " + str(dev_interval))
            
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                tolerate = 0
                best_epoch = epoch
                torch.save(encoder.state_dict(), 'fs_sup.pkl')
            else:
                tolerate += 1

        if tolerate > args.patience:
            print("Pretraining finished at epoch: " + str(epoch))
            print("Best pretrain epoch: " + str(best_epoch))
            break

    # testing
    encoder.load_state_dict(torch.load('fs_sup.pkl'))
    final_mean, final_std, final_interval = fs_test(test_num, class_list_test)
    print("novel_test_acc: " + str(final_mean))
    print("novel_test_std: " + str(final_std))
    print("novel_test_interval: " + str(final_interval))

    return final_mean, final_std, final_interval

if __name__ == '__main__':

    acc_mean = []
    acc_std = []
    acc_interval = []
    for __ in range(5):
        m, s, interval = train_eval()
        acc_mean.append(m)
        acc_std.append(s)
        acc_interval.append(interval)
    print("======"*10)
    print("Final acc: " + str(np.mean(acc_mean)))
    print("Final acc std: " + str(np.mean(acc_std)))
    print("Final acc interval: " + str(np.mean(acc_interval)))

