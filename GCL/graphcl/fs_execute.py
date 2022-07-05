import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
from models import DGI, LogReg
from utils import process
import pdb
import aug
import os
import argparse
from config import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from os.path import join
import pickle5 as pickle
from loss import *


parser = argparse.ArgumentParser("My DGI")

# parser.add_argument('--dataset',          type=str,           default="CoraFull",        help='data')
parser.add_argument('--aug_type',         type=str,           default="node",            help='aug type: mask, subgraph or edge')
parser.add_argument('--drop_percent',     type=float,         default=0.1,               help='drop percent')
# parser.add_argument('--seed',             type=int,           default=39,                help='seed')
parser.add_argument('--gpu',              type=int,           default=0,                 help='gpu')

args = parser.parse_args()

# print('-' * 100)
# print(args)
# print('-' * 100)

dataset = config["dataset"]
aug_type = args.aug_type
drop_percent = args.drop_percent
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# training params


batch_size = 1
nb_epochs = 10000
patience = config["patience"]
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = True


nonlinearity = 'prelu' # special name to separate parameters
# adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
# features, _ = process.preprocess_features(features)

path = join("./fs_data", config["dataset"])


def load_object(filename):
    with open(filename, 'rb') as fin:
        obj = pickle.load(fin)
    return obj

adj = load_object(join(path, "adj.pk"))
features = load_object(join(path, "features.pk"))
labels = load_object(join(path, "labels.pk"))
train_class = load_object(join(path, "train_class.pk"))
dev_class = load_object(join(path, "dev_class.pk"))
test_class = load_object(join(path, "test_class.pk"))
id_by_class = load_object(join(path, "id_by_class.pk"))
train_idx = []
for cla in train_class:
    train_idx.extend(id_by_class[cla])

nb_nodes = features.shape[0]  # node number
ft_size = features.shape[1]   # node features dim
nb_classes = labels.shape[1]  # classes = 6

features = torch.FloatTensor(features[np.newaxis])


'''
------------------------------------------------------------
edge node mask subgraph
------------------------------------------------------------
'''
print("Begin Aug:[{}]".format(args.aug_type))
if args.aug_type == 'edge':

    aug_features1 = features
    aug_features2 = features

    aug_adj1 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
    aug_adj2 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
    
elif args.aug_type == 'node':
    
    aug_features1, aug_adj1 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)
    aug_features2, aug_adj2 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)
    
elif args.aug_type == 'subgraph':
    
    aug_features1, aug_adj1 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)
    aug_features2, aug_adj2 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)

elif args.aug_type == 'mask':

    aug_features1 = aug.aug_random_mask(features,  drop_percent=drop_percent)
    aug_features2 = aug.aug_random_mask(features,  drop_percent=drop_percent)
    
    aug_adj1 = adj
    aug_adj2 = adj

else:
    assert False



'''
------------------------------------------------------------
'''

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    sp_aug_adj1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1)
    sp_aug_adj2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2)

else:
    adj = (adj + sp.eye(adj.shape[0])).todense()
    aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
    aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()


'''
------------------------------------------------------------
mask
------------------------------------------------------------
'''

'''
------------------------------------------------------------
'''
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
    aug_adj1 = torch.FloatTensor(aug_adj1[np.newaxis])
    aug_adj2 = torch.FloatTensor(aug_adj2[np.newaxis])


labels = torch.FloatTensor(labels[np.newaxis])
# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

model = DGI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    aug_features1 = aug_features1.cuda()
    aug_features2 = aug_features2.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
        sp_aug_adj1 = sp_aug_adj1.cuda()
        sp_aug_adj2 = sp_aug_adj2.cuda()
    else:
        adj = adj.cuda()
        aug_adj1 = aug_adj1.cuda()
        aug_adj2 = aug_adj2.cuda()

    labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()


def fs_test(model, features, adj, sp_adj, labels, test_num, id_by_class, test_class, n_way, k_shot, m_qry, sparse=True):
    model.eval()
    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
    embeds = embeds[0].detach().cpu().numpy()
    scaler = MinMaxScaler()
    scaler.fit(embeds)
    embeds = scaler.transform(embeds)
    labels = torch.argmax(labels[0], dim=1)


    test_acc_all = []
    for i in range(test_num):
        test_id_support, test_id_query, test_class_selected = \
            test_task_generator(id_by_class, test_class, n_way, k_shot, m_qry)

        train_z = embeds[test_id_support]
        test_z = embeds[test_id_query]

        train_y = np.array([test_class_selected.index(i) for i in torch.squeeze(labels)[test_id_support]])
        test_y = np.array([test_class_selected.index(i) for i in torch.squeeze(labels)[test_id_query]])
        clf = LogisticRegression(solver='lbfgs', max_iter=1000,
                                     multi_class='auto').fit(train_z, train_y)

        test_acc = clf.score(test_z, test_y)
        test_acc_all.append(test_acc)

    final_mean = np.mean(test_acc_all)
    final_std = np.std(test_acc_all)
    return final_mean, final_std

if config["sup"] == "unsup":
    b_xent = nn.BCEWithLogitsLoss()
elif config["sup"] == "sup":
    contrast_labels = relabeling(labels, train_class, dev_class, test_class, id_by_class)
    contrast_labels = torch.LongTensor(contrast_labels)
    contrast_labels = contrast_labels.cuda()
    supcon = SupConLoss()

n_way, k_shot, m_qry = config["n_way"], config["k_shot"], config["m_qry"]
test_num = config["test_num"]

best_acc = 0
best_std = 0
best_t = 0
cnt_wait = 0

for epoch in range(config["epoch_num"]):

    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    if config["sup"] == "unsup":
        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        if config["sup"] == "unsup":
            lbl = lbl.cuda()
    if config["sup"] == "unsup":
        logits, _, _ = model(features, shuf_fts, aug_features1, aug_features2,
                   sp_adj if sparse else adj, 
                   sp_aug_adj1 if sparse else aug_adj1,
                   sp_aug_adj2 if sparse else aug_adj2,  
                   sparse, None, None, None, aug_type=aug_type) 
        loss = b_xent(logits, lbl)
    
    elif config["sup"] == "sup":
        logits, h0, h2 = model(features, shuf_fts, aug_features1, aug_features2,
                   sp_adj if sparse else adj, 
                   sp_aug_adj1 if sparse else aug_adj1,
                   sp_aug_adj2 if sparse else aug_adj2,  
                   sparse, None, None, None, aug_type=aug_type)
        contrast_features = torch.cat([h0.unsqueeze(1), h2.unsqueeze(1)], dim=1)
        loss = supcon(contrast_features, contrast_labels)
    print('Loss:[{:.4f}]'.format(loss.item()))
    loss.backward()
    optimiser.step()

    # validation
    if epoch % 10:
        final_mean, final_std = fs_test(model, features, adj, sp_adj, labels, test_num, id_by_class, dev_class, n_way, k_shot, m_qry, sparse)
        print("===="*20)
        print("novel_dev_acc: " + str(final_mean))
        print("novel_dev_std: " + str(final_std))
        if best_acc < final_mean:
            best_acc = final_mean
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break


print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('model.pkl'))

#final test
final_mean, final_std = fs_test(model, features, adj, sp_adj, labels, test_num, id_by_class, test_class, n_way, k_shot, m_qry, sparse)
print("****"*20)
print("novel_test_acc: " + str(final_mean))
print("novel_test_std: " + str(final_std))


