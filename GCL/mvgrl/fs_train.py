from distutils.command import config
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from utils import sparse_mx_to_torch_sparse_tensor
from dataset import load, test_task_generator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from config import *
import random
from loss import *


# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


# Borrowed from https://github.com/PetarV-/DGI
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)


# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = Readout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)
        
        # print(c_1.size())
        # print(c_2.size())
        # print(h_1.size())
        # print(h_2.size())
        # print(h_3.size())
        # print(h_4.size())

        return ret, nn.functional.normalize((c_1).squeeze()), nn.functional.normalize((c_2).squeeze())

    def embed(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
        return (h_1 + h_2).detach(), c.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


def distribution_calibration(query, base_means, base_cov, k,alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

    return calibrated_mean, calibrated_cov


def fs_test(model, features, adj, diff, labels, test_num, id_by_class, test_class, n_way, k_shot, m_qry, sparse=False):
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
        diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    # features = features.cuda()
    # adj = adj.cuda()
    # diff = diff.cuda()

    model.eval()
    embeds, _ = model.embed(features, adj, diff, sparse, None)
    embeds = embeds[0].detach().cpu().numpy()
    scaler = MinMaxScaler()
    scaler.fit(embeds)
    embeds = scaler.transform(embeds)


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
    final_interval = 1.96 * (final_std / np.sqrt(len(test_acc_all)))
    return final_mean, final_std, final_interval


def train(dataset, verbose=False):

    nb_epochs =  config["epoch_num"]
    patience = config["patience"]
    lr = 0.001
    l2_coef = 0.0
    hid_units = 512
    sparse = False
    n_way, k_shot, m_qry = config["n_way"], config["k_shot"], config["m_qry"]
    test_num = config["test_num"]

    adj, diff, features, labels, train_class, dev_class, test_class, id_by_class = load(dataset)

    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    sample_size = 2000
    batch_size = 4

    labels = torch.LongTensor(labels)

    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    model = Model(ft_size, hid_units)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    # if torch.cuda.is_available():
    #     model.cuda()
    #     labels = labels.cuda()
    #     if config["sup"] == "unsup":
    #         lbl = lbl.cuda()

    if config["sup"] == "unsup":
        b_xent = nn.BCEWithLogitsLoss()
    elif config["sup"] == "sup":
        contrast_labels = relabeling(labels, train_class, dev_class, test_class, id_by_class)
        contrast_labels = torch.LongTensor(contrast_labels)
        # contrast_labels = contrast_labels.cuda()
        supcon = SupConLoss()
        b_xent = nn.BCEWithLogitsLoss()
    
    cnt_wait = 0
    best_acc = 0
    best_t = 0

    for epoch in range(nb_epochs):

        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        contrast_labels_batch = contrast_labels[idx]
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)

        bf = torch.FloatTensor(bf)
        idx = np.random.permutation(sample_size)
        shuf_fts = bf[:, idx, :]

        # if torch.cuda.is_available():
        #     bf = bf.cuda()
        #     ba = ba.cuda()
        #     bd = bd.cuda()
        #     shuf_fts = shuf_fts.cuda()

        model.train()
        optimiser.zero_grad()

        logits, n_0, n_1 = model(bf, shuf_fts, ba, bd, sparse, None, None, None)

        if config["sup"] == "unsup":
            loss = b_xent(logits, lbl)
        elif config["sup"] == "sup": 
            contrast_features = torch.cat([n_0.unsqueeze(1), n_1.unsqueeze(1)], dim=1)
            contrast_labels_batch = torch.LongTensor(batch_relabeling(contrast_labels_batch))
            sup_loss = supcon(contrast_features, contrast_labels_batch)
            unsup_loss = b_xent(logits, lbl)
            lmd = 0.1
            loss = loss = lmd * unsup_loss + (1 - lmd) * sup_loss

        loss.backward()
        optimiser.step()

        if verbose:
            print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        # validation
        if epoch % 10:
            final_mean, final_std, final_interval = fs_test(model, features, adj, diff, labels, test_num, id_by_class, dev_class, n_way, k_shot, m_qry, sparse)
            print("===="*20)
            print("novel_dev_acc: " + str(final_mean))
            print("novel_dev_std: " + str(final_std))
            print("novel_dev_interval: " + str(final_interval))
            if best_acc < final_mean:
                best_acc = final_mean
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'model.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                if verbose:
                    print('Early stopping!')
                break

    if verbose:
        print('Loading {}th epoch'.format(best_t))

    # final test
    model.load_state_dict(torch.load('model.pkl'))
    final_mean, final_std, final_interval = fs_test(model, features, adj, diff, labels, test_num, id_by_class, test_class, n_way, k_shot, m_qry, sparse)
    print("****"*20)
    print("novel_test_acc: " + str(final_mean))
    print("novel_test_std: " + str(final_std))
    print("novel_test_interval: " + str(final_interval))
    return final_mean, final_std, final_interval


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    # 'cora', 'citeseer', 'pubmed'
    dataset = config['dataset']
    seed = config['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    acc_mean = []
    acc_std = []
    acc_interval = []
    for __ in range(5):
        m, s, iterval = train(dataset)
        acc_mean.append(m)
        acc_std.append(s)
        acc_interval.append(iterval)
    print("Final acc: " + str(np.mean(acc_mean)))
    print("Final acc std: " + str(np.mean(acc_std)))
    print("Final acc interval: " + str(np.mean(acc_interval)))

