import random
import time
import torch
from torch_geometric.utils import to_undirected
import os.path as osp
import torch.nn.functional as F
import torch.nn as nn
from colorama import  Fore
import numpy as np
from time import perf_counter as t
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.datasets import Planetoid, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.sparse import coo_matrix
from data_unit.utils import blind_other_gpus, row_normalize, sparse_mx_to_torch_sparse_tensor,normalize_graph
from models import LogReg, SUGRL_Fast
from torch_geometric.utils import degree
import os
import argparse
from ruamel.yaml import YAML
from termcolor import cprint
from config import *
from sklearn.linear_model import LogisticRegression


def get_args_key(args):
    return "-".join([args.model_name, args.dataset_name, args.custom_key])

def get_args(model_name, dataset_class, dataset_name, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser(description='Parser for Simple Unsupervised Graph Representation Learning')
    # Basics
    parser.add_argument("--num-gpus-total", default=0, type=int)
    parser.add_argument("--num-gpus-to-use", default=0, type=int)
    parser.add_argument("--black-list", default=None, type=int, nargs="+")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--model_name", default=model_name)
    parser.add_argument("--custom_key", default=custom_key)
    parser.add_argument("--save_model", default=True)
    parser.add_argument("--seed", default=1234)
    # Dataset
    parser.add_argument('--data-root', default="~/graph-data", metavar='DIR', help='path to dataset')
    parser.add_argument("--dataset-class", default=dataset_class)
    parser.add_argument("--dataset-name", default=dataset_name)
    # Pretrain
    parser.add_argument("--pretrain", default=True, type=bool)
    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr2', '--learning-rate2', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate2', dest='lr2')
    parser.add_argument("--use-bn", default=False, type=bool)
    parser.add_argument('--w_loss1', type=float, default=1, help='')
    parser.add_argument('--w_loss2', type=float, default=1, help='')
    parser.add_argument('--w_loss3', type=float, default=1, help='')
    parser.add_argument('--margin1', type=float, default=0.8, help='')
    parser.add_argument('--margin2', type=float, default=0.2, help='')
    # Experiment specific parameters loaded from .yamls
    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset_name or args.dataset_class, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")
    # Update params from .yamls
    args = parser.parse_args()
    return args

def pprint_args(_args: argparse.Namespace):
    cprint("Args PPRINT: {}".format(get_args_key(_args)), "yellow")
    for k, v in sorted(_args.__dict__.items()):
        print("\t- {}: {}".format(k, v))

def get_dataset(args, dataset_kwargs):
    if args.dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers']:
        if args.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '../dataset')
            # dataset = dataset_cls(root=root, **kwargs)
            dataset = Planetoid(path, args.dataset_name)
        elif args.dataset_name in ['Photo', 'Computers']:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '../dataset')
            dataset = Amazon(path, args.dataset_name, pre_transform=None)  # transform=T.ToSparseTensor(),
        data = dataset[0]
        # data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        i = torch.LongTensor([data.edge_index[0].numpy(), data.edge_index[1].numpy()])
        v = torch.FloatTensor(torch.ones([data.num_edges]))
        A_sp = torch.sparse.FloatTensor(i, v, torch.Size([data.num_nodes, data.num_nodes]))
        A = A_sp.to_dense()
        I = torch.eye(A.shape[1]).to(A.device)
        A_I = A + I
        A_I_nomal = normalize_graph(A_I)
        A_I_nomal = A_I_nomal.to_sparse()

        lable = data.y
        nb_feature = data.num_features
        nb_classes = int(lable.max() - lable.min()) + 1
        nb_nodes = data.num_nodes
        data.x = torch.FloatTensor(data.x)
        eps = 2.2204e-16
        norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
        data.x = data.x.div(norm.expand_as(data.x))
    else:
        if args.dataset_name in ['ogbn-arxiv', 'ogbn-proteins']:  # 'ogbn-products',, 'ogbn-mag'
            dataset = PygNodePropPredDataset(root='../dataset/' + args.dataset_name, name=args.dataset_name)  # ,transform=T.ToSparseTensor()
            data = dataset[0]
            data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        elif args.dataset_name in ['ogbn-mag', 'ogbn-products']:
            dataset = PygNodePropPredDataset(name=args.dataset_name)
            split_idx = dataset.get_idx_split()
            if args.dataset_name in ['ogbn-mag']:
                rel_data = dataset[0]
                # We are only interested in paper <-> paper relations.
                data = Data(
                    x=rel_data.x_dict['paper'],
                    edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                    y=rel_data.y_dict['paper'])
                data.edge_index = to_undirected(data.edge_index, data.num_nodes)
            else:
                rel_data = dataset[0]
                data = Data(
                    x=rel_data.x,
                    edge_index=rel_data.edge_index,
                    y=rel_data.y)

        # data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        data.x = torch.FloatTensor(data.x)
        eps = 2.2204e-16
        norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
        data.x = data.x.div(norm.expand_as(data.x))
        adj = coo_matrix(
            (np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
            shape=(data.num_nodes, data.num_nodes))
        nb_nodes = data.num_nodes
        I = coo_matrix((np.ones(nb_nodes), (np.arange(0, nb_nodes, 1), np.arange(0, nb_nodes, 1))),
                       shape=(nb_nodes, nb_nodes))
        adj_I = adj + I  # coo_matrix(sp.eye(adj.shape[0]))
        adj_I = row_normalize(adj_I)
        A_I_nomal = sparse_mx_to_torch_sparse_tensor(adj_I).float()
        lable = data.y
        nb_feature = data.num_features
        nb_classes = int(lable.max() - lable.min()) + 1
    
    class_list = [i for i in range(nb_classes)]
    train_num = class_split[args.dataset_name]["train"]
    dev_num = class_split[args.dataset_name]["dev"]
    test_num = class_split[args.dataset_name]["test"]
    random.shuffle(class_list)
    train_class = class_list[: train_num]
    dev_class = class_list[train_num : train_num + dev_num]
    test_class = class_list[train_num + dev_num :]
    print("train_num: {}; dev_num: {}; test_num: {}".format(train_num, dev_num, test_num))

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(torch.squeeze(data.y).tolist()):
        id_by_class[cla].append(id)

    train_idx = []
    for cla in train_class:
        train_idx.extend(id_by_class[cla])

    return data, [A_I_nomal], [data.x], [lable, nb_feature, nb_classes, nb_nodes], id_by_class, train_class, dev_class, test_class


def fs_test(z, y, test_num, id_by_class, test_class, n_way, k_shot, m_qry):

    z = z.detach().cpu().numpy()
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


def run_SUGRL(args, gpu_id=None, **kwargs):
    # ===================================================#
    final_acc = 0
    best_acc = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    running_device = "cpu" if gpu_id is None \
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ===================================================#
    cprint("## Loading Dataset ##", "yellow")
    dataset_kwargs = {}
    data, adj_list, x_list, nb_list, id_by_class, train_class, dev_class, test_class \
     = get_dataset(args, dataset_kwargs)
    lable = nb_list[0]
    nb_feature = nb_list[1]
    nb_classes = nb_list[2]
    nb_nodes = nb_list[3]
    feature_X = x_list[0].to(running_device)
    A_I_nomal = adj_list[0].to(running_device)
    cprint("## Done ##", "yellow")
    # ===================================================#
    model = SUGRL_Fast(nb_feature, cfg=args.cfg,
                       dropout=args.dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(running_device)
    lable = lable.to(running_device)
    # ===================================================#
    if args.dataset_name == 'WikiCS':
        train_lbls = lable[data.train_mask[:, args.NewATop]]  # capture
        test_lbls = lable[data.test_mask]
    elif args.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        train_lbls = lable[data.train_mask]
        test_lbls = lable[data.test_mask]
    elif args.dataset_name in ['Photo', 'Computers', 'DBLP', 'Crocodile', 'CoraFull']:
        train_index = []
        test_index = []
        for j in range(lable.max().item() + 1):
            # num = ((lable == j) + 0).sum().item()
            index = torch.range(0, len(lable) - 1)[(lable == j).squeeze()]
            x_list0 = random.sample(list(index), int(len(index) * 0.1))
            for x in x_list0:
                train_index.append(int(x))
        for c in range(len(lable)):
            if int(c) not in train_index:
                test_index.append(int(c))
        train_lbls = lable[train_index].squeeze()
        test_lbls = lable[test_index]
        val_lbls = lable[train_index]
    elif args.dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-mag']:
        number_node = lable.size()[0]
        random_split = np.random.permutation(number_node)
        train_index = random_split[:int(number_node * 0.1)]
        test_index = random_split[int(number_node * 0.1):]
        train_lbls = lable[train_index].squeeze()
        test_lbls = lable[test_index].squeeze()
        feature_X = F.normalize(feature_X)
        feature_X = torch.spmm(A_I_nomal, feature_X)

    A_degree = degree(A_I_nomal._indices()[0], nb_nodes, dtype=int).tolist()
    if args.dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag']:
        edge_index = A_I_nomal._indices()[0]
    else:
        edge_index = A_I_nomal._indices()[1]
    # ===================================================#
    my_margin = args.margin1
    my_margin_2 = my_margin + args.margin2
    margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
    num_neg = args.NN
    lbl_z = torch.tensor([0.]).to(running_device)
    deg_list_2 = []
    deg_list_2.append(0)
    for i in range(nb_nodes):
        deg_list_2.append(deg_list_2[-1] + A_degree[i])
    idx_p_list = []
    for j in range(1, 101):
        random_list = [deg_list_2[i] + j % A_degree[i] for i in range(nb_nodes)]
        idx_p = edge_index[random_list]
        idx_p_list.append(idx_p)

    cnt_wait = 0
    best_acc = 0
    test_num = config['test_num']
    n_way = config['n_way']
    k_shot = config['k_shot']
    m_qry = config['m_qry']
    for current_iter, epoch in enumerate(tqdm(range(0, config['epoch_num']))):
        model.train()
        optimiser.zero_grad()
        idx_list = []
        for i in range(num_neg):
            idx_0 = np.random.permutation(nb_nodes)
            idx_list.append(idx_0)

        h_a, h_p = model(feature_X, A_I_nomal)

        h_p_1 = (h_a[idx_p_list[epoch % 100]] + h_a[idx_p_list[(epoch + 2) % 100]] + h_a[
            idx_p_list[(epoch + 4) % 100]] + h_a[idx_p_list[(epoch + 6) % 100]] + h_a[
                     idx_p_list[(epoch + 8) % 100]]) / 5
        s_p = F.pairwise_distance(h_a, h_p)
        s_p_1 = F.pairwise_distance(h_a, h_p_1)
        s_n_list = []
        for h_n in idx_list:
            s_n = F.pairwise_distance(h_a, h_a[h_n])
            s_n_list.append(s_n)
        margin_label = -1 * torch.ones_like(s_p)

        loss_mar = 0
        loss_mar_1 = 0
        mask_margin_N = 0
        for s_n in s_n_list:
            loss_mar += (margin_loss(s_p, s_n, margin_label)).mean()
            loss_mar_1 += (margin_loss(s_p_1, s_n, margin_label)).mean()
            mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).mean()

        loss = loss_mar * args.w_loss1 + loss_mar_1 * args.w_loss2 + mask_margin_N * args.w_loss3
        loss.backward()
        optimiser.step()
        string_1 = " loss_1: {:.3f}||loss_2: {:.3f}||loss_3: {:.3f}||".format(loss_mar.item(), loss_mar_1.item(),
                                                                              mask_margin_N.item())
        if args.pretrain:
            if os.path.exists(args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth'):
                    load_params = torch.load(args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth', map_location='cuda')
                    model_params = model.state_dict()
                    same_parsms = {k: v for k, v in load_params.items() if k in model_params.keys()}
                    model_params.update(same_parsms)
                    model.load_state_dict(model_params)
        if args.save_model:
            torch.save(model.state_dict(), args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth')
        if epoch % 10 == 0 and epoch != 0:
            model.eval()
            h_a, h_p = model.embed(feature_X, A_I_nomal)
            embs = h_p
            embs = embs / embs.norm(dim=1)[:, None]
            final_mean, final_std = fs_test(embs, lable, test_num, id_by_class, dev_class, n_way, k_shot, m_qry)
            print("===="*20)
            print("novel_dev_acc: " + str(final_mean))
            print("novel_dev_std: " + str(final_std))
            if best_acc < final_mean:
                best_acc = final_mean
                cnt_wait = 0
                torch.save(model.state_dict(), 'model.pkl')
            else:
                cnt_wait += 1

        if cnt_wait == config['patience']:
            print('Early stopping!')
            break

    print("=== Final Test ===")
    model.load_state_dict(torch.load('model.pkl'))
    final_mean, final_std = fs_test(embs, lable, test_num, id_by_class, test_class, n_way, k_shot, m_qry)
    print("novel_test_acc: " + str(final_mean))
    print("novel_test_std: " + str(final_std))

    return final_mean, final_std



if __name__ == '__main__':

    # num_total_runs = 5
    main_args = get_args(
        model_name="SUGRL",  # GCN SUnGRL
        dataset_class="PygNodePropPredDataset",
        # PygNodePropPredDataset
        dataset_name="ogbn-arxiv",  # ogbn-arxiv, ogbn-mag, ogbn-products
        custom_key="classification",  # classification, link, clu
    )
    ### Dataset (`--dataset-class`, `--dataset-name`,`--Custom-key`)
    # | Dataset class          | Dataset name | Custom key    |
    # | Planetoid              | Cora         | classification|
    # | Planetoid              | CiteSeer     | classification|
    # | Planetoid              | PubMed       | classification|
    # | MyAmazon               | Photo        | classification|
    # | MyAmazon               | Computers    | classification|
    # | PygNodePropPredDataset | ogbn-arxiv   | classification|
    # | PygNodePropPredDataset | ogbn-mag     | classification|
    # | PygNodePropPredDataset | ogbn-products| classification|
    pprint_args(main_args)

    if len(main_args.black_list) == main_args.num_gpus_total:
        alloc_gpu = [None]
        cprint("Use CPU", "yellow")
    else:
        alloc_gpu = blind_other_gpus(num_gpus_total=main_args.num_gpus_total,
                                     num_gpus_to_use=main_args.num_gpus_to_use,
                                     black_list=main_args.black_list)
        if not alloc_gpu:
            alloc_gpu = [int(np.random.choice([g for g in range(main_args.num_gpus_total)
                                               if g not in main_args.black_list], 1))]
        cprint("Use GPU the ID of which is {}".format(alloc_gpu), "yellow")

    # ===================================================#
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    acc_mean = []
    acc_std = []
    for __ in range(5):
        m, s = run_SUGRL(main_args, gpu_id=alloc_gpu[0])
        acc_mean.append(m)
        acc_std.append(s)
    cprint("======"*10)
    cprint("Final acc mean: " + str(np.mean(acc_mean)), "yellow")
    cprint("Final acc std: " + str(np.mean(acc_std)), "yellow")

