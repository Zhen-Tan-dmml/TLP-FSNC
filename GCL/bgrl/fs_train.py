import numpy as np

import torch
from torch import optim
# from tensorboardX import SummaryWriter
torch.manual_seed(0)

import models
import utils
import data
from data import download_fs_data, split, test_task_generator

import os
import sys
from config import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import random


class ModelTrainer:

    def __init__(self, args):
        self._args = args
        self._init()
        # self.writer = SummaryWriter(log_dir="runs/BGRL_dataset({})".format(args.name))

    def _init(self):
        args = self._args
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        self._dataset = download_fs_data(config["dataset"])[0]
        print(f"Data: {self._dataset}")
        hidden_layers = [int(l) for l in args.layers]
        layers = [self._dataset.x.shape[1]] + hidden_layers
        self._model = models.BGRL(layer_config=layers, pred_hid=args.pred_hid, dropout=args.dropout, epochs=args.epochs).to(self._device)
        print(self._model)

        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.lr, weight_decay= 1e-5)
        # learning rate
        scheduler = lambda epoch: epoch / 1000 if epoch < 1000 \
                    else ( 1 + np.cos((epoch-1000) * np.pi / (self._args.epochs - 1000))) * 0.5
        self._scheduler = optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda = scheduler)

    def train(self):
        print("start training!")
        patience = config["patience"]
        n_way, k_shot, m_qry = config["n_way"], config["k_shot"], config["m_qry"]
        test_num = config["test_num"]
        labels, train_class, dev_class, test_class, id_by_class = split(self._dataset, config["dataset"])
        cnt_wait = 0
        best_acc = 0
        best_std = 0
        best_t = 0
        # start training
        self._model.train()
        for epoch in range(config["epoch_num"]):
            
            self._dataset.to(self._device)

            augmentation = utils.Augmentation(float(self._args.aug_params[0]),float(self._args.aug_params[1]),float(self._args.aug_params[2]),float(self._args.aug_params[3]))
            view1, view2 = augmentation._feature_masking(self._dataset, self._device)

            v1_output, v2_output, loss = self._model(
                x1=view1.x, x2=view2.x, edge_index_v1=view1.edge_index, edge_index_v2=view2.edge_index,
                edge_weight_v1=view1.edge_attr, edge_weight_v2=view2.edge_attr)
                
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()
            self._model.update_moving_average()
            sys.stdout.write('\rEpoch {}/{}, loss {:.4f}, lr {}'.format(epoch + 1, self._args.epochs, loss.data, self._optimizer.param_groups[0]['lr']))
            sys.stdout.flush()
            
            if epoch % 10 == 0:
                print("")
                print("\nEvaluating {}th epoch..".format(epoch + 1))
                
                self.infer_embeddings()
                final_mean, final_std = self.fs_test(labels, test_num, id_by_class, dev_class, n_way, k_shot, m_qry)

                print("===="*20)
                print("novel_dev_acc: " + str(final_mean))
                print("novel_dev_std: " + str(final_std))
                if best_acc < final_mean:
                    best_acc = final_mean
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(self._model.state_dict(), 'model.pkl')
                else:
                    cnt_wait += 1

            if cnt_wait == patience:
                print('Early stopping!')
                break
        
        # final test
        self._model.load_state_dict(torch.load('model.pkl'))
        final_mean, final_std = self.fs_test(labels, test_num, id_by_class, test_class, n_way, k_shot, m_qry)
        print("****"*20)
        print("novel_test_acc: " + str(final_mean))
        print("novel_test_std: " + str(final_std))
        return final_mean, final_std 
        

    def infer_embeddings(self):
        
        self._model.train(False)
        self._embeddings = self._labels = None

        self._dataset.to(self._device)
        v1_output, v2_output, _ = self._model(
            x1=self._dataset.x, x2=self._dataset.x,
            edge_index_v1=self._dataset.edge_index,
            edge_index_v2=self._dataset.edge_index,
            edge_weight_v1=self._dataset.edge_attr,
            edge_weight_v2=self._dataset.edge_attr)
        emb = v1_output.detach()
        y = self._dataset.y.detach()
        if self._embeddings is None:
            self._embeddings, self._labels = emb, y
        else:
            self._embeddings = torch.cat([self._embeddings, emb])
            self._labels = torch.cat([self._labels, y])
        
    
    def fs_test(self, labels, test_num, id_by_class, test_class, n_way, k_shot, m_qry):
        embeds = self._embeddings.cpu().numpy()
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
        return final_mean, final_std


def train_eval(args):
    trainer = ModelTrainer(args)
    final_mean, final_std = trainer.train()    
    # trainer.writer.close()
    return final_mean, final_std


def main():
    seed = config['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    args = utils.parse_args()

    acc_mean = []
    acc_std = []
    
    for _ in range(5):
        torch.cuda.empty_cache()
        m, s = train_eval(args)
        acc_mean.append(m)
        acc_std.append(s)
    print("Final acc mean: " + str(np.mean(acc_mean)))
    print("Final acc std: " + str(np.mean(acc_std)))


if __name__ == "__main__":
    main()