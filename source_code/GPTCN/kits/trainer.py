import os
import torch
from torch import nn
import numpy as np
import time
import copy
from kits.loss import masked_softmax_cross_entropy_loss
from visdom import Visdom
import torchmetrics

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from kits.appDataset import AppInstallDataset


train_data = 'data//train_sample.csv'
valid_data = 'data//valid_sample.csv'
test_data  = 'data//test_sample.csv'

# mini_data_path = '/home/data/sunyingjie/zeng_appdata/mini_data/'
# train_data = mini_data_path + 'mini_train_data.csv'
# valid_data = mini_data_path + 'mini_valid_data.csv'
# test_data  = mini_data_path + 'mini_test_data.csv'

def parse_data(config):
    training_data = AppInstallDataset(train_data , config.num_class, config.length_his)
    validing_data = AppInstallDataset(valid_data , config.num_class, config.length_his)
    testing_data  = AppInstallDataset(test_data  , config.num_class, config.length_his)

    train_dataloader = DataLoader(training_data,
                                  batch_size=config.Batch_Size,
                                  shuffle=True
                                  )
    valid_dataloader = DataLoader(validing_data,
                                  batch_size=config.Batch_Size,
                                  shuffle=True
                                  )
    test_dataloader = DataLoader(testing_data,
                                 batch_size=config.Batch_Size,
                                 shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader


class BaseModelTrainer:
    def __init__(self, data, config, model):
        self.model = model
        self.config = config
        self.max_epoch = config.epoch
        self.device = config.device
        self.bestEpoch = 0

        self.train_data = data[0]
        self.valid_data = data[1]
        #self.test_data = data[2]

    def fit(self):
        self.savePath = 'savepath/'
        self.savePath = self.savePath + self.config.model_name + '/'

        print(self.savePath)

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.iter_per_epoch = len(self.train_data)

        print('iter_per_epoch',self.iter_per_epoch)
        # n = 0
        # assert n != 0, 'n is zero!'
        self.val_per_epoch = len(self.valid_data)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.5, total_iters=500)
        self.bestValLoss = float("inf")


        
        if self.config.checkpoint == True:
            self.config.checkpointpath = self.savePath  + '/' + 'model_after_90_epoch.pt'
            checkpoint = torch.load(self.config.checkpointpath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            self.bestValLoss = checkpoint['loss']
            self.bestState_dict = checkpoint['model_state_dict']
            self.bestEpoch = checkpoint['epoch']

        self.model = self.model.to(self.device)
        # self.optimizer = self.optimizer.to(self.device)
        self.loss_fn_seq = masked_softmax_cross_entropy_loss().to(self.device)

        self.train()

    def train(self):
        es = 0  # early stopping
        loss_app_list, loss_time_list, loss_list = [], [], []
        val_loss_app_list, val_loss_time_list, val_loss_list = [], [], []
        # # 训练可视化
        # win_loss_app = Visdom()
        # win_loss_time = Visdom()
        # win_loss = Visdom()

        # win_loss_app.line([[0., 0.]], [1.], win='app_1', opts=dict(title='loss_app', legend=['loss_app', 'val_loss_app']))
        # win_loss_time.line([[0., 0.]], [1.], win='time_1', opts=dict(title='loss_time', legend=['loss_time', 'val_loss_time']))
        # win_loss.line([[0., 0.]], [1.], win='all_1', opts=dict(title='loss', legend=['loss', 'val_loss']))

        for epoch in range(self.bestEpoch, self.max_epoch):
            start_time = time.time()
            epoch_loss_app, epoch_loss_time, epoch_loss = 0, 0, 0
            # train loop--------
            self.model.train()
            for batch, (X, y) in enumerate(self.train_data):
                # data to GPU
                #X = {key:X[key].to(self.device) for key in X}
                #y = {key:y[key].to(self.device) for key in y}

                # output of base model
                # print('x',X['new_soft_list'].size())
                # print('x',X['new_time_list'].size())
                # print('x',X['new_sep_list'].size())

                model_outputs = self.model(X, None)  # (input, task_emb)
                logits_new_app = model_outputs[1]
                logits_new_time = model_outputs[2]
                mask = X['new_mask_list']

                # loss
                loss_app = self.loss_fn_seq(logits_new_app, X['new_soft_list_onehot'], mask)
                loss_time = self.loss_fn_seq(logits_new_time, X['new_sep_list_onehot'], mask)
                loss = loss_app + loss_time

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss_app += loss_app/self.iter_per_epoch
                epoch_loss_time += loss_time/self.iter_per_epoch
                epoch_loss += loss/self.iter_per_epoch

            # valid loop 验证
            epoch_val_loss_app, epoch_val_loss_time, epoch_val_loss = 0, 0, 0
            self.model.eval()
            with torch.no_grad():
                for X, y in self.valid_data:
                    #X = {key:X[key].to(self.device) for key in X}
                    #y = {key:y[key].to(self.device) for key in y}

                    model_outputs = self.model(X, None)
                    logits_new_app = model_outputs[1]
                    logits_new_time = model_outputs[2]
                    mask = X['new_mask_list']

                    val_loss_app = self.loss_fn_seq(logits_new_app, X['new_soft_list_onehot'], mask)
                    val_loss_time = self.loss_fn_seq( logits_new_time, X['new_sep_list_onehot'], mask)
                    val_loss = val_loss_app + val_loss_time

                    # print(val_loss)

                    epoch_val_loss_app += val_loss_app/self.val_per_epoch
                    epoch_val_loss_time += val_loss_time/self.val_per_epoch
                    epoch_val_loss += val_loss/self.val_per_epoch

            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                path = self.savePath + "model_after_{}_epoch.pt".format(epoch+1)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    'loss': epoch_val_loss,
                }, path)

            # win_loss_app.line([[float(epoch_loss_app), float(epoch_val_loss_app)]], [epoch+1], win='app_1', update='append')
            # win_loss_time.line([[float(epoch_loss_time), float(epoch_val_loss_time)]], [epoch+1], win='time_1', update='append')
            # win_loss.line([[float(epoch_loss), float(epoch_val_loss)]], [epoch+1], win='all_1', update='append')

            # 保存模型的loss曲线
            loss_app_list.append(epoch_loss_app.item())
            loss_time_list.append(epoch_loss_time.item())
            loss_list.append(epoch_loss.item())
            val_loss_app_list.append(epoch_val_loss_app.item())
            val_loss_time_list.append(epoch_val_loss_time.item())
            val_loss_list.append(epoch_val_loss.item())

            # 更新状态
            if epoch_val_loss < self.bestValLoss:
                self.bestValLoss = epoch_val_loss
                self.bestState_dict = copy.deepcopy(self.model.state_dict())
                self.bestEpoch = epoch+1
                es = 0
            else:
                # early stop
                es += 1
                if es > 4:
                    print("early stop!")
                    break

            np.save(self.savePath + "loss_app.npy", loss_app_list)
            np.save(self.savePath + "loss_time.npy", loss_time_list)
            np.save(self.savePath + "loss.npy", loss_list)
            np.save(self.savePath + "val_loss_app.npy", val_loss_app_list)
            np.save(self.savePath + "val_loss_time.npy", val_loss_time_list)
            np.save(self.savePath + "val_loss.npy", val_loss_list)

            print(time.strftime("%H:%M:%S", time.localtime()))
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - loss_app: {epoch_loss_app:.4f} - loss_time : {epoch_loss_time:.4f}"
            )
            print(
                f"val_loss : {epoch_val_loss:.4f} - val_loss_app: {epoch_val_loss_app:.4f} - val_loss_time : {epoch_val_loss_time:.4f}"
            )
        # 保存最佳模型
        path = self.savePath+"bestmodel.pt"
        torch.save({
            'epoch': self.bestEpoch,
            'model_state_dict': self.bestState_dict,
            "optimizer": self.optimizer.state_dict(),
            'loss': self.bestValLoss,
        }, path)

class ModelTrainer:
    def __init__(self, data, config, model):
        self.model = model
        self.config = config
        self.max_epoch = config.epoch
        self.device = config.device
        self.bestEpoch = 0

        self.train_data = data[0]
        self.valid_data = data[1]
        #self.test_data = data[2]

    def fit(self):
        self.savePath = 'savepath/'
        self.savePath = self.savePath + self.config.model_name + '/'

        print(self.savePath)

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.iter_per_epoch = len(self.train_data)

        print('iter_per_epoch',self.iter_per_epoch)
        # n = 0
        # assert n != 0, 'n is zero!'
        self.val_per_epoch = len(self.valid_data)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.5, total_iters=500)
        self.bestValLoss = float("inf")


        
        if self.config.checkpoint == True:
            self.config.checkpointpath = self.savePath  + '/' + 'model_after_240_epoch.pt'
            checkpoint = torch.load(self.config.checkpointpath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            self.bestValLoss = checkpoint['loss']
            self.bestState_dict = checkpoint['model_state_dict']
            self.bestEpoch = checkpoint['epoch']

        self.model = self.model.to(self.device)
        # self.optimizer = self.optimizer.to(self.device)
        self.loss_fn_seq = masked_softmax_cross_entropy_loss().to(self.device)

        self.train()

    def train(self):
        es = 0  # early stopping
        loss_app_list, loss_time_list, loss_list = [], [], []
        val_loss_app_list, val_loss_time_list, val_loss_list = [], [], []
        # # 训练可视化
        # win_loss_app = Visdom()
        # win_loss_time = Visdom()
        # win_loss = Visdom()

        # win_loss_app.line([[0., 0.]], [1.], win='app_1', opts=dict(title='loss_app', legend=['loss_app', 'val_loss_app']))
        # win_loss_time.line([[0., 0.]], [1.], win='time_1', opts=dict(title='loss_time', legend=['loss_time', 'val_loss_time']))
        # win_loss.line([[0., 0.]], [1.], win='all_1', opts=dict(title='loss', legend=['loss', 'val_loss']))

        for epoch in range(self.bestEpoch, self.max_epoch):
            start_time = time.time()
            epoch_loss_app, epoch_loss_time, epoch_loss = 0, 0, 0
            # train loop--------
            self.model.train()
            for batch, (X, y) in enumerate(self.train_data):
                # data to GPU
                #X = {key:X[key].to(self.device) for key in X}
                #y = {key:y[key].to(self.device) for key in y}

                # output of base model
                # print('x',X['new_soft_list'].size())
                # print('x',X['new_time_list'].size())
                # print('x',X['new_sep_list'].size())

                model_outputs = self.model(X, None)  # (input, task_emb)
                logits_new_app = model_outputs[1]
                # logits_new_time = model_outputs[2]
                mask = X['new_mask_list']

                # loss
                loss_app = self.loss_fn_seq(logits_new_app, X['new_soft_list_onehot'], mask)
                # loss_time = self.loss_fn_seq(logits_new_time, X['new_sep_list_onehot'], mask)
                loss = loss_app 

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss_app += loss_app/self.iter_per_epoch
                # epoch_loss_time += loss_time/self.iter_per_epoch
                epoch_loss += loss/self.iter_per_epoch

            # valid loop 验证
            epoch_val_loss_app, epoch_val_loss_time, epoch_val_loss = 0, 0, 0
            self.model.eval()
            with torch.no_grad():
                for X, y in self.valid_data:
                    #X = {key:X[key].to(self.device) for key in X}
                    #y = {key:y[key].to(self.device) for key in y}

                    model_outputs = self.model(X, None)
                    logits_new_app = model_outputs[1]
                    # logits_new_time = model_outputs[2]
                    mask = X['new_mask_list']

                    val_loss_app = self.loss_fn_seq(logits_new_app, X['new_soft_list_onehot'], mask)
                    # val_loss_time = self.loss_fn_seq( logits_new_time, X['new_sep_list_onehot'], mask)
                    val_loss = val_loss_app 

                    # print(val_loss)

                    epoch_val_loss_app += val_loss_app/self.val_per_epoch
                    # epoch_val_loss_time += val_loss_time/self.val_per_epoch
                    epoch_val_loss += val_loss/self.val_per_epoch

            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                path = self.savePath + "model_after_{}_epoch.pt".format(epoch+1)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    'loss': epoch_val_loss,
                }, path)

            # win_loss_app.line([[float(epoch_loss_app), float(epoch_val_loss_app)]], [epoch+1], win='app_1', update='append')
            # win_loss_time.line([[float(epoch_loss_time), float(epoch_val_loss_time)]], [epoch+1], win='time_1', update='append')
            # win_loss.line([[float(epoch_loss), float(epoch_val_loss)]], [epoch+1], win='all_1', update='append')

            # 保存模型的loss曲线
            loss_app_list.append(epoch_loss_app.item())
            # loss_time_list.append(epoch_loss_time.item())
            loss_list.append(epoch_loss.item())


            val_loss_app_list.append(epoch_val_loss_app.item())
            # val_loss_time_list.append(epoch_val_loss_time.item())
            val_loss_list.append(epoch_val_loss.item())

            # 更新状态
            if epoch_val_loss < self.bestValLoss:
                self.bestValLoss = epoch_val_loss
                self.bestState_dict = copy.deepcopy(self.model.state_dict())
                self.bestEpoch = epoch + 1
                es = 0
            else:
                # early stop
                es += 1
                if es > 4:
                    print("early stop!")
                    break

            np.save(self.savePath + "loss_app.npy", loss_app_list)
            np.save(self.savePath + "loss_time.npy", loss_time_list)
            np.save(self.savePath + "loss.npy", loss_list)
            np.save(self.savePath + "val_loss_app.npy", val_loss_app_list)
            # np.save(self.savePath + "val_loss_time.npy", val_loss_time_list)
            np.save(self.savePath + "val_loss.npy", val_loss_list)

            print(time.strftime("%H:%M:%S", time.localtime()))
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - loss_app: {epoch_loss_app:.4f} - loss_time : {epoch_loss_time:.4f}"
            )
            print(
                f"val_loss : {epoch_val_loss:.4f} - val_loss_app: {epoch_val_loss_app:.4f} - val_loss_time : {epoch_val_loss_time:.4f}"
            )
        # 保存最佳模型
        path = self.savePath+"bestmodel.pt"
        torch.save({
            'epoch': self.bestEpoch,
            'model_state_dict': self.bestState_dict,
            "optimizer": self.optimizer.state_dict(),
            'loss': self.bestValLoss,
        }, path)
