from torchvision import datasets, transforms
import torch
import numpy as np
import model as m
import math
import random



epoch_num = 50


# 初始化客户端
class Client():
    def __init__(self, client_num, total_iter,data_len,c,L):
        self.client_num = client_num
        self.total_iter = total_iter
        self.each_client_data_len = data_len
        self.c = c
        self.L = L

        self.model = m.ALL_CNN_MNIST()
        self.model1 = m.ALL_CNN_MNIST()
        self.model2 = m.ALL_CNN_MNIST()
        self.g = m.ALL_CNN_MNIST()
        print(self.model)


        # data length of each client
        self.data_len = []
        # data sample of each client
        self.train_loader = []
        # communication time of each client
        self.communication_time = [0] * self.client_num

        # privacy budget of each client
        self.e = [0] * self.client_num
        self.delta = 0.01

        # hyper-parameters
        # learning rate
        self.lr = 0.002
        # regularization parameter
        self.mu = 1

    # Assign data for each client
    def assign_training_set(self):

        # load training dataset
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST('D:/home/aistudio/data/data68', train=True, download=True,
                                    transform=transform_train)

        # shuffle
        temp_data = []
        num_list = list(range(len(train_data)))
        select = random.sample(num_list, len(train_data))
        for j in select:
            num_list.remove(j)
            temp_data.append(train_data[j])

        # assign data
        data_index = 0
        for i in range(self.client_num):
            data_len = 0
            data = []
            per_num = self.each_client_data_len

            for k in range(per_num):
                data.append(temp_data[data_index])
                data_index = data_index + 1
            data_len = per_num
            print(data_len)
            self.data_len.append(data_len)

            train_loader = torch.utils.data.DataLoader(
                data,
                batch_size=data_len, shuffle=True,
                num_workers=0, drop_last=False
            )
            self.train_loader.append(train_loader)
        print(self.data_len)



    # Reset communication time for each client
    def data_clean(self):
        self.communication_time = [0] * self.client_num



    # Return the data length of the client idx
    def get_dataLen(self, idx):
        return self.data_len[idx]

    # Set the privacy budget for each client
    def set_budget(self, epsilon,delta):
        for i in range(self.client_num):
            self.e[i] = epsilon
        self.delta = delta

    # Return the problem-related parameter
    def getL_delta_epsilon_datalen(self):
        return self.L, self.delta, self.e[0], self.data_len[0]

    # Download models
    def set_model(self, model):
        torch.save(model, "net_params.pkl")
        self.model = torch.load("net_params.pkl")
        self.model1 = torch.load("net_params.pkl")
        self.model2 = torch.load("net_params.pkl")


    # training for NbAFL
    def training_DPFL(self, idx, com_time, C):
        self.communication_time[idx] = self.communication_time[idx] + 1
        self.model = self.model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr)
        # sensitivity of model parameters
        delta_gradient = 2 * C / self.data_len[idx]
        # training
        for epo in range(epoch_num):
            for v1, v2 in zip(self.model.parameters(), self.model2.parameters()):
                with torch.no_grad():
                    v2.zero_()
                    v2.add_(v1)
            self.model.train()
            for i, (data, label) in enumerate(self.train_loader[idx]):
                data = data.to(device)
                label = label.to(device)
                pred = self.model(data)
                loss = loss_fn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Train Epoch: {},  Loss: {}".format(epo, loss.item()))
            # update local model
            for gradv, gv, ggv, ggv2 in zip(self.g.parameters(), self.model.parameters(), self.model1.parameters(),
                                            self.model2.parameters()):
                with torch.no_grad():
                    gradv.zero_()
                    gradv.add_(ggv2 - ggv)
                    gradv.mul_(self.lr * self.mu)
                    gv.add_(-gradv)
            # clip
            sum = 0
            for gv in self.model.parameters():
                with torch.no_grad():
                    sum = sum + gv.norm(2) * gv.norm(2)
            for gv in self.model.parameters():
                with torch.no_grad():
                    gv /= max(1, math.sqrt(sum) / C)
        # add noise
        noise_scale = self.L * self.c * delta_gradient / self.e[idx]
        for gv in self.model.parameters():
            with torch.no_grad():
                noise = torch.from_numpy(np.random.normal(0.0, noise_scale, gv.shape))
                gv.add_(noise)
        return self.model

    # Get norm of original model parameter
    def training_FedSGD_pre(self, idx, com_time):

        self.communication_time[idx] = self.communication_time[idx] + 1
        self.model = self.model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr)

        # training
        for epo in range(epoch_num):
            for v1,v2 in zip(self.model.parameters(),self.model2.parameters()):
                with torch.no_grad():
                    v2.zero_()
                    v2.add_(v1)
            self.model.train()
            for i, (data, label) in enumerate(self.train_loader[idx]):
                data = data.to(device)
                label = label.to(device)
                pred = self.model(data)
                loss = loss_fn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Train Epoch: {},  Loss: {}".format(epo, loss.item()))
            for gradv, gv, ggv, ggv2 in zip(self.g.parameters(), self.model.parameters(), self.model1.parameters(),self.model2.parameters()):
                with torch.no_grad():
                    gradv.zero_()
                    gradv.add_(ggv2 - ggv)
                    gradv.mul_(self.lr*self.mu)
                    gv.add_(-gradv)
        sum = 0
        for gv in self.model.parameters():
            with torch.no_grad():
                sum = sum + gv.norm(2) * gv.norm(2)
        sum = math.sqrt(sum)
        print("norm",sum)
        return sum

    # training for Non-private
    def training_FedSGD_after(self, idx, com_time,C):

        self.communication_time[idx] = self.communication_time[idx] + 1
        self.model = self.model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr)

        # training
        for epo in range(epoch_num):
            for v1, v2 in zip(self.model.parameters(), self.model2.parameters()):
                with torch.no_grad():
                    v2.zero_()
                    v2.add_(v1)
            self.model.train()
            for i, (data, label) in enumerate(self.train_loader[idx]):
                data = data.to(device)
                label = label.to(device)
                pred = self.model(data)
                loss = loss_fn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Train Epoch: {},  Loss: {}".format(epo, loss.item()))

            for gradv, gv, ggv, ggv2 in zip(self.g.parameters(), self.model.parameters(), self.model1.parameters(),
                                            self.model2.parameters()):
                with torch.no_grad():
                    gradv.zero_()
                    gradv.add_(ggv2 - ggv)
                    gradv.mul_(self.lr * self.mu)
                    gv.add_(-gradv)
            sum = 0
            for gv in self.model.parameters():
                with torch.no_grad():
                    sum = sum + gv.norm(2) * gv.norm(2)
            for gv in self.model.parameters():
                with torch.no_grad():
                    gv /= max(1, math.sqrt(sum) / C)
        return self.model


    # Evalute the loss of global model in clients
    def getloss(self, idx):
        self.model = self.model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr)
        loss_total = 0
        # eval
        self.model.eval()
        for i, (data, label) in enumerate(self.train_loader[idx]):
            data = data.to(device)
            label = label.to(device)
            pred = self.model(data)
            loss = loss_fn(pred, label)
            loss_total = loss_total+loss
        loss_total = loss_total / (i+1)
        return loss_total










