import torch
from torchvision import datasets, transforms
import numpy as np
import Clients
import model as m
import random
import scipy
from scipy.optimize import fsolve
from sympy import *
import math
import os
import json


total_iter = 25

delta = 0.01
data_len = 100
c = 1.5*math.sqrt(2 * math.log(1.25 / (delta)))
L = 1


if __name__ == '__main__':



    # train with MNIST dataset
    model = m.ALL_CNN_MNIST()  # global model
    tmp_model = m.ALL_CNN_MNIST()  # temporary model
    ts_model = m.ALL_CNN_MNIST()  # temporary model
    start_model = m.ALL_CNN_MNIST()  # initial global model
    local_model = m.ALL_CNN_MNIST()  # local models



    torch.save(model, "net_params.pkl")
    start_model = torch.load("net_params.pkl")

    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')






    # NbAFL-different N
    e = 60
    N = [50,60,80,100]
    for client_num in N:

        clients = Clients.Client(client_num, total_iter, data_len, c, L)
        clients.assign_training_set()

        try:

            torch.save(start_model, "net_params.pkl")
            model = torch.load("net_params.pkl")

            device = torch.device("cpu")
            client = list(range(0, client_num, 1))

            if client_num == 50:
                file = 'DPFL_N_50.txt'
            elif client_num == 60:
                file = 'DPFL_N_60.txt'
            elif client_num == 80:
                file = 'DPFL_N_80.txt'
            else:
                file = 'DPFL_N_100.txt'




            clients.set_budget(e,delta)
            x = []
            y = []

            clients.data_clean()

            for i in range(total_iter):
                norm_ave = 0
                loss = [0] * client_num

                select_client_num = 0
                torch.save(model, "net_params.pkl")
                tmp_model = torch.load("net_params.pkl")
                for v in ts_model.parameters():
                    with torch.no_grad():
                        v.zero_()

                norm_temp = [0] * client_num
                for k in client:
                    clients.set_model(tmp_model)
                    norm_temp[k] = clients.training_FedSGD_pre(k, i + 1)
                norm_temp.sort()
                print(norm_temp)
                if client_num % 2 != 0:
                    C = norm_temp[int(client_num / 2)]
                else:
                    C = (norm_temp[int(client_num / 2) - 1] + norm_temp[int(client_num / 2)]) / 2
                print("C:", C)

                for k in client:
                    print("communication time is ", i + 1, " NO ", k)

                    clients.set_model(tmp_model)

                    local_model = clients.training_DPFL(k, i + 1, C)

                    for lv, tv in zip(local_model.parameters(), ts_model.parameters()):
                        with torch.no_grad():

                            tv.add_(lv)
                for tv, gv in zip(ts_model.parameters(), model.parameters()):
                    with torch.no_grad():
                        gv.zero_()
                        tv.mul_(1 / client_num)
                        gv.add_(tv)

                L_client, delta, epsilon, datalen = clients.getL_delta_epsilon_datalen()
                print(L_client, delta, epsilon, datalen)
                if total_iter > L_client * math.sqrt(client_num):
                    print("Yes!!!")

                    noise_scale = 2 * c * C * math.sqrt(total_iter * total_iter - L_client * L_client * client_num) / (
                                epsilon * datalen * client_num)
                    for gv in model.parameters():
                        with torch.no_grad():
                            noise = torch.from_numpy(np.random.normal(0.0, noise_scale, gv.shape))
                            gv.add_(noise)
                else:
                    print("No!!!")
                loss_avg = 0
                for k in client:
                    clients.set_model(model)
                    loss[k] = clients.getloss(k)
                    loss_avg = loss_avg + loss[k]
                print("loss", loss)

                loss_avg = loss_avg / client_num


                print(i + 1, loss_avg)
                x.append(i + 1)
                y.append(loss_avg.item())

            # store the result
            infile = open(file, 'w')
            for x_, y_ in zip(x, y):
                infile.writelines(str(x_) + ',' + str(y_) + '\n')
            infile.close()
        finally:
            print("ok---", N)












