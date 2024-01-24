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


client_num = 50
total_iter = 25

delta = 0.01
data_len = 100
c = 1.25*math.sqrt(2 * math.log(1.25 / (delta)))
L = 3


if __name__ == '__main__':


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



    clients = Clients.Client(client_num,total_iter,data_len,c,L)

    clients.assign_training_set()







    # NbAFL-different K for different E

    E = [50,60,100]
    K_vector = [10,15,20,25,30,35,40,45,50]
    for e in E:
        for K in K_vector:
            try:

                torch.save(start_model, "net_params.pkl")
                model = torch.load("net_params.pkl")

                device = torch.device("cpu")
                client = list(range(0, client_num, 1))
                if e == 50:
                    if K == 10:
                        file = 'DPFL_e_50_K_10_partial.txt'
                    elif K == 15:
                        file = 'DPFL_e_50_K_15_partial.txt'
                    elif K == 20:
                        file = 'DPFL_e_50_K_20_partial.txt'
                    elif K == 25:
                        file = 'DPFL_e_50_K_25_partial.txt'
                    elif K == 30:
                        file = 'DPFL_e_50_K_30_partial.txt'
                    elif K == 35:
                        file = 'DPFL_e_50_K_35_partial.txt'
                    elif K == 40:
                        file = 'DPFL_e_50_K_40_partial.txt'
                    elif K == 45:
                        file = 'DPFL_e_50_K_45_partial.txt'
                    elif K == 50:
                        file = 'DPFL_e_50_K_50_partial.txt'
                elif e==60:
                    if K == 10:
                        file = 'DPFL_e_60_K_10_partial.txt'
                    elif K == 15:
                        file = 'DPFL_e_60_K_15_partial.txt'
                    elif K == 20:
                        file = 'DPFL_e_60_K_20_partial.txt'
                    elif K == 25:
                        file = 'DPFL_e_60_K_25_partial.txt'
                    elif K == 30:
                        file = 'DPFL_e_60_K_30_partial.txt'
                    elif K == 35:
                        file = 'DPFL_e_60_K_35_partial.txt'
                    elif K == 40:
                        file = 'DPFL_e_60_K_40_partial.txt'
                    elif K == 45:
                        file = 'DPFL_e_60_K_45_partial.txt'
                    elif K == 50:
                        file = 'DPFL_e_60_K_50_partial.txt'
                else:
                    if K == 10:
                        file = 'DPFL_e_100_K_10_partial.txt'
                    elif K == 15:
                        file = 'DPFL_e_100_K_15_partial.txt'
                    elif K == 20:
                        file = 'DPFL_e_100_K_20_partial.txt'
                    elif K == 25:
                        file = 'DPFL_e_100_K_25_partial.txt'
                    elif K == 30:
                        file = 'DPFL_e_100_K_30_partial.txt'
                    elif K == 35:
                        file = 'DPFL_e_100_K_35_partial.txt'
                    elif K == 40:
                        file = 'DPFL_e_100_K_40_partial.txt'
                    elif K == 45:
                        file = 'DPFL_e_100_K_45_partial.txt'
                    elif K == 50:
                        file = 'DPFL_e_100_K_50_partial.txt'



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
                    select_client = np.random.choice(client, K, replace =False )
                    print("select,",select_client)


                    norm_temp = [0] * K

                    j = 0
                    for k in select_client:
                        clients.set_model(tmp_model)
                        norm_temp[j] = clients.training_FedSGD_pre(k, i + 1)
                        j = j + 1
                    norm_temp.sort()
                    print(norm_temp)
                    if K % 2 != 0:
                        C = norm_temp[int(K / 2)]
                    else:
                        C = (norm_temp[int(K / 2) - 1] + norm_temp[int(K / 2)]) / 2

                    print("C:", C)


                    for k in select_client:
                        print("communication time is ", i + 1, " NO ", k)
                        clients.set_model(tmp_model)

                        local_model = clients.training_DPFL(k, i + 1, C)

                        for lv, tv in zip(local_model.parameters(), ts_model.parameters()):
                            with torch.no_grad():

                                tv.add_(lv)
                    for tv, gv in zip(ts_model.parameters(), model.parameters()):
                        with torch.no_grad():
                            gv.zero_()
                            tv.mul_(1 / K)
                            gv.add_(tv)

                    L_client, delta, epsilon, datalen = clients.getL_delta_epsilon_datalen()
                    print(L_client, delta, epsilon, datalen)

                    gamma = -math.log(1-K/client_num+K/client_num*math.exp(-epsilon/(L_client*math.sqrt(K))))
                    print("gamma,",gamma)
                    if total_iter > epsilon/gamma:
                        print("Yes!!!", total_iter, epsilon/gamma)
                        b = (- total_iter / epsilon) * math.log(
                            1 - client_num / K + client_num / K * math.exp(- epsilon / total_iter))

                        noise_scale = 2 * c * C * math.sqrt(total_iter * total_iter/(b*b) - L_client * L_client * client_num) / (
                                    epsilon * datalen * K)
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
                print("ok---",e,"---",K)



    # Non-Private
    K_vector = [10,15,20,25,30,35,40,45,50]
    for K in K_vector:
        try:

            torch.save(start_model, "net_params.pkl")
            model = torch.load("net_params.pkl")

            device = torch.device("cpu")
            client = list(range(0, client_num, 1))
            if K == 10:
                file = 'Non_Private_K_10_partial.txt'
            elif K == 15:
                file = 'Non_Private_K_15_partial.txt'
            elif K == 20:
                file = 'Non_Private_K_20_partial.txt'
            elif K == 25:
                file = 'Non_Private_K_25_partial.txt'
            elif K == 30:
                file = 'Non_Private_K_30_partial.txt'
            elif K == 35:
                file = 'Non_Private_K_35_partial.txt'
            elif K == 40:
                file = 'Non_Private_K_40_partial.txt'
            elif K == 45:
                file = 'Non_Private_K_45_partial.txt'
            elif K == 50:
                file = 'Non_Private_K_50_partial.txt'


            x = []
            y = []


            clients.data_clean()

            for i in range(total_iter):
                norm_ave = 0
                loss = [0]* client_num

                select_client_num = 0
                torch.save(model, "net_params.pkl")
                tmp_model = torch.load("net_params.pkl")
                for v in ts_model.parameters():
                    with torch.no_grad():
                        v.zero_()

                select_client = np.random.choice(client, K, replace=False)
                print("select,", select_client)

                norm_temp = [0] * K

                j = 0
                for k in select_client:
                    clients.set_model(tmp_model)
                    norm_temp[j] = clients.training_FedSGD_pre(k, i + 1)
                    j = j+1
                norm_temp.sort()
                print(norm_temp)
                if K % 2 != 0:
                    C = norm_temp[int(K / 2)]
                else:
                    C = (norm_temp[int(K / 2) - 1] + norm_temp[int(K / 2)]) / 2
                print("C:", C)



                for k in select_client:
                    print("communication time is ", i + 1, " NO ", k)
                    clients.set_model(tmp_model)
                    local_model = clients.training_FedSGD_after(k, i + 1,C)

                    for lv, tv in zip(local_model.parameters(), ts_model.parameters()):
                        with torch.no_grad():
                            tv.add_(lv)
                for tv, gv in zip(ts_model.parameters(), model.parameters()):
                    with torch.no_grad():
                        gv.zero_()
                        tv.mul_(1/K)
                        gv.add_(tv)
                loss_avg = 0
                for k in client:
                    clients.set_model(model)
                    loss[k] = clients.getloss(k)
                    loss_avg = loss_avg + loss[k]
                loss_avg = loss_avg / client_num
                print(i+1,loss_avg)
                x.append(i+1)
                y.append(loss_avg.item())

            # store the result
            infile = open(file, 'w')
            for x_, y_ in zip(x, y):
                infile.writelines(str(x_) + ',' + str(y_) + '\n')
            infile.close()
        finally:
            print("ok---Non-Private---",K)






