import numpy as np
import torch
import pandas as pd
from torch import nn
from scipy.linalg import pinv
import math

# 启用NumPy和R对象之间的自动转换
#numpy2ri.activate()

# 导入R中的emplik包
#emplik = importr('emplik')
from warmup_scheduler import GradualWarmupScheduler

def g_D_ind(train_data,X_test, Beta,n_layer,n_node,n_lr,n_epoch):
    Z_train = torch.Tensor(train_data['Z'])
    X_train = torch.Tensor(train_data['X'])
    time_train =  torch.Tensor(train_data['time'])
    Status_train =  torch.Tensor(train_data['Status'])
    X_test = torch.Tensor(X_test)
    Beta = torch.Tensor(Beta)

    class DNNModel(torch.nn.Module):
        def __init__(self):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(5, n_node))
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 1))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred

    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)

    def my_loss_ind(Status1, Time1, Z1, beta, g_X1):
        LLR = torch.zeros(Z1.shape[0])
        for i in range(Z1.shape[0]):
            LLR[i] = torch.sum(torch.exp(Z1*beta + g_X1)[Time1 >= Time1[i]])
        loss_fun = -torch.mean(Status1 * (Z1*beta + g_X1 - torch.log(LLR)))
        return loss_fun


    for epoch in range(n_epoch):
        pred_g_X = model(X_train)
        loss = my_loss_ind(Status_train, time_train, Z_train, Beta, pred_g_X[:, 0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    g_train = model(X_train)
    g_test = model(X_test)
    g_train = g_train[:,0].detach().numpy()
    g_test = g_test[:,0].detach().numpy()
    return {
        'g_train': g_train,
        'g_test': g_test
    }
