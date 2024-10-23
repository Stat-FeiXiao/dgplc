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

def g_D(train_data,X_test,Lambda,corstr, rho, pphi,Beta,n_layer,n_node,n_lr,n_epoch):
    Z_train = torch.Tensor(train_data['Z'])
    X_train = torch.Tensor(train_data['X'])
    Status_train =  torch.Tensor(train_data['Status'])
    id_train = torch.Tensor(train_data['id'])
    X_test = torch.Tensor(X_test)
    Lambda = torch.Tensor(Lambda)
    Beta = torch.Tensor(Beta)

    K = len(id_train.unique())
    id_train_series = pd.Series(id_train.numpy())
    n = id_train_series.value_counts().sort_index().values

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

    def my_loss(Status1, id, corstr, K, n, Z1, beta, baseF1, g_X1):
        Z = torch.Tensor(Z1).reshape(-1, 1).float()  # Ensure Z1 is a 2D tensor
        g_X = torch.Tensor(g_X1).reshape(-1, 1).float()
        Status = torch.Tensor(Status1).reshape(-1, 1)
        baseF = torch.Tensor(baseF1).reshape(-1, 1)

        # Calculate mu using tensors
        mu = torch.exp(Z * beta + g_X)

        # Calculate S1 (residuals)
        S1 = Status - mu * baseF

        # Initialize G and C as PyTorch tensors
        if corstr == "independence":
            G = torch.zeros((Z.shape[1], 1))
            C = torch.zeros((Z.shape[1], Z.shape[1]))
        else:
            G = torch.zeros((2 * Z.shape[1], 1))
            C = torch.zeros((2 * Z.shape[1], 2 * Z.shape[1]))

        for i in range(K):
            M1 = torch.diag(torch.ones(n[i]))  # Diagonal matrix for M1

            if corstr == "exchangeable":
                M2 = torch.ones((n[i], n[i]))
                M2.fill_diagonal_(0)  # Fill the diagonal with zeros (in-place operation)
            elif corstr == "AR1":
                M2 = torch.zeros((n[i], n[i]))
                M2[0, 1] = 1
                for o in range(1, n[i] - 1):
                    M2[o, o + 1] = 1
                    M2[o, o - 1] = 1
                M2[n[i] - 1, n[i] - 2] = 1

            # No need to convert to NumPy. Stay in PyTorch.
            mu_tensor = mu[id == (i + 1)]  # Filter the relevant mu values for the group
            Z_tensor = Z[id == (i + 1), :]  # Filter the relevant Z values for the group

            # Ensure mu_tensor is 1D for diag operation
            if mu_tensor.ndim == 2:
                mu_tensor = mu_tensor.flatten()  # Flatten to a 1D array

            # Perform matrix multiplication with PyTorch
            D1 = torch.diag(mu_tensor) @ Z_tensor  # Matrix multiplication

            mu_diag = torch.diag(mu_tensor)  # Create diagonal matrix
            mu_sqrt = torch.sqrt(mu_diag)  # Square root of the diagonal matrix
            mu_sqrt_inv = torch.inverse(mu_sqrt)  # Pseudoinverse (torch.inverse)

            V1 = mu_sqrt_inv @ M1 @ mu_sqrt_inv  # Matrix multiplication
            G_1 = torch.mm(D1.T, torch.mm(V1, S1[id == (i + 1)]))  # Multiply with S1

            if corstr != "independence":
                V2 = mu_sqrt_inv @ M2 @ mu_sqrt_inv  # Matrix multiplication for V2
                G_1 = torch.vstack([G_1, torch.mm(D1.T, torch.mm(V2, S1[id == (i + 1)]))])

            G += G_1
            C += torch.mm(G_1, G_1.T)

        loss_fun = torch.mm(G.T, torch.inverse(C)) @ G  # Matrix multiplication for the final loss function

        return loss_fun



    for epoch in range(n_epoch):
        pred_g_X = model(X_train)
        loss = my_loss(Status_train, id_train, corstr, K, n, Z_train, Beta, Lambda, pred_g_X[:, 0])
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
