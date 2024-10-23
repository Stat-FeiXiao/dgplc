
import torch
from torch import nn
import numpy as np

def LFD(train_data,Lambda_U,g_train,Beta,n_layer,n_node,n_lr,n_epoch):
    Z_train = torch.Tensor(train_data['Z'])
    De_train = torch.Tensor(train_data['De'])
    X_U = torch.Tensor(np.c_[train_data['X'], train_data['U']])
    Lambda_U = torch.Tensor(Lambda_U)
    Beta = torch.Tensor(np.array([Beta]))
    class DNNAB(torch.nn.Module):
        def __init__(self):
            super(DNNAB, self).__init__()
            layers = []
            layers.append(nn.Linear(6, n_node))
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 1))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred

    model = DNNAB()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def Loss(De, Z, Beta, Lambda_U, g_X, a_b):
        h_v = Lambda_U * torch.exp( Z*Beta + g_X)
        Q_y = h_v * (De * torch.exp(-h_v)/(1-torch.exp(-h_v)+1e-5) - (1-De))
        Loss_f = torch.mean(Q_y**2 * (Z-a_b)**2)
        return Loss_f

    for epoch in range(n_epoch):
        pred_ab = model(X_U)
        loss = Loss(De_train, Z_train, Beta, Lambda_U, g_train, pred_ab[:, 0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ab_train = model(X_U)
    ab_train = ab_train[:,0].detach().numpy()
    return ab_train
