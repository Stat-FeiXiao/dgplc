import numpy as np
from Beta_estimate_ind import Beta_est_ind
from C_estimation import C_est
from I_spline import I_S
from baseL import baseL
from g_deep_ind import g_D_ind

def Est_deep_ind(train_data,X_test, Beta0,n_layer,n_node,n_lr,n_epoch,nodevec,m, c0, eps, itermax):
    Status_train = train_data['Status']
    time_train = train_data['time']
    Z_train = train_data['Z']
    Beta0 = np.array([Beta0])
    g_X=train_data['g_X']#np.zeros((Z_train.shape[0], 1))
    convergence = False
    for loop in range(itermax):
        g_X = g_D_ind(train_data, X_test,  Beta0, n_layer, n_node, n_lr, n_epoch)
        g_train = g_X['g_train']
        Ind_fit = Beta_est_ind(Status_train, Z_train, Beta0, time_train, g_train)
        Beta1 =Ind_fit
        if (abs(Beta0-Beta1) <= eps):
            convergence = True
            break
        Beta0 = Beta1
    return {
        'g_train': g_train,
        'g_test': g_X['g_test'],
        'Beta': Beta1,
        'convergence': convergence,
    }
