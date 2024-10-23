import numpy as np
from Beta_estimate import Beta_est
from C_estimation import C_est
from I_spline import I_S
from baseL import baseL
from g_deep import g_D

def Est_deep(train_data,X_test,corstr, Beta0,n_layer,n_node,n_lr,n_epoch,nodevec,m, c0, eps, itermax):
    Status_train = train_data['Status']
    time_train = train_data['time']
    id_train = train_data['id']
    Z_train = train_data['Z']
    Beta0 = np.array([Beta0])
    Lambda = I_S(m,c0,time_train,nodevec)
    g_X=train_data['g_X']#np.zeros((Z_train.shape[0], 1))
  #  Lambda = baseL(time_train, Status_train, Z_train, Beta0, g_X)['Lambda']
  #  GEE_fit= Beta_est(Lambda, Status_train,Z_train,Beta0, id_train, eps, corstr, itermax, g_X)
  #  Beta0=GEE_fit['beta']
    rho=0 #GEE_fit['rho']
    pphi=1 #GEE_fit['pphi']
    convergence = False
    for loop in range(itermax):
        g_X = g_D(train_data, X_test, Lambda, corstr, rho, pphi, Beta0, n_layer, n_node, n_lr, n_epoch)
        g_train = g_X['g_train']
        c1 = C_est( m,time_train,Status_train,Z_train, Beta0, g_train, nodevec,id_train,corstr)
        Lambda = I_S(m,c1,time_train,nodevec)
        #Lambda = baseL(time_train, Status_train, Z_train, Beta0, g_train)['Lambda']
        GEE_fit = Beta_est(Lambda, Status_train, Z_train, Beta0, id_train, eps, corstr, itermax, g_train)
        Beta1 =GEE_fit['beta']
        print(Beta1)
        rho = GEE_fit['rho']
        pphi = GEE_fit['pphi']
        if (abs(Beta0-Beta1) <= eps):
            convergence = True
            break
        #c0 = c1
        Beta0 = Beta1
    return {
        'g_train': g_train,
        'g_test': g_X['g_test'],
        'c': c1,
        'Beta': Beta1,
        'convergence': convergence #and GEE_fit['convergence'],
    }
