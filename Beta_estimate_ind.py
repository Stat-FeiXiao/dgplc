
import numpy as np
import scipy.optimize as spo
def Beta_est_ind(De, Z, beta_int, Time, g_X):
    def BF(*args):
        LLR = np.zeros(Z.shape[0])
        for i in range(Z.shape[0]):
            LLR[i] = np.sum(np.exp(Z * args[0] + g_X)[Time >= Time[i]])
        Loss_F = -np.mean(De * (Z * args[0] + g_X - np.log(LLR)))
        return Loss_F
    result = spo.minimize(BF,beta_int,method='SLSQP')
    return result['x']