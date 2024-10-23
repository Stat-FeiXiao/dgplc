import numpy as np
import pandas as pd


def baseL(Time, Status, X1, beta, g_X):
    t2 = Time
    t11 = np.sort(Time)
    c11 = Status[np.argsort(Time)]
    X = np.array(X1).reshape(-1, 1)
    x111 = X[np.argsort(Time), :]
    tt1 = np.unique(t11[c11 == 1])
    kk = len(np.unique(t11[c11 == 1]))
    dd = np.array(pd.Series(t11[c11 == 1]).value_counts().sort_index())
    gSS = np.zeros(kk)
    alpha = np.zeros(kk)
    Kn = len(Time)
    print(type(g_X))
    g_X=np.array(g_X, dtype='float32')
    gSS[0] = alpha[0] = dd[0] / np.sum(np.exp(    beta @ x111[np.min(np.where(t11 == tt1[0])):Kn, :].T + g_X[np.min(np.where(t11 == tt1[0])):Kn]  ))

    for i in range(1, kk):
        alpha[i] = dd[i] / np.sum(np.exp( beta @ x111[np.min(np.where(t11 == tt1[i])):Kn, :].T + g_X[np.min(np.where(t11 == tt1[i])):Kn] ))
        gSS[i] = gSS[i - 1] + alpha[i]

    gSS3 = np.zeros(Kn)

    for i in range(Kn):
        kk1 = 0
        if t2[i] < tt1[0]:
            gSS3[i] = 0
        elif t2[i] >= tt1[kk - 1]:
            gSS3[i] = gSS[kk - 1]
        else:
            while t2[i] >= tt1[kk1]:
                kk1 += 1
            gSS3[i] = gSS[kk1 - 1]

    bcumhaz = gSS3

    return {'Lambda': bcumhaz, 'alpha': alpha}