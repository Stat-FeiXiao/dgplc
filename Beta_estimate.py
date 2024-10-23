import numpy as np
from scipy.linalg import pinv, sqrtm
from scipy.optimize import root_scalar


def Beta_est(Lambda, Status, X1, beta, id, eps, corstr, itermax, g_X):

    kk = np.sum(Status)
    K = len(np.unique(id))
    n = np.bincount(id)[1:]  # Shift index since R index starts at 1
    gbeta = beta1 = beta
    X = np.array(X1).reshape(-1, 1)
    mu = np.exp(X @ gbeta + g_X)

    Lambda1 = np.where(Lambda == 0, 1e-08, Lambda)

    newY1 = Status / Lambda1
    W1 = np.diag(Lambda)

    res = (newY1 - mu) / np.sqrt(mu)
    pphi = np.sum(res ** 2) / (np.sum(n) - X.shape[1])
    resm = np.zeros((max(n), K))
    for i in range(K):
        resm[:n[i], i] = res[id == i + 1]

    res = resm.T
    rho = 0

    # Update rho based on corstr
    if corstr == "exchangeable":
        rres = 0
        for i in range(K):
            if n[i] == 1:
                rres += res[i, 0]
            else:
                rres += np.sum(res[i, :-1] * np.sum(res[i, 1:], axis=0))
        rho = pphi ** (-1) * rres / (np.sum(n * (n - 1) / 2) - X.shape[1])
    elif corstr == "AR1":
        rres = 0
        for i in range(K):
            if n[i] > 1:
                rres += np.sum(res[i, :-1] * res[i, 1:])
        rho = pphi ** (-1) * rres / (np.sum(n - 1) - X.shape[1])

    S1 = Status - mu * Lambda
    SK1 = 1

    #while True:
    QC = None
    if corstr == "independence":
        QC = np.diag(n[0] * [1])
    elif corstr == "exchangeable":
        QC = np.full((n[0], n[0]), rho)
        np.fill_diagonal(QC, 1)
    elif corstr == "AR1":
        exponent = np.abs(np.subtract.outer(np.arange(n[0]), np.arange(n[0])))
        QC = rho ** exponent

    W1 = np.diag(Lambda[id == 1])
    D1 = np.diag(mu[id == 1]) @ X[id == 1, :]
    print(min(mu))
    V1 = pinv(sqrtm(np.diag(mu[id == 1]))) @ (pphi ** (-1) * pinv(QC)) @ pinv(sqrtm(np.diag(mu[id == 1])))
    G = D1.T @ V1 @ S1[id == 1]
    G1 = -D1.T @ V1 @ W1 @ D1

    for i in range(1, K):
        if corstr == "independence":
            QC = np.diag(n[i] * [1])
        elif corstr == "exchangeable":
            QC = np.full((n[i], n[i]), rho)
            np.fill_diagonal(QC, 1)
        elif corstr == "AR1":
            exponent = np.abs(np.subtract.outer(np.arange(n[i]), np.arange(n[i])))
            QC = rho ** exponent

        W1 = np.diag(Lambda[id == (i+1)])
        D1 = np.diag(mu[id == (i+1)]) @ X[id == (i+1), :]
        V1 = pinv(sqrtm(np.diag(mu[id == (i+1)]))) @ (pphi ** (-1) * pinv(QC)) @ pinv(sqrtm(np.diag(mu[id == (i+1)])))
        G = G + (D1.T @ V1 @ S1[id == (i+1)])
        G1 = G1 - (D1.T @ V1 @ W1 @ D1)

    gbeta = gbeta - pinv(G1) @ G
    mu = np.exp(X @ gbeta + g_X)
    S1 = Status - mu * Lambda

    # Update loop condition based on your conditions and break logic
    #if not (np.any(np.abs(gbeta - beta1) > eps) and SK1 <= itermax):
    #    break
    #beta1 = gbeta
    #SK1 += 1

    #convergence = (SK1 <= itermax) , 'convergence': convergence
    return {'beta': gbeta, 'rho': rho, 'pphi': pphi}