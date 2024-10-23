
import numpy as np
import pandas as pd
import scipy.optimize as spo
from I_spline import I_U

def C_est(m, U, De, Z, Beta, g_X, nodevec,id,corstr):
    Iu = I_U(m, U, nodevec)
    Z = Z.reshape(-1, 1)
    g_X = g_X.reshape(-1, 1)
    De = De.reshape(-1, 1)
    K = len(np.unique(id))
    id_train_series = pd.Series(id)
    n = id_train_series.value_counts().sort_index().values
    def LF(*args):
        a = args[0]
        mu = np.exp(Z * Beta + g_X)
        S1 = De - mu * np.dot(Iu, a).reshape(-1, 1)

        # Initialize G and C as PyTorch tensors
        if corstr == "independence":
            G = np.zeros((Z.shape[1], 1))
            C = np.zeros((Z.shape[1], Z.shape[1]))
        else:
            G = np.zeros((2 * Z.shape[1], 1))
            C = np.zeros((2 * Z.shape[1], 2 * Z.shape[1]))

        for i in range(K):
            M1 = np.diag(np.ones(n[i]))  # Diagonal matrix for M1

            if corstr == "exchangeable":
                M2 = np.ones((n[i], n[i]))
                np.fill_diagonal(M2, 0)
            elif corstr == "AR1":
                M2 = np.zeros((n[i], n[i]))
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
            D1 = np.diag(mu_tensor) @ Z_tensor  # Matrix multiplication

            mu_diag = np.diag(mu_tensor)  # Create diagonal matrix
            mu_sqrt = np.sqrt(mu_diag)  # Square root of the diagonal matrix
            mu_sqrt_inv = np.linalg.inv(mu_sqrt)  # Pseudoinverse (torch.inverse)

            V1 = mu_sqrt_inv @ M1 @ mu_sqrt_inv  # Matrix multiplication

            G_1 = D1.T @ V1 @ S1[id == (i + 1)]

            if corstr != "independence":
                V2 = mu_sqrt_inv @ M2 @ mu_sqrt_inv  # Matrix multiplication for V2
                G_1 = np.vstack([G_1, np.matmul(D1.T, np.matmul(V2, S1[id == (i + 1)]))     ])

            G += G_1
            C += np.matmul(G_1, G_1.T)

        Loss_F1 = np.matmul(G.T, np.linalg.inv(C)) @ G  # Matrix multiplication for the final loss function

        return Loss_F1
    bnds = []
    for i in range(m+3):
        bnds.append((0,1000))
    result = spo.minimize(LF,np.ones(m+3),method='SLSQP',bounds=bnds)
    return result['x']