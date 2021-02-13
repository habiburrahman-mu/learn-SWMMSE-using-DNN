import numpy as np
import time
from SWMMSE import SWMMSE

def channel(N, num_train, Pmax=1, Pmin=0, var_noise=1, seed=1758):
    print('Generate Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    Pini = Pmax * np.ones(N)
    X = np.zeros((N ** 2, num_train))
    Y = np.zeros((num_train, N ))
    X_t = np.zeros((num_train, N, N))
    total_time = 0.0
    for loop in range(num_train):
        CH = 1 / np.sqrt(2) * (np.random.randn(N, N) + 1j * np.random.randn(N, N))
        H = abs(CH)
        X[:, loop] = np.reshape(H, (N ** 2,), order="F")
        H = np.reshape(X[:, loop], (N, N), order="F")
        X_t[loop, :, :] = H
        mid_time = time.time()
        Y[loop, :] = SWMMSE(Pini, H, Pmax, var_noise)
        total_time = total_time + time.time() - mid_time
    return X_t, Y, total_time