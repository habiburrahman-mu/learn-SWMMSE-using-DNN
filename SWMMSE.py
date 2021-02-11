import numpy as np
import math
def SWMMSE(p_int, H, Pmax, var_noise):
    N = np.size(p_int)
    vnew = 0
    v = np.sqrt(p_int)
    u = np.zeros(N)
    w = np.zeros(N)
    a = np.zeros(N)
    b = np.zeros(N)
    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        vnew = 0
        for i in range(N):
            u[i] = H[i, i] * v[i] / ((np.square(H[i, :])) @ (np.square(v)) + var_noise)
            w[i] = 1 / (1 - u[i] * v[i] * H[i, i])
            vnew = vnew + math.log2(w[i])
        for i in range(N):
            a[i] = a[i] + sum(w * np.square(u) * np.square(H[:, i]))
            b[i] = b[i] + w[i] * u[i] * H[i, i]
            btmp = b[i]/a[i]
            v[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp
        VV[iter] = vnew
        if vnew - vold <= 1e-2:
            break
    p_opt = np.square(v)
    return p_opt