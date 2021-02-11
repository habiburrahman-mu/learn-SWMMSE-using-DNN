import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
from channel import channel

import time

N = 10
num_train = 100000
num_test = 10000
trainseed = 7
testseed = 3

Xtrain, Ytrain, wtime = channel(N, num_train, seed=trainseed)
X, Y, swmmsetime = channel(N, num_test, seed=testseed)

sio.savemat('data/Train_data_%d_%d.mat' %(N, num_train), {'Xtrain': Xtrain, 'Ytrain': Ytrain})
print("Train data Saved")
sio.savemat('data/Test_data_%d_%d.mat' %(N, num_test), {'X': X, 'Y': Y, 'swmmsetime': swmmsetime})
print("Test data Saved")