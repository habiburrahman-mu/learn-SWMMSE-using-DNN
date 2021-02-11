import scipy.io as sio
import numpy as np

import tensorflow as tf
import time

N = 10
num_train = 100000
num_test = 10000
epochs = 100
batch_size = 256
learning_rate = 0.0005

var = 1

load = sio.loadmat('data/Train_data_%d_%d.mat' % (N, num_train))
loadTest = sio.loadmat('data/Test_data_%d_%d.mat' % (N, num_test))
Htrain = load['Xtrain']
Ptrain = load['Ytrain']
H_test = loadTest['Xtest']
P_test = loadTest['Ytest']
timeW = loadTest['swmmsetime']
swmmsetime = timeW[0, 0]

weights = {
    'w_1': tf.Variable(tf.random_normal([N*N, 100], stddev=0.1)),
    'w_2': tf.Variable(tf.random_normal([100, 100], stddev=0.1)),
    'w_3': tf.Variable(tf.random_normal([100, 100], stddev=0.1)),
    'w_out': tf.Variable(tf.random_normal([100, N])),
}

biases = {

    'b_1': tf.Variable(tf.random_normal([100], stddev=0.1)),
    'b_2': tf.Variable(tf.random_normal([100], stddev=0.1)),
    'b_3': tf.Variable(tf.random_normal([100], stddev=0.1)),
    'b_out': tf.Variable(tf.random_normal([N])),
}

def network(input_data):
    h_layer_1 = tf.add(tf.matmul(input_data, weights['w_1']), biases['b_1'])  #h1 = wx+b
    h_layer_1 = tf.nn.relu(h_layer_1)

    h_layer_2 = tf.add(tf.matmul(h_layer_1, weights['w_2']), biases['b_1'])
    h_layer_2 = tf.nn.relu(h_layer_2)

    h_layer_3 = tf.add(tf.matmul(h_layer_2, weights['w_3']), biases['b_3'])
    h_layer_3 = tf.nn.relu(h_layer_3)

    output = tf.matmul(h_layer_3, weights['w_out']) + biases['b_out']
    output = tf.nn.relu6(output) / 6

    return output

valid_split = 0.1

total_sample_size = num_train
validation_sample_size = int(total_sample_size*valid_split)
training_sample_size = total_sample_size - validation_sample_size

Htrain = np.reshape(Htrain, (total_sample_size, N*N))
number_input = N*N
number_output = N

H_train = Htrain[ 0:training_sample_size, :]
P_train = Ptrain[0:training_sample_size, :]
H_val = Htrain[training_sample_size:total_sample_size, :]
P_val = Ptrain[training_sample_size:total_sample_size, :]

x = tf.placeholder(tf.float32, [None, N*N])
y = tf.placeholder(tf.float32, [None, N])

total_batch = int(total_sample_size / batch_size)

x_bar = network(x)
loss = tf.reduce_mean(tf.square(x_bar - y))

optimizer = tf.train.AdamOptimizer(learning_rate)
objective = optimizer.minimize(loss)

#init = tf.global_variables_initializer()

save_data = np.zeros((epochs, 3))

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

#session.run(init)

start_time = time.time()
for e in range(epochs):
    for b in range(total_batch):
        batch = np.random.randint(total_sample_size, size=batch_size)
        _, training_cost = session.run([objective, loss], feed_dict={x: H_train, y: P_train})
        save_data[e, 0] = training_cost
    validation_cost = session.run(loss, feed_dict={x: H_val, y: P_val})
    save_data[e, 1] = validation_cost
    save_data[e, 2] = e
    if (e % 2 == 0):
        print('\n %d ' % e, ' %f ' % (training_cost), ' %f ' % (validation_cost), ' %f ' % (time.time() - start_time), )
    else:
        print("#", end="")
print("training time: %0.2f s" % (time.time() - start_time))
sio.savemat('MSETime_%d_%d_%d' % (N, batch_size, learning_rate * 10000), {'train': save_data[:, 0], 'validation': save_data[:, 1], 'epoch': save_data[:, 2]})

start_time = time.time()
H_test_r = np.reshape(H_test, (num_test, N*N))
predicted_power = session.run(x_bar, feed_dict={x: H_test_r, y: P_test})
pred_time = time.time()-start_time
print(predicted_power.shape)
predicted_power = np.reshape(predicted_power,(num_test,N))

def IC_sum_rate(H, p, var_noise):
    H = np.square(H)
    fr = np.diag(H)*p
    ag = np.dot(H,p) + var_noise - fr
    y = np.sum(np.log(1+fr/ag) )
    return y
def np_sum_rate(X,Y):
    avg = 0
    n = X.shape[0]
    for i in range(n):
        avg += IC_sum_rate(X[i,:,:],Y[i,:],1)/n
    return avg

sum_rate_dnn = np_sum_rate(H_test,predicted_power)*np.log2(np.exp(1))
sum_rate_swmmse = np_sum_rate(H_test,P_test)*np.log2(np.exp(1))

print('sum rate for DNN', sum_rate_dnn)
print('sum rate for SWMMSE', sum_rate_swmmse)
print("%f%% in %f sec time" % (sum_rate_dnn/sum_rate_swmmse*100, pred_time))
print(swmmsetime)