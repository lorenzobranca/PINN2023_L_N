### easy test for deeponet

#########################

#du/dt = k*sin(t) ; u(0) = 0


########################

import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt

def sol(t, k):

    return k*np.cos(t)


my_data  = np.zeros((100,50))
x_branch = np.zeros((100,50))

k_space = np.random.uniform(-1,1,100)
t_space = np.sort(np.random.uniform(0,2*np.pi,50))


#t_space = (np.expand_dims(t_space, axis=-1))
count=0
for k_ in k_space:

    my_data[ count, :]  = k_*np.cos(t_space)
    x_branch[count, :]  = k_*np.sin(t_space) 
    count += 1

t_space = (np.expand_dims(t_space, axis=-1))

y_train = (my_data[0:20,:])
x_train = ((x_branch[0:20,:]), (t_space))

y_test = (my_data[20:,:])
x_test = ((x_branch[20:,:]), (t_space))

print(np.shape((y_train)))
data = dde.data.TripleCartesianProd(X_train=x_train, y_train=y_train, X_test=x_test, 
        y_test=y_test)

m = 50
dim_x = 1
net = dde.nn.DeepONetCartesianProd(
    [m, 40, 40],
    [dim_x, 40, 40],
    "relu",
    "Glorot normal",
)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=10000)

dde.utils.plot_loss_history(losshistory)
plt.savefig('easy_test.png')
