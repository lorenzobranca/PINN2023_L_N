### easy test for deeponet

#########################

#du/dt = k*sin(t) ; u(0) = 0

########################

import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt

# Define ODE solution and operator.
def u(t, k, flag="chart"):
    if flag == "chart":
        return k[:,np.newaxis] * np.cos(t)[np.newaxis,:]
    if flag == "scalar":
        return ( k * np.cos(t) )[:,np.newaxis]
def v(t,k,flag="chart"):
    return k[:,np.newaxis] * np.sin(t)[np.newaxis,:]

# Define the dimension of the problem.
len_t = 100
len_k = 110000

# Define the grid for the cartesian product.
k = np.random.uniform(-1,1,len_k)
t = np.sort(np.random.uniform(0,2*np.pi,len_t))
t_prime = np.random.uniform(0,2*np.pi,len_k)

# Prepare the dataset.
X_branch = v(t,k)
X_trunc = ( np.expand_dims( t_prime , axis = -1 ) )
Y = u(t_prime,k,flag="scalar")

# Divide between train and test dataset
len_train = 10000
y_train = (Y[:len_train,:]) 
x_train = ((X_branch[:len_train,:]),(X_trunc[:len_train,:]))

y_test = (Y[len_train:,:])
x_test = ((X_branch[len_train:,:]),(X_trunc[len_train:,:]))

# Format the data.
data = dde.data.Triple(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test) 
# Build the DeepONet.
m = len_t
dim_x = 1

net = dde.nn.DeepONet(
        [m, 40, 40],
        [dim_x, 40, 40],
        "relu",
        "Glorot normal",
)
# Define, compile and train the model.
model = dde.Model(data,net)
model.compile("adam",lr=0.001)
losshistory, train_state = model.train(iterations=10000)

# Plot result.
dde.utils.plot_loss_history(losshistory)
plt.savefig("easy_test_unanligned.png")

