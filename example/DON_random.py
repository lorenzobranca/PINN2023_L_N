### easy test for deeponet

#########################

#du/dt = k*sin(t) ; u(0) = 0

########################

import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt

# Define ODE solution and operator.
def u(t, k):
    return k[:,np.newaxis] @ np.cos(t)[np.newaxis,:]
def v(t,k):
    return k[:,np.newaxis] @ np.sin(t)[np.newaxis,:]

# Define the dimension of the problem.
len_t = 50
len_k = 100

# Define the grid for the cartesian product.
k = np.random.uniform(-1,1,len_k)
t = np.sort(np.random.uniform(0,2*np.pi,len_t))
t_prime = np.sort(np.random.uniform(0,2*np.pi,1))

# Prepare the dataset.
X_branch = v(t,k)
X_trunc = (np.expand_dims(t,axis=-1))
Y = u(t_prime,k)

# Divide between train and test dataset.
y_train = (Y[:20,:]) 
x_train = ((X_branch[:20,:]),(X_trunc))

y_test = (Y[20:,:])
x_test = ((X_branch[20:,:]),(X_trunc))

# Format the data.
data = dde.data.Triple(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test) 
# Build the DeepONet.
m = len_t
dim_x = 1

net = dde.nn.DeepONet(
        [m,40,40],
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
