import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')

import numpy as np

from modules import Dense, Sequential, Sigmoid, ReLU, SoftMax
from criterions import MSECriterion
from optimizers import sgd_momentum

from utils.data_generator import generate_cat_eye, generate_spirale, generate_two_classes
from utils.batch_generator import get_batches
from utils.metrics import accuracy_score

###############################
# Use this example to debug your code, start with logistic regression and then 
# test other layers. You do not need to change anything here. This code is 
# provided for you to test the layers. Next you will use similar code in MNIST task.
###############################


###############################
#### generate_data
X,Y = generate_two_classes(500)
print("Data dimenstions: ", X.shape, Y.shape)
# plt.scatter(X[:,0], X[:,1], c=Y.argmax(axis=-1))
# plt.show()

###############################
#### build model
net = Sequential()
net.add(Dense(2, 4))
net.add(ReLU())
net.add(Dense(4, 2))
net.add(SoftMax())

criterion = MSECriterion() # loss function

# Optimizer params
optimizer_config = {'learning_rate' : 1e-2, 'momentum': 0.9}
optimizer_state = {}

# Looping params
n_epoch = 20
batch_size = 128

##############################
#### Training loop

loss_history = []
acc_history = []

for i in range(n_epoch):
    print(f"EPOCH: {i}")
    for x_batch, y_batch in get_batches(X, Y, batch_size):
        net.zeroGradParameters()
        
        # Forward
        predictions = net.forward(x_batch)
        loss = criterion.forward(predictions, y_batch)
    
        # Backward
        dp = criterion.backward(predictions, y_batch)
        net.backward(x_batch, dp)
        
        # Update weights
        sgd_momentum(net.getParameters(), 
                     net.getGradParameters(), 
                     optimizer_config,
                     optimizer_state)      
        
        loss_history.append(loss)
        acc_history.append(100*
            accuracy_score(predictions.argmax(axis=-1), y_batch.argmax(axis=-1)))

################################
#### Visualize training statistics
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.title("Training loss")
plt.xlabel("#iteration")
plt.ylabel("loss")
plt.plot(loss_history, 'b')

plt.subplot(122)
plt.title("Training accuracy")
plt.xlabel("#iteration")
plt.ylabel("acc")
plt.plot(acc_history, 'r')

plt.show()

print('Current loss: %f' % loss)    

#####################################
#### Visualize class boundaries

h = 0.02

x_min = X[:, 0].min() - 0.5
x_max =  X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
X0, Y0 = np.meshgrid(np.arange(x_min, x_max, h),
                 np.arange(y_min, y_max, h))

X0flat = X0.reshape(-1)
Y0flat = Y0.flatten()

X1 = np.stack([X0flat, Y0flat]).T

net.evaluate()
c = np.argmax(net.forward(X1), axis=-1).reshape(X0.shape)

plt.figure(figsize=(10,8))
plt.contourf(X0, Y0, c, cmap="jet", alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=Y.argmax(axis=-1), s=40, cmap="jet", edgecolors="white")
plt.xlim(X0.min(), X0.max())
plt.ylim(Y0.min(), Y0.max())
plt.gca().set_aspect("equal")
plt.show()